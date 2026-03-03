"""Translation script for Neural Machine Translation models.

Supports:
- Batch translation for efficiency
- Multiple architecture types (LSTM, Transformer)
- Beam search decoding
- GPU acceleration
- Progress tracking and logging
"""

import argparse
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import random
import torch
import torch.nn as nn
from torch import cuda

import lib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="translate.py - Neural Machine Translation")

# Data options
parser.add_argument("-data", required=True,
                    help="Path to the *-train.pt file from preprocess.py")
parser.add_argument("-batch_size", type=int, default=32,
                    help="Batch size for translation")
parser.add_argument("-save_dir", 
                    help="Directory to save predictions (defaults to test_src directory)")
parser.add_argument("-load_from", required=True,
                    help="Path to load a trained model checkpoint")
parser.add_argument("-test_src", required=True,
                    help="Path to the source file to be translated")

# Translation options
parser.add_argument("-beam_size", type=int, default=5,
                    help="Beam size for beam search (1 for greedy)")
parser.add_argument("-max_length", type=int, default=100,
                    help="Maximum translation length")
parser.add_argument("-n_best", type=int, default=1,
                    help="Output n-best translations")
parser.add_argument("-replace_unk", action="store_true",
                    help="Replace unknown tokens with source attention")

# GPU options
parser.add_argument("-gpus", default=[0], nargs="+", type=int,
                    help="GPU device IDs to use")
parser.add_argument("-seed", type=int, default=3435,
                    help="Random seed for reproducibility")

# Logging
parser.add_argument("-verbose", action="store_true",
                    help="Print verbose output")


opt = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

# Configure CUDA
opt.cuda = len(opt.gpus) > 0 and torch.cuda.is_available()

# Create save directory if needed
if opt.save_dir:
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
else:
    opt.save_dir = str(Path(opt.test_src).parent)

if torch.cuda.is_available() and not opt.cuda:
    logger.warning("CUDA device available but not being used. Consider using -gpus 0")

if opt.cuda:
    cuda.set_device(opt.gpus[0])
    logger.info(f"Using GPU: {torch.cuda.get_device_name(opt.gpus[0])}")


def load_test_data(src_file: str, dicts: Dict) -> Tuple[List, List, List]:
    """Load and prepare test data for translation.
    
    Args:
        src_file: Path to source text file
        dicts: Dictionary containing vocabularies
        
    Returns:
        Tuple of (source_sequences, target_sequences, positions)
    """
    logger.info(f"Loading test data from {src_file}")
    
    try:
        with open(src_file, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
    except FileNotFoundError:
        logger.error(f"Source file not found: {src_file}")
        raise
    
    src_dicts = dicts["src"]
    src = []
    
    for line in lines:
        src_words = line.strip().split()
        src_idx = src_dicts.convertToIdx(src_words, lib.Constants.UNK_WORD)
        src.append(src_idx)
    
    logger.info(f"Prepared {len(src)} sentences for translation")
    
    # For translation, target is same as source (will be ignored)
    tgt = src
    pos = list(range(len(src)))
    
    return src, tgt, pos


def translate_batch(
    model: nn.Module,
    data: lib.Dataset,
    dicts: Dict,
    opt
) -> List[List[str]]:
    """Translate all batches in the dataset.
    
    Args:
        model: Trained translation model
        data: Dataset containing source sequences
        dicts: Dictionary containing vocabularies
        opt: Command-line options
        
    Returns:
        List of translated sentences (as lists of tokens)
    """
    model.eval()
    all_predictions = []
    all_indices = []
    
    logger.info("Starting translation...")
    
    with torch.no_grad():
        for i in tqdm(range(len(data)), desc="Translating", disable=not opt.verbose):
            batch = data[i]
            
            # Get attention mask for LSTM models
            if hasattr(model.decoder, 'attn'):
                attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
                model.decoder.attn.applyMask(attention_mask)
            
            # Translate
            predictions = model.translate(batch, opt.max_length)
            predictions = predictions.t().tolist()
            
            # Get batch indices for reordering
            indices = batch[2]
            
            all_predictions.extend(predictions)
            all_indices.extend(indices)
    
    # Reorder predictions to match input order
    sorted_pairs = sorted(zip(all_predictions, all_indices), key=lambda x: x[1])
    predictions, _ = zip(*sorted_pairs)
    
    # Convert to words
    tgt_dict = dicts["tgt"]
    translations = []
    
    for pred in predictions:
        # Clean up special tokens
        pred = lib.Reward.clean_up_sentence(pred, remove_unk=False, remove_eos=True)
        # Convert indices to words
        words = [tgt_dict.getLabel(idx) for idx in pred]
        translations.append(words)
    
    return translations


def save_translations(translations: List[List[str]], output_file: str):
    """Save translations to file.
    
    Args:
        translations: List of translated sentences
        output_file: Path to output file
    """
    logger.info(f"Saving translations to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in translations:
            f.write(' '.join(sent) + '\n')
    
    logger.info(f"Successfully saved {len(translations)} translations")



def main():
    """Main translation pipeline."""
    logger.info("=" * 80)
    logger.info("Neural Machine Translation - Inference")
    logger.info("=" * 80)
    
    # Load dataset dictionaries
    logger.info(f'Loading dictionaries from "{opt.data}"')
    try:
        dataset = torch.load(opt.data)
        dicts = dataset["dicts"]
    except FileNotFoundError:
        logger.error(f"Data file not found: {opt.data}")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Load trained model
    if not os.path.exists(opt.load_from):
        logger.error(f"Model checkpoint not found: {opt.load_from}")
        return
    
    logger.info(f"Loading model from {opt.load_from}")
    try:
        checkpoint = torch.load(opt.load_from, map_location='cpu')
        model = checkpoint["model"]
        
        # Log model information
        if hasattr(model, 'encoder'):
            encoder_type = type(model.encoder).__name__
            logger.info(f"Model architecture: {encoder_type}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Move model to GPU
    if opt.cuda:
        model.cuda(opt.gpus[0])
        logger.info(f"Model moved to GPU {opt.gpus[0]}")
    
    # Load test data
    try:
        src, tgt, pos = load_test_data(opt.test_src, dicts)
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return
    
    # Create dataset
    test_dataset = {
        "src": src,
        "tgt": tgt,
        "pos": pos
    }
    test_data = lib.Dataset(test_dataset, opt.batch_size, opt.cuda, eval=False)
    
    # Translate
    try:
        translations = translate_batch(model, test_data, dicts, opt)
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    output_file = os.path.join(opt.save_dir, Path(opt.test_src).stem + ".pred")
    try:
        save_translations(translations, output_file)
    except Exception as e:
        logger.error(f"Error saving translations: {e}")
        return
    
    logger.info("=" * 80)
    logger.info("Translation completed successfully!")
    logger.info(f"Output saved to: {output_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
