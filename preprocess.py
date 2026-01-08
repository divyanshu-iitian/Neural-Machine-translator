"""
Data Preprocessing Pipeline for Neural Machine Translation

This module handles:
- Vocabulary building with frequency-based pruning
- Parallel corpus processing
- Sequence length filtering  
- Train/validation/test set preparation
- BPE/Word-level tokenization support
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import torch

import lib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="preprocess.py")

parser.add_argument("-train_src", required=True,
                    help="Path to the training source data")
parser.add_argument("-train_tgt", required=True,
                    help="Path to the training target data")

parser.add_argument("-train_xe_src", required=True,
                    help="Path to the pre-training source data")
parser.add_argument("-train_xe_tgt", required=True,
                    help="Path to the pre-training target data")

parser.add_argument("-train_pg_src", required=True,
                    help="Path to the bandit training source data")
parser.add_argument("-train_pg_tgt", required=True,
                    help="Path to the bandit training target data")

parser.add_argument("-valid_src", required=True,
                    help="Path to the validation source data")
parser.add_argument("-valid_tgt", required=True,
                     help="Path to the validation target data")

parser.add_argument("-test_src", required=True,
                    help="Path to the test source data")
parser.add_argument("-test_tgt", required=True,
                     help="Path to the test target data")

parser.add_argument("-save_data", required=True,
                    help="Output file for the prepared data")

parser.add_argument("-src_vocab_size", type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument("-tgt_vocab_size", type=int, default=50000,
                    help="Size of the target vocabulary")

parser.add_argument("-seq_length", type=int, default=80,
                    help="Maximum sequence length")
parser.add_argument("-seed",       type=int, default=3435,
                    help="Random seed")

parser.add_argument("-report_every", type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()
torch.manual_seed(opt.seed)
logger.info(f"Preprocessing configuration: {opt}")


def build_vocabulary(filename: str, size: int) -> lib.Dict:
    """
    Build vocabulary from text file with frequency-based pruning.
    
    Args:
        filename: Path to input text file
        size: Maximum vocabulary size
        
    Returns:
        Vocabulary dictionary object
    """
    vocab = lib.Dict([
        lib.Constants.PAD_WORD, 
        lib.Constants.UNK_WORD,
        lib.Constants.BOS_WORD, 
        lib.Constants.EOS_WORD
    ])

    logger.info(f"Reading vocabulary from: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip().split():
                vocab.add(word.lower())  # Normalize to lowercase

    original_size = vocab.size()
    vocab = vocab.prune(size)
    
    logger.info(f"Vocabulary created: {vocab.size()} tokens (pruned from {original_size})")

    return vocab


def initialize_vocabulary(name: str, data_file: str, vocab_size: int, save_file: str) -> lib.Dict:
    """
    Create and save vocabulary from corpus.
    
    Args:
        name: Vocabulary name (for logging)
        data_file: Source data file
        vocab_size: Maximum vocabulary size
        save_file: Output file path
        
    Returns:
        Vocabulary object
    """
    logger.info(f"Building {name} vocabulary...")
    vocab = build_vocabulary(data_file, vocab_size)
    
    logger.info(f"Saving {name} vocabulary to: {save_file}")
    vocab.writeFile(save_file)
    
    return vocab


def process_parallel_data(which: str, src_file: str, tgt_file: str, 
                          src_dicts: lib.Dict, tgt_dicts: lib.Dict) -> Tuple[List, List, List]:
    """
    Process parallel corpus with length filtering and indexing.
    
    Args:
        which: Dataset split name ('train_xe', 'train_pg', 'valid', 'test')
        src_file: Source language file
        tgt_file: Target language file
        src_dicts: Source vocabulary
        tgt_dicts: Target vocabulary
        
    Returns:
        Tuple of (source_indices, target_indices, positions)
    """
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    logger.info(f"Processing {which}: {src_file} & {tgt_file}")
    
    with open(src_file, 'r', encoding='utf-8') as srcF, \
         open(tgt_file, 'r', encoding='utf-8') as tgtF:
        
        while True:
            src_line = srcF.readline()
            tgt_line = tgtF.readline()
            
            if not src_line or not tgt_line:
                if src_line != tgt_line:
                    logger.warning("Source and target files have different lengths!")
                break
            
            src_words = src_line.strip().split()
            tgt_words = tgt_line.strip().split()
            
            # Length filtering
            if len(src_words) <= opt.seq_length and len(tgt_words) <= opt.seq_length:
                src.append(src_dicts.convertToIdx(src_words, lib.Constants.UNK_WORD))
                tgt.append(tgt_dicts.convertToIdx(tgt_words, lib.Constants.UNK_WORD,
                                                   eosWord=lib.Constants.EOS_WORD))
                sizes.append(len(src_words))
            else:
                # Keep all test data regardless of length
                if which == "test":
                    src.append(src_dicts.convertToIdx(src_words, lib.Constants.UNK_WORD))
                    tgt.append(tgt_dicts.convertToIdx(tgt_words, lib.Constants.UNK_WORD,
                                                       eosWord=lib.Constants.EOS_WORD))
                    sizes.append(len(src_words))
                else:
                    ignored += 1

            count += 1
            if count % opt.report_every == 0:
                logger.info(f"Processed {count} sentence pairs...")

    assert len(src) == len(tgt), "Source and target lengths must match!"
    
    logger.info(f"Prepared {len(src)} sentences ({ignored} ignored due to length > {opt.seq_length})")

    return src, tgt, list(range(len(src)))


def prepare_dataset(which: str, src_path: str, tgt_path: str, dicts: Dict) -> Dict:
    """
    Prepare dataset split with source and target processing.
    
    Args:
        which: Dataset name
        src_path: Source file path
        tgt_path: Target file path
        dicts: Vocabularies dictionary
        
    Returns:
        Dictionary containing processed data
    """
    logger.info(f"Preparing {which} dataset...")
    res = {}
    res["src"], res["tgt"], res["pos"] = process_parallel_data(
        which, src_path, tgt_path, dicts["src"], dicts["tgt"]
    )
    return res


def main():
    """Main preprocessing pipeline."""
    
    logger.info("=" * 80)
    logger.info("Starting NMT Data Preprocessing Pipeline")
    logger.info("=" * 80)
    
    # Build vocabularies
    dicts = {}
    dicts["src"] = initialize_vocabulary(
        "source", opt.train_src, opt.src_vocab_size,
        opt.save_data + ".src.dict"
    )
    dicts["tgt"] = initialize_vocabulary(
        "target", opt.train_tgt, opt.tgt_vocab_size,
        opt.save_data + ".tgt.dict"
    )

    # Prepare all dataset splits
    save_data = {
        "dicts": dicts,
        "train_xe": prepare_dataset("train_xe", opt.train_xe_src, opt.train_xe_tgt, dicts),
        "train_pg": prepare_dataset("train_pg", opt.train_pg_src, opt.train_pg_tgt, dicts),
        "valid": prepare_dataset("valid", opt.valid_src, opt.valid_tgt, dicts),
        "test": prepare_dataset("test", opt.test_src, opt.test_tgt, dicts)
    }

    # Save processed data
    output_file = opt.save_data + "-train.pt"
    logger.info(f"Saving processed data to: {output_file}")
    torch.save(save_data, output_file)
    
    logger.info("=" * 80)
    logger.info("Preprocessing completed successfully!")
    logger.info("=" * 80)
    
    # Print statistics
    logger.info("Dataset Statistics:")
    for split_name in ["train_xe", "train_pg", "valid", "test"]:
        num_samples = len(save_data[split_name]["src"])
        logger.info(f"  {split_name:12s}: {num_samples:8,} sentence pairs")


if __name__ == "__main__":
    main()
