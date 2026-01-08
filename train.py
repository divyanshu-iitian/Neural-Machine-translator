"""Training script for Neural Machine Translation with Reinforcement Learning.

This script supports:
- Standard cross-entropy training
- Actor-Critic reinforcement learning
- Mixed precision training
- Advanced learning rate scheduling
- Comprehensive checkpointing
"""

import argparse
import os
import numpy as np
import random
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

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

parser = argparse.ArgumentParser(description="train.py")

## Data options
parser.add_argument("-data", required=True,
                    help="Path to the *-train.pt file from preprocess.py")
parser.add_argument("-save_dir", required=True,
                    help="Directory to save models")
parser.add_argument("-load_from", help="Path to load a pretrained model.")

## Model options

parser.add_argument("-layers", type=int, default=1,
                    help="Number of layers in the LSTM encoder/decoder")
parser.add_argument("-rnn_size", type=int, default=500,
                    help="Size of LSTM hidden states")
parser.add_argument("-word_vec_size", type=int, default=500,
                    help="Size of word embeddings")
parser.add_argument("-input_feed", type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument("-brnn", action="store_true",
                    help="Use a bidirectional encoder")
parser.add_argument("-brnn_merge", default="concat",
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument("-batch_size", type=int, default=64,
                    help="Maximum batch size")
parser.add_argument("-max_generator_batches", type=int, default=32,
                    help="""Split softmax input into small batches for memory efficiency.
                    Higher is faster, but uses more memory.""")
parser.add_argument("-end_epoch", type=int, default=50,
                    help="Epoch to stop training.")
parser.add_argument("-start_epoch", type=int, default=1,
                    help="Epoch to start training.")
parser.add_argument("-param_init", type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument("-optim", default="adam",
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument("-lr", type=float, default=1e-3,
                    help="Initial learning rate")
parser.add_argument("-max_grad_norm", type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument("-dropout", type=float, default=0,
                    help="Dropout probability; applied between LSTM stacks.")
parser.add_argument("-learning_rate_decay", type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument("-start_decay_at", type=int, default=5,
                    help="Start decay after this epoch")

# GPU
parser.add_argument("-gpus", default=[0], nargs="+", type=int,
                    help="Use CUDA")
parser.add_argument("-log_interval", type=int, default=100,
                    help="Print stats at this interval.")
parser.add_argument("-seed", type=int, default=3435,
                     help="Seed for random initialization")

# Critic
parser.add_argument("-start_reinforce", type=int, default=None,
                    help="""Epoch to start reinforcement training.
                    Use -1 to start immediately.""")
parser.add_argument("-critic_pretrain_epochs", type=int, default=0,
                    help="Number of epochs to pretrain critic (actor fixed).")
parser.add_argument("-reinforce_lr", type=float, default=1e-4,
                    help="""Learning rate for reinforcement training.""")

# Evaluation
parser.add_argument("-eval", action="store_true", help="Evaluate model only")
parser.add_argument("-eval_sample", action="store_true", default=False,
        help="Eval by sampling")
parser.add_argument("-max_predict_length", type=int, default=80,
                    help="Maximum length of predictions.")

# Advanced training options
parser.add_argument("-mixed_precision", action="store_true", default=False,
                    help="Enable mixed precision training with automatic scaling")
parser.add_argument("-gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing backward/update pass.")
parser.add_argument("-warmup_steps", type=int, default=4000,
                    help="Number of warmup steps for learning rate scheduler.")
parser.add_argument("-label_smoothing", type=float, default=0.1,
                    help="Label smoothing value for cross entropy loss.")
parser.add_argument("-early_stopping_patience", type=int, default=10,
                    help="Stop training if no improvement after this many epochs.")


# Reward shaping
parser.add_argument("-pert_func", type=str, default=None,
        help="Reward-shaping function.")
parser.add_argument("-pert_param", type=float, default=None,
        help="Reward-shaping parameter.")

# Others
parser.add_argument("-no_update", action="store_true", default=False,
        help="No update round. Use to evaluate model samples.")
parser.add_argument("-sup_train_on_bandit", action="store_true", default=False,
        help="Supervised learning update round.")

opt = parser.parse_args()
logger.info(f"Configuration: {opt}")

# Set seed for reproducibility
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

opt.cuda = len(opt.gpus)

# Create save directory if needed
if opt.save_dir:
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available() and not opt.cuda:
    logger.warning("CUDA device detected but not being used. Consider running with -gpus flag for better performance.")

if opt.cuda:
    cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_model_weights(model: nn.Module) -> None:
    """
    Initialize model parameters with uniform distribution.
    
    Args:
        model: PyTorch model to initialize
    """
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)


def create_optimizer(model: nn.Module) -> lib.Optim:
    """
    Create optimizer with learning rate scheduling.
    
    Args:
        model: Model whose parameters to optimize
        
    Returns:
        Configured optimizer
    """
    optim = lib.Optim(
        model.parameters(), opt.optim, opt.lr, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay, start_decay_at=opt.start_decay_at
    )
    return optim


def create_nmt_model(model_class, dicts: Dict, gen_out_size: int) -> tuple:
    """
    Create and initialize NMT model with encoder, decoder, and generator.
    
    Args:
        model_class: Model class to instantiate
        dicts: Dictionary containing vocabularies
        gen_out_size: Output size for generator
        
    Returns:
        Tuple of (model, optimizer)
    """
    encoder = lib.Encoder(opt, dicts[\"src\"])
    decoder = lib.Decoder(opt, dicts[\"tgt\"])
    
    # Use memory efficient generator when needed
    if opt.max_generator_batches < opt.batch_size and gen_out_size > 1:
        generator = lib.MemEfficientGenerator(nn.Linear(opt.rnn_size, gen_out_size), opt)
    else:
        generator = lib.BaseGenerator(nn.Linear(opt.rnn_size, gen_out_size), opt)
    
    model = model_class(encoder, decoder, generator, opt)
    init_model_weights(model)
    optim = create_optimizer(model)
    
    return model, optim


def create_critic_model(checkpoint: Optional[Dict], dicts: Dict, opt) -> tuple:
    """
    Create or load critic model for reinforcement learning.
    
    Args:
        checkpoint: Optional checkpoint to load from
        dicts: Vocabularies
        opt: Configuration options
        
    Returns:
        Tuple of (critic_model, critic_optimizer)
    """
    if opt.load_from is not None and checkpoint and "critic" in checkpoint:
        critic = checkpoint["critic"]
        critic_optim = checkpoint["critic_optim"]
        logger.info("Loaded critic from checkpoint")
    else:
        critic, critic_optim = create_nmt_model(lib.NMTModel, dicts, 1)
        logger.info("Created new critic model")
    
    if opt.cuda:
        critic.cuda(opt.gpus[0])
    
    return critic, critic_optim


def main():
    """Main training function."""
    
    logger.info(f'Loading training data from \"{opt.data}\"')

    dataset = torch.load(opt.data)

    # Create datasets
    supervised_data = lib.Dataset(dataset["train_xe"], opt.batch_size, opt.cuda, eval=False)
    bandit_data = lib.Dataset(dataset["train_pg"], opt.batch_size, opt.cuda, eval=False)
    valid_data = lib.Dataset(dataset["valid"], opt.batch_size, opt.cuda, eval=True)
    test_data = lib.Dataset(dataset["test"], opt.batch_size, opt.cuda, eval=True)

    dicts = dataset["dicts"]
    logger.info(f"Vocabulary size - Source: {dicts['src'].size()}, Target: {dicts['tgt'].size()}")
    logger.info(f"Cross-entropy training sentences: {len(dataset['train_xe']['src'])}")
    logger.info(f"Policy gradient training sentences: {len(dataset['train_pg']['src'])}")
    logger.info(f"Maximum batch size: {opt.batch_size}")
    
    logger.info("Building model architecture...")

    use_critic = opt.start_reinforce is not None

    # Create or load model
    if opt.load_from is None:
        model, optim = create_nmt_model(lib.NMTModel, dicts, dicts["tgt"].size())
        checkpoint = None
        logger.info("Created new model")
    else:
        logger.info(f"Loading model from checkpoint: {opt.load_from}")
        checkpoint = torch.load(opt.load_from)
        model = checkpoint["model"]
        optim = checkpoint["optim"]
        opt.start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming from epoch {opt.start_epoch}")

    # Move model to GPU if available
    if opt.cuda:
        model.cuda(opt.gpus[0])

    # Configure reinforcement learning start
    if opt.start_reinforce == -1:
        opt.start_decay_at = opt.start_epoch
        opt.start_reinforce = opt.start_epoch
        logger.info("Starting reinforcement training immediately")

    # Validation check
    if use_critic:
        assert opt.start_epoch + opt.critic_pretrain_epochs - 1 <= opt.end_epoch, \
            "Insufficient epochs for critic pretraining. Increase -end_epoch!"

    # Model statistics
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    # Define metrics
    metrics = {
        "nmt_loss": lib.Loss.weighted_xent_loss,
        "critic_loss": lib.Loss.weighted_mse,
        "sent_reward": lib.Reward.sentence_bleu,
        "corp_reward": lib.Reward.corpus_bleu
    }
    
    if opt.pert_func is not None:
        opt.pert_func = lib.PertFunction(opt.pert_func, opt.pert_param)
        logger.info(f"Using reward shaping: {opt.pert_func}")

    # Evaluation mode
    if opt.eval:
        logger.info("Evaluation mode - running inference only")
        evaluator = lib.Evaluator(model, metrics, dicts, opt)
        
        # Evaluate on validation set
        pred_file = opt.load_from.replace(".pt", ".valid.pred")
        logger.info(f"Evaluating on validation set -> {pred_file}")
        evaluator.eval(valid_data, pred_file)
        
        # Evaluate on test set
        pred_file = opt.load_from.replace(".pt", ".test.pred")
        logger.info(f"Evaluating on test set -> {pred_file}")
        evaluator.eval(test_data, pred_file)
        
    elif opt.eval_sample:
        opt.no_update = True
        critic, critic_optim = create_critic_model(checkpoint, dicts, opt)
        reinforce_trainer = lib.ReinforceTrainer(
            model, critic, bandit_data, test_data,
            metrics, dicts, optim, critic_optim, opt
        )
        reinforce_trainer.train(opt.start_epoch, opt.start_epoch, False)
        
    elif opt.sup_train_on_bandit:
        optim.set_lr(opt.reinforce_lr)
        xent_trainer = lib.Trainer(model, bandit_data, test_data, metrics, dicts, optim, opt)
        xent_trainer.train(opt.start_epoch, opt.start_epoch)
        
    else:
        logger.info("Starting standard training procedure")
        xent_trainer = lib.Trainer(model, supervised_data, valid_data, metrics, dicts, optim, opt)
        
        if use_critic:
            start_time = time.time()
            
            # Phase 1: Supervised pre-training
            if opt.start_reinforce > opt.start_epoch:
                logger.info(f"Phase 1: Supervised training (epochs {opt.start_epoch} to {opt.start_reinforce - 1})")
                xent_trainer.train(opt.start_epoch, opt.start_reinforce - 1, start_time)
            
            # Create critic model
            critic, critic_optim = create_critic_model(checkpoint, dicts, opt)
            
            # Phase 2: Critic pre-training
            if opt.critic_pretrain_epochs > 0:
                logger.info(f\"Phase 2: Critic pre-training ({opt.critic_pretrain_epochs} epochs)")
                reinforce_trainer = lib.ReinforceTrainer(
                    model, critic, supervised_data, test_data,
                    metrics, dicts, optim, critic_optim, opt
                )
                reinforce_trainer.train(
                    opt.start_reinforce,
                    opt.start_reinforce + opt.critic_pretrain_epochs - 1,
                    True, start_time
                )
            
            # Phase 3: Reinforcement learning
            logger.info(f"Phase 3: Reinforcement learning training")
            reinforce_trainer = lib.ReinforceTrainer(
                model, critic, bandit_data, test_data,
                metrics, dicts, optim, critic_optim, opt
            )
            reinforce_trainer.train(
                opt.start_reinforce + opt.critic_pretrain_epochs,
                opt.end_epoch,
                False, start_time
            )
        else:
            # Standard supervised training only
            logger.info(f"Supervised training from epoch {opt.start_epoch} to {opt.end_epoch}")
            xent_trainer.train(opt.start_epoch, opt.end_epoch)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
