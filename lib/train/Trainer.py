"""
Standard Trainer for Neural Machine Translation with Cross-Entropy Loss

Features:
- Efficient batch processing
- Learning rate scheduling
- Checkpointing and model saving
- Validation with BLEU metrics
- Progress tracking and logging
"""

import datetime
import math
import os
import time
import logging
from typing import Optional

import torch

import lib

# Setup logging
logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Handles standard supervised training with cross-entropy loss.
    
    Args:
        model: NMT model to train
        train_data: Training dataset
        eval_data: Validation dataset
        metrics: Dictionary of metric functions
        dicts: Vocabularies
        optim: Optimizer
        opt: Configuration options
    """
    
    def __init__(self, model, train_data, eval_data, metrics, dicts, optim, opt):

        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.evaluator = lib.Evaluator(model, metrics, dicts, opt)
        self.loss_func = metrics["nmt_loss"]
        self.dicts = dicts
        self.optim = optim
        self.opt = opt
        self.best_valid_loss = float('inf')
        self.patience_counter = 0

        logger.info(f"Trainer initialized")
        logger.info(f"Model architecture:\n{model}")

    def train(self, start_epoch: int, end_epoch: int, start_time: Optional[float] = None):
        """
        Main training loop across multiple epochs.
        
        Args:
            start_epoch: Starting epoch number
            end_epoch: Final epoch number
            start_time: Optional start time for tracking
        """
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
            
        logger.info(f"Starting training from epoch {start_epoch} to {end_epoch}")
        
        for epoch in range(start_epoch, end_epoch + 1):
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Cross-Entropy Training - Epoch {epoch}/{end_epoch}")
            logger.info("=" * 80)
            logger.info(f"Learning rate: {self.optim.lr:.6f}")
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            train_ppl = math.exp(min(train_loss, 100))
            logger.info(f'Training perplexity: {train_ppl:.2f}')

            # Validation
            valid_loss, valid_sent_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            
            logger.info(f'Validation perplexity: {valid_ppl:.2f}')
            logger.info(f'Validation sentence-level BLEU: {valid_sent_reward * 100:.2f}')
            logger.info(f'Validation corpus-level BLEU: {valid_corpus_reward * 100:.2f}')

            # Learning rate scheduling
            self.optim.updateLearningRate(valid_loss, epoch)

            # Save checkpoint
            checkpoint = {
                'model': self.model,
                'dicts': self.dicts,
                'opt': self.opt,
                'epoch': epoch,
                'optim': self.optim,
                'valid_loss': valid_loss,
                'valid_ppl': valid_ppl,
                'valid_bleu': valid_corpus_reward
            }
            
            model_name = os.path.join(self.opt.save_dir, f"model_{epoch}.pt")
            torch.save(checkpoint, model_name)
            logger.info(f"Checkpoint saved: {model_name}")
            
            # Early stopping check
            if hasattr(self.opt, 'early_stopping_patience'):
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.patience_counter = 0
                    # Save best model
                    best_model_name = os.path.join(self.opt.save_dir, "model_best.pt")
                    torch.save(checkpoint, best_model_name)
                    logger.info(f"New best model saved: {best_model_name}")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement for {self.patience_counter} epochs")
                    
                    if self.patience_counter >= self.opt.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                        break
            
    def train_epoch(self, epoch: int) -> float:
        """
        Train for a single epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        self.train_data.shuffle()

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()
        
        for i in range(len(self.train_data)):
            batch = self.train_data[i]
            targets = batch[1]

            # Zero gradients
            self.model.zero_grad()
            
            # Apply attention mask for padding
            attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
            self.model.decoder.attn.applyMask(attention_mask)
            
            # Forward pass
            outputs = self.model(batch, eval=False)

            # Calculate loss
            weights = targets.ne(lib.Constants.PAD).float()
            num_words = weights.data.sum()
            loss = self.model.backward(outputs, targets, weights, num_words, self.loss_func)

            # Optimizer step
            self.optim.step()

            # Update statistics
            report_loss += loss
            total_loss += loss
            total_words += num_words
            report_words += num_words
            
            # Logging
            if i % self.opt.log_interval == 0 and i > 0:
                current_ppl = math.exp(report_loss / report_words)
                tokens_per_sec = report_words / (time.time() - last_time)
                elapsed_time = str(datetime.timedelta(seconds=int(time.time() - self.start_time)))
                
                logger.info(
                    f"Epoch {epoch:3d}, Batch {i:6d}/{len(self.train_data)} | "
                    f"Perplexity: {current_ppl:8.2f} | "
                    f"Speed: {tokens_per_sec:5.0f} tokens/s | "
                    f"Elapsed: {elapsed_time}"
                )

                report_loss = report_words = 0
                last_time = time.time()

        return total_loss / total_words

