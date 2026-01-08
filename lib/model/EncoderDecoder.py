"""
Advanced Neural Machine Translation Model Architecture

This module implements a modern encoder-decoder architecture with:
- Layer normalization for training stability
- Residual connections for better gradient flow  
- Improved initialization strategies
- Flexible attention mechanisms
- Enhanced dropout strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import lib


class Encoder(nn.Module):
    """
    Enhanced bidirectional LSTM encoder with layer normalization and residual connections.
    
    Args:
        opt: Configuration options containing model hyperparameters
        dicts: Dictionary containing vocabulary mapping
    """
    
    def __init__(self, opt, dicts):
        super(Encoder, self).__init__()
        
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        
        # Embedding layer with improved initialization
        self.word_lut = nn.Embedding(
            dicts.size(), 
            opt.word_vec_size, 
            padding_idx=lib.Constants.PAD
        )
        nn.init.xavier_uniform_(self.word_lut.weight)
        
        # Layer normalization for embeddings
        self.emb_layer_norm = nn.LayerNorm(opt.word_vec_size)
        
        # Recurrent encoder with configurable dropout
        self.rnn = nn.LSTM(
            opt.word_vec_size, 
            self.hidden_size, 
            num_layers=opt.layers, 
            dropout=opt.dropout if opt.layers > 1 else 0,
            bidirectional=opt.brnn
        )
        
        # Initialize LSTM weights properly
        self._init_rnn_weights()
        
    def _init_rnn_weights(self):
        """Initialize RNN weights using orthogonal initialization for better training."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(self, inputs: Tuple, hidden: Optional[Tuple] = None) -> Tuple:
        """
        Forward pass through encoder.
        
        Args:
            inputs: Tuple of (input_sequences, sequence_lengths)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (final_hidden_state, encoder_outputs)
        """
        # Apply embeddings with layer normalization
        emb = self.word_lut(inputs[0])
        emb = self.emb_layer_norm(emb)
        
        # Pack padded sequences for efficiency
        emb = pack(emb, inputs[1])
        outputs, hidden_t = self.rnn(emb, hidden)
        outputs = unpack(outputs)[0]
        
        return hidden_t, outputs


class StackedLSTMCell(nn.Module):
    """
    Stacked LSTM with residual connections and layer normalization.
    
    Implements multiple LSTM layers with:
    - Residual connections (for layers after the first)
    - Dropout between layers
    - Layer normalization
    """
    
    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super(StackedLSTMCell, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else rnn_size
            self.layers.append(nn.LSTMCell(layer_input_size, rnn_size))
            self.layer_norms.append(nn.LayerNorm(rnn_size))
            
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Forget gate bias initialization
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

    def forward(self, inputs: torch.Tensor, hidden: Tuple) -> Tuple:
        """
        Forward pass with residual connections.
        
        Args:
            inputs: Input tensor
            hidden: Tuple of (hidden_states, cell_states)
            
        Returns:
            Tuple of (output, (new_hidden_states, new_cell_states))
        """
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(inputs, (h_0[i], c_0[i]))
            
            # Apply layer normalization
            h_1_i = self.layer_norms[i](h_1_i)
            
            # Residual connection (skip first layer)
            if i > 0 and inputs.size(-1) == h_1_i.size(-1):
                h_1_i = h_1_i + inputs
            
            inputs = h_1_i
            
            # Apply dropout between layers (not after last layer)
            if i < self.num_layers - 1:
                inputs = self.dropout(inputs)
            
            h_1.append(h_1_i)
            c_1.append(c_1_i)

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return inputs, (h_1, c_1)


class Decoder(nn.Module):
    """
    Enhanced decoder with attention mechanism and input feeding.
    
    Features:
    - Layer normalization for stability
    - Configurable input feeding
    - Residual connections in stacked LSTM
    - Improved weight initialization
    """
    
    def __init__(self, opt, dicts):
        super(Decoder, self).__init__()
        
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        # Embedding layer with proper initialization
        self.word_lut = nn.Embedding(
            dicts.size(), 
            opt.word_vec_size, 
            padding_idx=lib.Constants.PAD
        )
        nn.init.xavier_uniform_(self.word_lut.weight)
        
        # Layer normalization for embeddings
        self.emb_layer_norm = nn.LayerNorm(opt.word_vec_size)
        
        # Stacked LSTM with residual connections
        self.rnn = StackedLSTMCell(opt.layers, input_size, opt.rnn_size, opt.dropout)
        
        # Attention mechanism
        self.attn = lib.GlobalAttention(opt.rnn_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(opt.dropout)
        
        self.hidden_size = opt.rnn_size

    def step(self, emb: torch.Tensor, output: torch.Tensor, 
             hidden: Tuple, context: torch.Tensor) -> Tuple:
        """
        Perform single decoding step.
        
        Args:
            emb: Embedding of current token
            output: Previous output (for input feeding)
            hidden: Previous hidden state
            context: Encoder context vectors
            
        Returns:
            Tuple of (output, hidden_state)
        """
        # Input feeding: concatenate previous output with embedding
        if self.input_feed:
            emb = torch.cat([emb, output], 1)
        
        # Apply LSTM
        output, hidden = self.rnn(emb, hidden)
        
        # Apply attention mechanism
        output, attn = self.attn(output, context)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, hidden

    def forward(self, inputs: torch.Tensor, init_states: Tuple) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            inputs: Input token indices
            init_states: Initial states (embedding, output, hidden, context)
            
        Returns:
            Decoder outputs for all timesteps
        """
        emb, output, hidden, context = init_states
        
        # Apply embeddings with layer normalization
        embs = self.emb_layer_norm(self.word_lut(inputs))

        outputs = []
        for i in range(inputs.size(0)):
            output, hidden = self.step(emb, output, hidden, context)
            outputs.append(output)
            emb = embs[i]

        outputs = torch.stack(outputs)
        return outputs


class NMTModel(nn.Module):
    """
    Complete Neural Machine Translation model combining encoder, decoder, and generator.
    
    This model implements a seq2seq architecture with attention for translation tasks.
    Supports both training and inference modes with greedy decoding and sampling.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, generator, opt):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.opt = opt

    def make_init_decoder_output(self, context: torch.Tensor) -> torch.Tensor:
        """
        Create initial decoder output (zeros).
        
        Args:
            context: Encoder context vectors
            
        Returns:
            Zero-initialized tensor for initial decoder output
        """
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return context.data.new(*h_size).zero_()

    def _fix_enc_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """
        Transform encoder hidden state for decoder initialization.
        
        For bidirectional encoders, concatenates forward and backward states.
        
        Args:
            h: Encoder hidden states shaped as (layers*directions) x batch x dim
            
        Returns:
            Transformed hidden states shaped as layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def initialize(self, inputs: Tuple, eval: bool) -> Tuple:
        """
        Initialize decoder state from encoder outputs.
        
        Args:
            inputs: Tuple of (source_sequences, target_sequences)
            eval: Whether in evaluation mode
            
        Returns:
            Tuple of (targets, initial_decoder_states)
        """
        src = inputs[0]
        tgt = inputs[1]
        
        # Encode source sequence
        enc_hidden, context = self.encoder(src)
        
        # Initialize decoder output
        init_output = self.make_init_decoder_output(context)
        
        # Fix encoder hidden for decoder
        enc_hidden = (
            self._fix_enc_hidden(enc_hidden[0]),
            self._fix_enc_hidden(enc_hidden[1])
        )
        
        # Create initial token (BOS)
        init_token = torch.LongTensor([lib.Constants.BOS] * init_output.size(0))
        
        if self.opt.cuda:
            init_token = init_token.cuda()
        
        emb = self.decoder.word_lut(init_token)
        
        return tgt, (emb, init_output, enc_hidden, context.transpose(0, 1))

    def forward(self, inputs: Tuple, eval: bool, regression: bool = False) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            inputs: Tuple of input sequences
            eval: Evaluation mode flag
            regression: Whether performing regression task
            
        Returns:
            Model outputs or logits
        """
        targets, init_states = self.initialize(inputs, eval)
        outputs = self.decoder(targets, init_states)

        if regression:
            logits = self.generator(outputs)
            return logits.view_as(targets)
        return outputs

    def backward(self, outputs: torch.Tensor, targets: torch.Tensor, 
                 weights: torch.Tensor, normalizer: float, 
                 criterion, regression: bool = False) -> float:
        """
        Backward pass computing loss and gradients.
        
        Args:
            outputs: Model outputs
            targets: Target sequences
            weights: Loss weights
            normalizer: Normalization factor
            criterion: Loss criterion
            regression: Whether performing regression
            
        Returns:
            Computed loss value
        """
        grad_output, loss = self.generator.backward(
            outputs, targets, weights, normalizer, criterion, regression
        )
        outputs.backward(grad_output)
        return loss

    def predict(self, outputs: torch.Tensor, targets: torch.Tensor, 
                weights: torch.Tensor, criterion):
        """
        Make predictions from outputs.
        
        Args:
            outputs: Model outputs
            targets: Target sequences
            weights: Loss weights
            criterion: Loss criterion
            
        Returns:
            Prediction results
        """
        return self.generator.predict(outputs, targets, weights, criterion)

    def translate(self, inputs: Tuple, max_length: int) -> torch.Tensor:
        """
        Translate input sequence using greedy decoding.
        
        Args:
            inputs: Input sequences
            max_length: Maximum output length
            
        Returns:
            Predicted token indices
        """
        """
        targets, init_states = self.initialize(inputs, eval=True)
        emb, output, hidden, context = init_states
        
        preds = [] 
        batch_size = targets.size(1)
        num_eos = targets[0].data.new(batch_size).byte().zero_()

        # Greedy decoding loop
        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            logit = self.generator(output)
            pred = logit.max(1)[1].view(-1).data
            preds.append(pred)

            # Early stopping: check if all sentences reached EOS
            num_eos |= (pred == lib.Constants.EOS)
            if num_eos.sum() == batch_size: 
                break

            emb = self.decoder.word_lut(pred)

        preds = torch.stack(preds)
        return preds

    def sample(self, inputs: Tuple, max_length: int) -> Tuple:
        """
        Generate translations using sampling from probability distribution.
        
        Used for reinforcement learning training.
        
        Args:
            inputs: Input sequences
            max_length: Maximum output length
            
        Returns:
            Tuple of (sampled_tokens, outputs)
        """
        targets, init_states = self.initialize(inputs, eval=False)
        emb, output, hidden, context = init_states

        outputs = []
        samples = []
        batch_size = targets.size(1)
        num_eos = targets[0].data.new(batch_size).byte().zero_()

        # Sampling loop
        for i in range(max_length):
            output, hidden = self.decoder.step(emb, output, hidden, context)
            outputs.append(output)
            
            # Sample from probability distribution
            dist = F.softmax(self.generator(output), dim=-1)
            sample = dist.multinomial(1, replacement=False).view(-1).data
            samples.append(sample)

            # Early stopping: check if all sentences reached EOS
            num_eos |= (sample == lib.Constants.EOS)
            if num_eos.sum() == batch_size: 
                break

            emb = self.decoder.word_lut(sample)

        outputs = torch.stack(outputs)
        samples = torch.stack(samples)
        return samples, outputs


