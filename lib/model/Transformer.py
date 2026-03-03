"""
Transformer Architecture for Neural Machine Translation

Implements the "Attention is All You Need" architecture with modern improvements:
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Residual connections
- Label smoothing support

Reference: Vaswani et al., 2017 (https://arxiv.org/abs/1706.03762)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import lib


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with scaled dot-product attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, 1, seq_len] or [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Padding mask [batch_size, 1, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Decoder input [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source padding mask [batch_size, 1, src_seq_len]
            tgt_mask: Target causal mask [batch_size, tgt_seq_len, tgt_seq_len]
            
        Returns:
            Tuple of (output, cross_attention_weights)
        """
        # Self-attention with residual and layer norm
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross-attention with residual and layer norm
        cross_output, cross_attn = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_output))
        
        # Feed-forward with residual and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x, cross_attn


class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple layers."""
    
    def __init__(self, opt, dicts):
        super(TransformerEncoder, self).__init__()
        
        d_model = opt.d_model if hasattr(opt, 'd_model') else opt.rnn_size
        num_heads = opt.num_heads if hasattr(opt, 'num_heads') else 8
        d_ff = opt.d_ff if hasattr(opt, 'd_ff') else d_model * 4
        num_layers = opt.layers
        dropout = opt.dropout
        
        self.d_model = d_model
        self.embedding = nn.Embedding(dicts.size(), d_model, padding_idx=lib.Constants.PAD)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, inputs: Tuple) -> Tuple:
        """
        Args:
            inputs: Tuple of (src_tokens, src_lengths)
            
        Returns:
            Tuple of (None, encoder_output)
        """
        src_tokens, src_lengths = inputs
        
        # Create padding mask
        src_mask = (src_tokens != lib.Constants.PAD).unsqueeze(1)
        
        # Embedding and positional encoding
        x = self.embedding(src_tokens) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model] -> [seq_len, batch, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to [batch, seq_len, d_model]
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        x = self.norm(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model] -> [seq_len, batch, d_model]
        
        return None, x


class TransformerDecoder(nn.Module):
    """Transformer decoder with multiple layers."""
    
    def __init__(self, opt, dicts):
        super(TransformerDecoder, self).__init__()
        
        d_model = opt.d_model if hasattr(opt, 'd_model') else opt.rnn_size
        num_heads = opt.num_heads if hasattr(opt, 'num_heads') else 8
        d_ff = opt.d_ff if hasattr(opt, 'd_ff') else d_model * 4
        num_layers = opt.layers
        dropout = opt.dropout
        
        self.d_model = d_model
        self.embedding = nn.Embedding(dicts.size(), d_model, padding_idx=lib.Constants.PAD)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self,
        inputs: Tuple,
        encoder_output: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Tuple of (tgt_tokens, None, init_state)
            encoder_output: Encoder outputs [src_len, batch, d_model]
            context: Previous context (not used in Transformer)
            
        Returns:
            Tuple of (decoder_output, attention_weights)
        """
        tgt_tokens = inputs[0]
        batch_size, tgt_len = tgt_tokens.size()
        
        # Embedding and positional encoding
        x = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1))
        x = x.transpose(0, 1)  # [batch, tgt_len, d_model]
        
        # Create masks
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(tgt_tokens.device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        src_mask = None  # Can add source padding mask if needed
        
        # Transpose encoder output for decoder
        encoder_output = encoder_output.transpose(0, 1)  # [batch, src_len, d_model]
        
        # Pass through decoder layers
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, encoder_output, src_mask, tgt_mask)
        
        x = self.norm(x)
        x = x.transpose(0, 1)  # [batch, tgt_len, d_model] -> [tgt_len, batch, d_model]
        
        return x, attn_weights


class TransformerModel(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks."""
    
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, generator):
        super(TransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
    
    def forward(self, inputs: Tuple) -> torch.Tensor:
        """
        Args:
            inputs: Tuple of (src, tgt)
            
        Returns:
            Model output logits
        """
        src, tgt = inputs
        
        # Encode
        _, encoder_output = self.encoder(src)
        
        # Decode
        tgt_input = (tgt[0][:-1], None, None)  # Remove last token for teacher forcing
        decoder_output, _ = self.decoder(tgt_input, encoder_output)
        
        # Generate
        output = self.generator(decoder_output)
        
        return output
    
    def translate(self, src: Tuple, max_length: int, beam_size: int = 1) -> torch.Tensor:
        """
        Translate source sequence using greedy decoding or beam search.
        
        Args:
            src: Source sequence tuple (tokens, lengths)
            max_length: Maximum generation length
            beam_size: Beam size (1 for greedy)
            
        Returns:
            Predicted token indices
        """
        self.eval()
        with torch.no_grad():
            # Encode
            _, encoder_output = self.encoder(src)
            batch_size = src[0].size(1)
            
            # Initialize decoder input with <s> token
            tgt_tokens = torch.full((batch_size, 1), lib.Constants.BOS, 
                                   dtype=torch.long, device=encoder_output.device)
            
            # Greedy decoding
            for _ in range(max_length):
                tgt_input = (tgt_tokens, None, None)
                decoder_output, _ = self.decoder(tgt_input, encoder_output)
                
                # Get last token predictions
                logits = self.generator(decoder_output[-1:, :, :])
                next_token = logits.argmax(dim=-1).squeeze(0)
                
                # Append to sequence
                tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(1)], dim=1)
                
                # Check if all sequences have generated EOS
                if (next_token == lib.Constants.EOS).all():
                    break
            
            return tgt_tokens.transpose(0, 1)
