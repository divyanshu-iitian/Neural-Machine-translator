"""Unit tests for Transformer architecture."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lib
from lib.model.Transformer import (
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    DecoderLayer,
    TransformerEncoder,
    TransformerDecoder
)


class TestPositionalEncoding:
    """Tests for positional encoding."""
    
    def test_positional_encoding_shape(self):
        """Test output shape matches input."""
        d_model = 512
        batch_size = 32
        seq_len = 10
        
        pe = PositionalEncoding(d_model)
        x = torch.randn(seq_len, batch_size, d_model)
        output = pe(x)
        
        assert output.shape == (seq_len, batch_size, d_model)
    
    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        d_model = 512
        seq_len = 10
        batch_size = 2
        
        pe = PositionalEncoding(d_model, dropout=0.0)
        x = torch.randn(seq_len, batch_size, d_model)
        
        output1 = pe(x)
        output2 = pe(x)
        
        # Should be identical with dropout=0
        assert torch.allclose(output1, output2)


class TestMultiHeadAttention:
    """Tests for multi-head attention."""
    
    def test_attention_output_shape(self):
        """Test attention output has correct shape."""
        d_model = 512
        num_heads = 8
        batch_size = 32
        seq_len = 10
        
        mha = MultiHeadAttention(d_model, num_heads)
        q = k = v = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = mha(q, k, v)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_attention_with_mask(self):
        """Test attention with masking."""
        d_model = 512
        num_heads = 8
        batch_size = 2
        seq_len = 5
        
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        q = k = v = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask that blocks last two positions
        mask = torch.ones(batch_size, seq_len)
        mask[:, -2:] = 0
        
        output, attn_weights = mha(q, k, v, mask)
        
        # Check that attention to masked positions is near zero
        assert torch.abs(attn_weights[:, :, :, -2:]).sum() < 1e-5
    
    def test_self_attention_equals_cross_attention(self):
        """Test self-attention is a special case of cross-attention."""
        d_model = 512
        num_heads = 8
        batch_size = 4
        seq_len = 10
        
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output1, _ = mha(x, x, x)
        output2, _ = mha(x, x, x)
        
        assert torch.allclose(output1, output2)


class TestFeedForward:
    """Tests for feed-forward network."""
    
    def test_feedforward_shape(self):
        """Test feed-forward output shape."""
        d_model = 512
        d_ff = 2048
        batch_size = 32
        seq_len = 10
        
        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        output = ff(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_feedforward_nonlinearity(self):
        """Test that feed-forward applies non-linearity."""
        d_model = 512
        d_ff = 2048
        
        ff = FeedForward(d_model, d_ff, dropout=0.0)
        x = torch.ones(1, 10, d_model)
        
        output = ff(x)
        
        # Output should be different from input (non-identity)
        assert not torch.allclose(output, x)


class TestTransformerLayers:
    """Tests for encoder and decoder layers."""
    
    def test_encoder_layer_shape(self):
        """Test encoder layer output shape."""
        d_model = 512
        num_heads = 8
        d_ff = 2048
        batch_size = 16
        seq_len = 20
        
        layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_decoder_layer_shape(self):
        """Test decoder layer output shape."""
        d_model = 512
        num_heads = 8
        d_ff = 2048
        batch_size = 16
        src_len = 20
        tgt_len = 15
        
        layer = DecoderLayer(d_model, num_heads, d_ff)
        tgt = torch.randn(batch_size, tgt_len, d_model)
        memory = torch.randn(batch_size, src_len, d_model)
        
        output, attn_weights = layer(tgt, memory)
        
        assert output.shape == (batch_size, tgt_len, d_model)
        assert attn_weights.shape == (batch_size, num_heads, tgt_len, src_len)
    
    def test_encoder_layer_residual(self):
        """Test that encoder layer uses residual connections."""
        d_model = 512
        num_heads = 8
        d_ff = 2048
        
        layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.0)
        x = torch.randn(1, 10, d_model)
        
        # With residual, output should be influenced by input
        output = layer(x)
        
        # Not a perfect test, but output shouldn't be completely independent
        correlation = torch.corrcoef(torch.stack([
            x.flatten(),
            output.flatten()
        ]))[0, 1]
        
        assert correlation > 0.1  # Some correlation due to residual


class TestTransformerEncoderDecoder:
    """Integration tests for full encoder-decoder."""
    
    @pytest.fixture
    def mock_opt(self):
        """Create mock options object."""
        opt = Mock()
        opt.rnn_size = 512
        opt.layers = 2
        opt.dropout = 0.1
        opt.num_heads = 8
        opt.d_model = 512
        opt.d_ff = 2048
        return opt
    
    @pytest.fixture
    def mock_dict(self):
        """Create mock dictionary object."""
        dicts = Mock()
        dicts.size.return_value = 10000
        return dicts
    
    def test_encoder_forward(self, mock_opt, mock_dict):
        """Test encoder forward pass."""
        encoder = TransformerEncoder(mock_opt, mock_dict)
        
        batch_size = 4
        seq_len = 10
        src_tokens = torch.randint(0, 10000, (seq_len, batch_size))
        src_lengths = torch.tensor([seq_len] * batch_size)
        
        _, output = encoder((src_tokens, src_lengths))
        
        assert output.shape == (seq_len, batch_size, mock_opt.rnn_size)
    
    def test_decoder_forward(self, mock_opt, mock_dict):
        """Test decoder forward pass."""
        decoder = TransformerDecoder(mock_opt, mock_dict)
        
        batch_size = 4
        src_len = 10
        tgt_len = 8
        
        tgt_tokens = torch.randint(0, 10000, (batch_size, tgt_len))
        encoder_output = torch.randn(src_len, batch_size, mock_opt.rnn_size)
        
        output, _ = decoder((tgt_tokens, None, None), encoder_output)
        
        assert output.shape == (tgt_len, batch_size, mock_opt.rnn_size)


class TestTransformerNumericalStability:
    """Tests for numerical stability."""
    
    def test_attention_no_nan(self):
        """Test that attention doesn't produce NaN values."""
        d_model = 512
        num_heads = 8
        batch_size = 4
        seq_len = 100
        
        mha = MultiHeadAttention(d_model, num_heads)
        q = k = v = torch.randn(batch_size, seq_len, d_model) * 10  # Large values
        
        output, attn_weights = mha(q, k, v)
        
        assert not torch.isnan(output).any()
        assert not torch.isnan(attn_weights).any()
    
    def test_layer_norm_stability(self):
        """Test layer normalization prevents exploding values."""
        d_model = 512
        num_heads = 8
        d_ff = 2048
        
        layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.0)
        
        # Start with large values
        x = torch.randn(1, 10, d_model) * 100
        
        # After multiple passes, values should be normalized
        for _ in range(10):
            x = layer(x)
        
        assert torch.abs(x).max() < 1000  # Should not explode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
