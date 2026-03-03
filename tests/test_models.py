"""Unit tests for model components."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lib
from lib.model.EncoderDecoder import Encoder, StackedLSTMCell


class TestEncoder:
    """Tests for LSTM Encoder."""
    
    @pytest.fixture
    def mock_opt(self):
        """Create mock options."""
        opt = Mock()
        opt.layers = 2
        opt.rnn_size = 512
        opt.word_vec_size = 300
        opt.brnn = True
        opt.dropout = 0.3
        return opt
    
    @pytest.fixture
    def mock_dict(self):
        """Create mock dictionary."""
        dicts = Mock()
        dicts.size.return_value = 10000
        return dicts
    
    def test_encoder_initialization(self, mock_opt, mock_dict):
        """Test encoder initializes correctly."""
        encoder = Encoder(mock_opt, mock_dict)
        
        assert encoder.layers == 2
        assert encoder.hidden_size == 256  # 512 // 2 for bidirectional
        assert encoder.num_directions == 2
    
    def test_encoder_forward_shape(self, mock_opt, mock_dict):
        """Test encoder output shape."""
        encoder = Encoder(mock_opt, mock_dict)
        
        batch_size = 4
        seq_len = 10
        src_tokens = torch.randint(0, 10000, (seq_len, batch_size))
        src_lengths = torch.tensor([seq_len] * batch_size)
        
        hidden, output = encoder((src_tokens, src_lengths))
        
        # Check output shape: [seq_len, batch, hidden_size * num_directions]
        assert output.shape == (seq_len, batch_size, mock_opt.rnn_size)
    
    def test_encoder_different_lengths(self, mock_opt, mock_dict):
        """Test encoder with different sequence lengths."""
        encoder = Encoder(mock_opt, mock_dict)
        
        batch_size = 4
        max_len = 15
        src_tokens = torch.randint(0, 10000, (max_len, batch_size))
        src_lengths = torch.tensor([15, 12, 8, 5])
        
        hidden, output = encoder((src_tokens, src_lengths))
        
        # Should handle variable lengths
        assert output.shape[0] == max_len
        assert output.shape[1] == batch_size


class TestStackedLSTMCell:
    """Tests for StackedLSTMCell."""
    
    def test_stacked_lstm_initialization(self):
        """Test StackedLSTMCell initializes correctly."""
        num_layers = 3
        input_size = 300
        rnn_size = 512
        dropout = 0.3
        
        cell = StackedLSTMCell(num_layers, input_size, rnn_size, dropout)
        
        assert cell.num_layers == 3
        assert len(cell.layers) == 3
        assert len(cell.layer_norms) == 3
    
    def test_stacked_lstm_forward_shape(self):
        """Test StackedLSTMCell output shape."""
        num_layers = 2
        input_size = 300
        rnn_size = 512
        batch_size = 4
        dropout = 0.0
        
        cell = StackedLSTMCell(num_layers, input_size, rnn_size, dropout)
        
        inputs = torch.randn(batch_size, input_size)
        h_0 = torch.randn(num_layers, batch_size, rnn_size)
        c_0 = torch.randn(num_layers, batch_size, rnn_size)
        
        output, (h_1, c_1) = cell(inputs, (h_0, c_0))
        
        assert output.shape == (batch_size, rnn_size)
        assert h_1.shape == (num_layers, batch_size, rnn_size)
        assert c_1.shape == (num_layers, batch_size, rnn_size)
    
    def test_residual_connections(self):
        """Test that residual connections work."""
        num_layers = 3
        input_size = 512  # Same as rnn_size for residual
        rnn_size = 512
        batch_size = 2
        dropout = 0.0
        
        cell = StackedLSTMCell(num_layers, input_size, rnn_size, dropout)
        
        inputs = torch.randn(batch_size, input_size)
        h_0 = torch.randn(num_layers, batch_size, rnn_size)
        c_0 = torch.randn(num_layers, batch_size, rnn_size)
        
        output, _ = cell(inputs, (h_0, c_0))
        
        # Output should not be identical to input (transformation applied)
        assert not torch.allclose(output, inputs)


class TestModelIntegration:
    """Integration tests for complete models."""
    
    @pytest.fixture
    def create_simple_model(self):
        """Create a simple model for testing."""
        opt = Mock()
        opt.layers = 1
        opt.rnn_size = 256
        opt.word_vec_size = 128
        opt.brnn = True
        opt.dropout = 0.1
        opt.input_feed = 1
        opt.brnn_merge = 'concat'
        
        src_dict = Mock()
        src_dict.size.return_value = 5000
        
        tgt_dict = Mock()
        tgt_dict.size.return_value = 5000
        
        encoder = Encoder(opt, src_dict)
        
        return encoder, opt, src_dict, tgt_dict
    
    def test_encoder_eval_mode(self, create_simple_model):
        """Test encoder in eval mode."""
        encoder, opt, _, _ = create_simple_model
        
        encoder.eval()
        
        batch_size = 2
        seq_len = 5
        src_tokens = torch.randint(0, 5000, (seq_len, batch_size))
        src_lengths = torch.tensor([seq_len] * batch_size)
        
        with torch.no_grad():
            hidden, output = encoder((src_tokens, src_lengths))
        
        assert output.requires_grad == False
    
    def test_encoder_gradients(self, create_simple_model):
        """Test that encoder produces gradients."""
        encoder, opt, _, _ = create_simple_model
        
        encoder.train()
        
        batch_size = 2
        seq_len = 5
        src_tokens = torch.randint(0, 5000, (seq_len, batch_size))
        src_lengths = torch.tensor([seq_len] * batch_size)
        
        hidden, output = encoder((src_tokens, src_lengths))
        
        # Create dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that embeddings have gradients
        assert encoder.word_lut.weight.grad is not None


class TestLayerNormalization:
    """Tests for layer normalization effects."""
    
    def test_layer_norm_reduces_variance(self):
        """Test that layer norm stabilizes activations."""
        batch_size = 8
        d_model = 512
        
        # Create layer norm
        ln = nn.LayerNorm(d_model)
        
        # Input with high variance
        x = torch.randn(batch_size, d_model) * 10
        
        # Apply layer norm
        x_norm = ln(x)
        
        # Normalized output should have lower variance
        assert x_norm.var(dim=1).mean() < x.var(dim=1).mean()
    
    def test_layer_norm_preserves_shape(self):
        """Test that layer norm preserves tensor shape."""
        shapes = [(8, 512), (4, 10, 256), (2, 5, 3, 128)]
        
        for shape in shapes:
            d_model = shape[-1]
            ln = nn.LayerNorm(d_model)
            x = torch.randn(*shape)
            x_norm = ln(x)
            
            assert x_norm.shape == x.shape


class TestModelSaveLoad:
    """Tests for model saving and loading."""
    
    @pytest.fixture
    def temp_model_path(self, tmp_path):
        """Create temporary path for model."""
        return tmp_path / "model.pt"
    
    def test_encoder_save_load(self, temp_model_path):
        """Test saving and loading encoder."""
        opt = Mock()
        opt.layers = 1
        opt.rnn_size = 256
        opt.word_vec_size = 128
        opt.brnn = True
        opt.dropout = 0.1
        
        src_dict = Mock()
        src_dict.size.return_value = 1000
        
        # Create and save encoder
        encoder1 = Encoder(opt, src_dict)
        torch.save(encoder1.state_dict(), temp_model_path)
        
        # Load into new encoder
        encoder2 = Encoder(opt, src_dict)
        encoder2.load_state_dict(torch.load(temp_model_path))
        
        # Test with same input
        batch_size = 2
        seq_len = 5
        src_tokens = torch.randint(0, 1000, (seq_len, batch_size))
        src_lengths = torch.tensor([seq_len] * batch_size)
        
        encoder1.eval()
        encoder2.eval()
        
        with torch.no_grad():
            _, out1 = encoder1((src_tokens, src_lengths))
            _, out2 = encoder2((src_tokens, src_lengths))
        
        # Outputs should be identical
        assert torch.allclose(out1, out2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
