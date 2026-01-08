"""
Advanced Global Attention Mechanisms

Implements scalable attention mechanisms with:
- Scaled dot-product attention
- Configurable attention types (dot, general, concat)
- Numerical stability improvements
- Efficient masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_INF = float('inf')


class GlobalAttention(nn.Module):
    """
    Global attention mechanism with multiple scoring functions.
    
    Supports different attention variants:
    - 'dot': Simple dot product
    - 'general': Parameterized dot product (default)
    - 'concat': Concatenation-based
    
    Args:
        dim: Hidden dimension size
        attn_type: Type of attention mechanism
    """
    
    def __init__(self, dim: int, attn_type: str = 'general'):
        super(GlobalAttention, self).__init__()
        
        self.dim = dim
        self.attn_type = attn_type
        self.mask = None
        
        # Attention scoring
        if attn_type == 'general':
            self.linear_in = nn.Linear(dim, dim, bias=False)
            nn.init.xavier_uniform_(self.linear_in.weight)
        elif attn_type == 'concat':
            self.linear_query = nn.Linear(dim, dim, bias=False)
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, 1, bias=False)
            nn.init.xavier_uniform_(self.linear_query.weight)
            nn.init.xavier_uniform_(self.linear_context.weight)
            nn.init.xavier_uniform_(self.v.weight)
        
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
        # Output projection
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        nn.init.xavier_uniform_(self.linear_out.weight)
        
        # Activation
        self.tanh = nn.Tanh()
        
        # Scaling factor for numerical stability
        self.scale = 1.0 / math.sqrt(dim)

    def applyMask(self, mask: torch.Tensor):
        """
        Set attention mask to ignore padding tokens.
        
        Args:
            mask: Boolean mask tensor
        """
        self.mask = mask

    def score(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores based on attention type.
        
        Args:
            query: Query vector (batch x dim)
            context: Context vectors (batch x sourceL x dim)
            
        Returns:
            Attention scores (batch x sourceL)
        """
        if self.attn_type == 'dot':
            # Simple scaled dot-product
            targetT = query.unsqueeze(2)  # batch x dim x 1
            scores = torch.bmm(context, targetT).squeeze(2)
            scores = scores * self.scale
            
        elif self.attn_type == 'general':
            # Parameterized dot-product
            targetT = self.linear_in(query).unsqueeze(2)  # batch x dim x 1
            scores = torch.bmm(context, targetT).squeeze(2)
            scores = scores * self.scale
            
        elif self.attn_type == 'concat':
            # Concatenation-based attention
            sourceL = context.size(1)
            query_expanded = query.unsqueeze(1).expand(-1, sourceL, -1)
            
            query_proj = self.linear_query(query_expanded)
            context_proj = self.linear_context(context)
            
            scores = self.v(self.tanh(query_proj + context_proj)).squeeze(2)
        
        return scores

    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> tuple:
        """
        Apply attention mechanism.
        
        Args:
            inputs: Query inputs (batch x dim)
            context: Encoder context (batch x sourceL x dim)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Compute attention scores
        attn_scores = self.score(inputs, context)
        
        # Apply mask if provided (for padding)
        if self.mask is not None:
            attn_scores.data.masked_fill_(self.mask, -_INF)
        
        # Normalize to get attention weights
        attn_weights = self.softmax(attn_scores)
        
        # Compute weighted context
        attn3 = attn_weights.view(attn_weights.size(0), 1, attn_weights.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        
        # Combine with input and apply output transformation
        combined = torch.cat((weighted_context, inputs), 1)
        output = self.tanh(self.linear_out(combined))

        return output, attn_weights
