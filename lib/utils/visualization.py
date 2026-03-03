"""
Attention visualization utilities for analyzing translation models.

Supports:
- Heatmap visualization of attention weights
- Multi-head attention visualization
- Attention comparison across layers
- Interactive plots for research papers
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib seaborn")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("plotly not installed. Install with: pip install plotly")


class AttentionVisualizer:
    """Visualize attention weights from NMT models."""
    
    def __init__(self, output_dir: str = "./attention_plots"):
        """
        Initialize attention visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_attention_heatmap(
        self,
        attention: Union[torch.Tensor, np.ndarray],
        source_tokens: List[str],
        target_tokens: List[str],
        title: str = "Attention Weights",
        output_file: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis'
    ) -> Optional[str]:
        """
        Create heatmap visualization of attention weights.
        
        Args:
            attention: Attention matrix [target_len, source_len]
            source_tokens: List of source tokens
            target_tokens: List of target tokens
            title: Plot title
            output_file: Output filename (auto-generated if None)
            figsize: Figure size (width, height)
            cmap: Colormap name
            
        Returns:
            Path to saved plot or None if saving failed
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib required for heatmap visualization")
            return None
        
        # Convert to numpy if tensor
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            attention,
            xticklabels=source_tokens,
            yticklabels=target_tokens,
            ax=ax,
            cmap=cmap,
            cbar_kws={'label': 'Attention Weight'},
            square=False,
            linewidths=0.1,
            linecolor='gray'
        )
        
        ax.set_xlabel('Source Tokens', fontsize=12)
        ax.set_ylabel('Target Tokens', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        if output_file is None:
            output_file = f"attention_{len(target_tokens)}x{len(source_tokens)}.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Attention heatmap saved to {output_path}")
        return str(output_path)
    
    def plot_multihead_attention(
        self,
        attention_heads: Union[torch.Tensor, np.ndarray],
        source_tokens: List[str],
        target_tokens: List[str],
        output_file: Optional[str] = None,
        max_heads: int = 8
    ) -> Optional[str]:
        """
        Visualize multiple attention heads in a grid.
        
        Args:
            attention_heads: Attention tensor [num_heads, target_len, source_len]
            source_tokens: List of source tokens
            target_tokens: List of target tokens
            output_file: Output filename
            max_heads: Maximum number of heads to visualize
            
        Returns:
            Path to saved plot
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib required for multi-head visualization")
            return None
        
        # Convert to numpy if tensor
        if isinstance(attention_heads, torch.Tensor):
            attention_heads = attention_heads.detach().cpu().numpy()
        
        num_heads = min(attention_heads.shape[0], max_heads)
        
        # Create grid layout
        ncols = min(4, num_heads)
        nrows = (num_heads + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each head
        for i in range(num_heads):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]
            
            sns.heatmap(
                attention_heads[i],
                xticklabels=source_tokens if row == nrows - 1 else [],
                yticklabels=target_tokens if col == 0 else [],
                ax=ax,
                cmap='viridis',
                cbar=True,
                square=False
            )
            
            ax.set_title(f'Head {i+1}', fontsize=10, fontweight='bold')
            
            if row == nrows - 1:
                ax.set_xlabel('Source', fontsize=9)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            
            if col == 0:
                ax.set_ylabel('Target', fontsize=9)
                plt.setp(ax.get_yticklabels(), fontsize=7)
        
        # Remove empty subplots
        for i in range(num_heads, nrows * ncols):
            row = i // ncols
            col = i % ncols
            fig.delaxes(axes[row, col])
        
        plt.suptitle('Multi-Head Attention Visualization', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = f"multihead_attention_{num_heads}heads.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Multi-head attention plot saved to {output_path}")
        return str(output_path)
    
    def create_interactive_attention(
        self,
        attention: Union[torch.Tensor, np.ndarray],
        source_tokens: List[str],
        target_tokens: List[str],
        title: str = "Interactive Attention Visualization",
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Create interactive attention visualization using Plotly.
        
        Args:
            attention: Attention matrix [target_len, source_len]
            source_tokens: List of source tokens
            target_tokens: List of target tokens
            title: Plot title
            output_file: Output HTML filename
            
        Returns:
            Path to saved HTML file
        """
        if not HAS_PLOTLY:
            logger.error("plotly required for interactive visualization")
            return None
        
        # Convert to numpy if tensor
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=source_tokens,
            y=target_tokens,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Source: %{x}<br>Target: %{y}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Source Tokens',
            yaxis_title='Target Tokens',
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        # Save as HTML
        if output_file is None:
            output_file = "interactive_attention.html"
        
        output_path = self.output_dir / output_file
        fig.write_html(str(output_path))
        
        logger.info(f"Interactive attention plot saved to {output_path}")
        return str(output_path)
    
    def plot_attention_distribution(
        self,
        attention: Union[torch.Tensor, np.ndarray],
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot distribution of attention weights.
        
        Args:
            attention: Attention matrix [target_len, source_len]
            output_file: Output filename
            
        Returns:
            Path to saved plot
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib required for distribution plot")
            return None
        
        # Convert to numpy if tensor
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        # Flatten attention weights
        weights = attention.flatten()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(weights, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Attention Weight', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Attention Weights', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(weights, vert=True)
        axes[1].set_ylabel('Attention Weight', fontsize=11)
        axes[1].set_title('Box Plot', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {weights.mean():.3f}\nStd: {weights.std():.3f}\nMax: {weights.max():.3f}'
        axes[1].text(1.15, weights.mean(), stats_text, fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = "attention_distribution.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Attention distribution plot saved to {output_path}")
        return str(output_path)
    
    def compare_attention_layers(
        self,
        attention_layers: List[np.ndarray],
        source_tokens: List[str],
        target_tokens: List[str],
        layer_names: Optional[List[str]] = None,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Compare attention patterns across different layers.
        
        Args:
            attention_layers: List of attention matrices
            source_tokens: List of source tokens
            target_tokens: List of target tokens
            layer_names: Names for each layer
            output_file: Output filename
            
        Returns:
            Path to saved plot
        """
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib required for layer comparison")
            return None
        
        num_layers = len(attention_layers)
        
        if layer_names is None:
            layer_names = [f'Layer {i+1}' for i in range(num_layers)]
        
        # Create grid
        ncols = min(3, num_layers)
        nrows = (num_layers + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        
        if num_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        # Plot each layer
        for i, (attn, name) in enumerate(zip(attention_layers, layer_names)):
            ax = axes[i]
            
            sns.heatmap(
                attn,
                xticklabels=source_tokens,
                yticklabels=target_tokens,
                ax=ax,
                cmap='viridis',
                cbar=True
            )
            
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Source', fontsize=9)
            ax.set_ylabel('Target', fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            plt.setp(ax.get_yticklabels(), fontsize=7)
        
        # Remove empty subplots
        for i in range(num_layers, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Attention Comparison Across Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        if output_file is None:
            output_file = f"attention_layers_comparison.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Layer comparison plot saved to {output_path}")
        return str(output_path)


def extract_attention_from_model(
    model: torch.nn.Module,
    src_tokens: torch.Tensor,
    tgt_tokens: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Extract attention weights from a translation model.
    
    Args:
        model: Translation model
        src_tokens: Source token tensor
        tgt_tokens: Target token tensor
        
    Returns:
        Attention weights tensor or None if extraction fails
    """
    model.eval()
    
    with torch.no_grad():
        try:
            # Forward pass
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'attn'):
                # LSTM-based model
                _ = model((src_tokens, tgt_tokens))
                if hasattr(model.decoder.attn, 'attention'):
                    return model.decoder.attn.attention
            
            # For Transformer models, attention is returned during forward
            # This would need model-specific implementation
            
        except Exception as e:
            logger.warning(f"Failed to extract attention: {e}")
    
    return None


# Convenience function
def visualize_translation_attention(
    model: torch.nn.Module,
    source_sentence: List[str],
    target_sentence: List[str],
    output_dir: str = "./attention_plots",
    plot_type: str = "heatmap"
) -> Optional[str]:
    """
    End-to-end attention visualization for a translation pair.
    
    Args:
        model: Translation model
        source_sentence: List of source tokens
        target_sentence: List of target tokens
        output_dir: Output directory for plots
        plot_type: Type of plot ('heatmap', 'interactive', 'distribution')
        
    Returns:
        Path to saved visualization
    """
    visualizer = AttentionVisualizer(output_dir)
    
    # This is a simplified example - actual implementation would need
    # to get attention weights from the model
    logger.info("Visualizing attention for translation pair")
    logger.info(f"Source: {' '.join(source_sentence)}")
    logger.info(f"Target: {' '.join(target_sentence)}")
    
    # Placeholder: Would extract actual attention from model
    attention = torch.softmax(torch.randn(len(target_sentence), len(source_sentence)), dim=1)
    
    if plot_type == "heatmap":
        return visualizer.plot_attention_heatmap(attention, source_sentence, target_sentence)
    elif plot_type == "interactive":
        return visualizer.create_interactive_attention(attention, source_sentence, target_sentence)
    elif plot_type == "distribution":
        return visualizer.plot_attention_distribution(attention)
    else:
        logger.error(f"Unknown plot type: {plot_type}")
        return None
