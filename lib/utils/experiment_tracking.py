"""
Experiment tracking and comparison utilities for NMT research.

Supports:
- Weights & Biases (wandb) integration
- TensorBoard logging
- Experiment comparison and visualization
- Hyperparameter tracking
- Model checkpointing with metadata
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface."""
    
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        log_dir: str = "./experiments",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: str = "nmt-research",
        wandb_entity: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration dictionary
            log_dir: Directory for logs and checkpoints
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
        """
        self.experiment_name = experiment_name
        self.config = config
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Initialize timestamp
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Save configuration
        self._save_config()
        
        # Initialize wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    config=config,
                    dir=str(self.log_dir)
                )
                logger.info(f"Weights & Biases initialized: {wandb.run.url}")
            except ImportError:
                logger.warning("wandb not installed. Install with: pip install wandb")
                self.use_wandb = False
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / self.timestamp
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard initialized: {tb_dir}")
            except ImportError:
                logger.warning("tensorboard not installed. Install with: pip install tensorboard")
                self.use_tensorboard = False
        
        # Metrics storage
        self.metrics_history = []
    
    def _save_config(self):
        """Save experiment configuration to JSON file."""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = ""
    ):
        """
        Log metrics to all enabled trackers.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch number
            prefix: Optional prefix for metric names (e.g., "train/", "val/")
        """
        # Add prefix to metric names
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Store in history
        record = {"step": step, **prefixed_metrics, "timestamp": datetime.now().isoformat()}
        self.metrics_history.append(record)
        
        # Log to wandb
        if self.use_wandb:
            self.wandb.log(prefixed_metrics, step=step)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for name, value in prefixed_metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(name, value, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: Any):
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_sample: Sample input for tracing
        """
        if self.use_tensorboard:
            try:
                self.tb_writer.add_graph(model, input_sample)
                logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
        
        if self.use_wandb:
            try:
                self.wandb.watch(model, log="all", log_freq=100)
            except Exception as e:
                logger.warning(f"Failed to watch model with wandb: {e}")
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        metric_value: float,
        metric_name: str = "loss",
        is_best: bool = False
    ):
        """
        Save model checkpoint with metadata.
        
        Args:
            checkpoint: Checkpoint dictionary
            metric_value: Value of the tracking metric
            metric_name: Name of the tracking metric
            is_best: Whether this is the best checkpoint so far
        """
        # Add metadata
        checkpoint["metadata"] = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "metric_value": metric_value,
        }
        
        # Save regular checkpoint
        ckpt_path = self.log_dir / f"checkpoint_step_{checkpoint.get('step', 0)}.pt"
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.log_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
            
            if self.use_wandb:
                # Save as wandb artifact
                try:
                    artifact = self.wandb.Artifact(
                        name=f"{self.experiment_name}_best",
                        type="model",
                        description=f"Best model with {metric_name}={metric_value:.4f}"
                    )
                    artifact.add_file(str(best_path))
                    self.wandb.log_artifact(artifact)
                except Exception as e:
                    logger.warning(f"Failed to save wandb artifact: {e}")
    
    def log_text(self, name: str, text: str, step: int):
        """
        Log text samples (e.g., translations).
        
        Args:
            name: Name/tag for the text
            text: Text content
            step: Current step
        """
        if self.use_tensorboard:
            self.tb_writer.add_text(name, text, step)
        
        if self.use_wandb:
            self.wandb.log({name: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """
        Log histogram of values (e.g., gradients, weights).
        
        Args:
            name: Name of the histogram
            values: Tensor of values
            step: Current step
        """
        if self.use_tensorboard:
            self.tb_writer.add_histogram(name, values, step)
        
        if self.use_wandb:
            self.wandb.log({name: self.wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def finish(self):
        """Clean up and finish experiment tracking."""
        # Save final metrics history
        history_path = self.log_dir / "metrics_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics history saved to {history_path}")
        
        # Close TensorBoard
        if self.use_tensorboard:
            self.tb_writer.close()
        
        # Finish wandb
        if self.use_wandb:
            self.wandb_run.finish()
        
        # Log duration
        duration = datetime.now() - self.start_time
        logger.info(f"Experiment finished. Duration: {duration}")


class ExperimentComparator:
    """Compare multiple experiments and visualize results."""
    
    def __init__(self, experiments_dir: str = "./experiments"):
        """
        Initialize experiment comparator.
        
        Args:
            experiments_dir: Directory containing experiment logs
        """
        self.experiments_dir = Path(experiments_dir)
    
    def list_experiments(self) -> List[str]:
        """List all available experiments."""
        if not self.experiments_dir.exists():
            return []
        
        experiments = [d.name for d in self.experiments_dir.iterdir() if d.is_dir()]
        return sorted(experiments)
    
    def load_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        Load experiment data.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary containing config and metrics
        """
        exp_dir = self.experiments_dir / experiment_name
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        # Load config
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Load metrics history
        history_path = exp_dir / "metrics_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = []
        
        return {
            "name": experiment_name,
            "config": config,
            "metrics": metrics,
            "path": str(exp_dir)
        }
    
    def compare_experiments(
        self,
        experiment_names: List[str],
        metric_name: str = "val/loss"
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments on a specific metric.
        
        Args:
            experiment_names: List of experiment names to compare
            metric_name: Name of the metric to compare
            
        Returns:
            Comparison results dictionary
        """
        results = {}
        
        for exp_name in experiment_names:
            try:
                exp_data = self.load_experiment(exp_name)
                metrics = exp_data["metrics"]
                
                # Extract the specific metric
                values = [m.get(metric_name) for m in metrics if metric_name in m]
                
                if values:
                    results[exp_name] = {
                        "best": min(values) if "loss" in metric_name else max(values),
                        "final": values[-1],
                        "history": values,
                        "config": exp_data["config"]
                    }
            except Exception as e:
                logger.warning(f"Failed to load experiment {exp_name}: {e}")
        
        return results
    
    def generate_comparison_report(
        self,
        experiment_names: List[str],
        output_file: str = "comparison_report.md"
    ):
        """
        Generate a markdown report comparing experiments.
        
        Args:
            experiment_names: List of experiments to compare
            output_file: Output markdown file path
        """
        report = ["# Experiment Comparison Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Load all experiments
        experiments = []
        for exp_name in experiment_names:
            try:
                experiments.append(self.load_experiment(exp_name))
            except Exception as e:
                logger.warning(f"Skipping {exp_name}: {e}")
        
        # Summary table
        report.append("## Summary\n\n")
        report.append("| Experiment | Architecture | Best Val Loss | Final Val Loss |\n")
        report.append("|------------|--------------|---------------|----------------|\n")
        
        for exp in experiments:
            name = exp["name"]
            config = exp["config"]
            metrics = exp["metrics"]
            
            arch = config.get("model_type", "N/A")
            val_losses = [m.get("val/loss") for m in metrics if "val/loss" in m]
            
            best_loss = f"{min(val_losses):.4f}" if val_losses else "N/A"
            final_loss = f"{val_losses[-1]:.4f}" if val_losses else "N/A"
            
            report.append(f"| {name} | {arch} | {best_loss} | {final_loss} |\n")
        
        # Individual experiment details
        report.append("\n## Experiment Details\n\n")
        
        for exp in experiments:
            report.append(f"### {exp['name']}\n\n")
            report.append("**Configuration:**\n\n")
            for key, value in exp["config"].items():
                report.append(f"- `{key}`: {value}\n")
            report.append("\n")
        
        # Write report
        output_path = self.experiments_dir / output_file
        with open(output_path, 'w') as f:
            f.writelines(report)
        
        logger.info(f"Comparison report saved to {output_path}")
        return str(output_path)


def create_experiment_tracker(opt) -> ExperimentTracker:
    """
    Factory function to create experiment tracker from options.
    
    Args:
        opt: Command-line options object
        
    Returns:
        Initialized ExperimentTracker
    """
    # Build config dictionary from options
    config = vars(opt) if hasattr(opt, '__dict__') else opt
    
    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config.get("model_type", "lstm")
    exp_name = f"{model_type}_{timestamp}"
    
    tracker = ExperimentTracker(
        experiment_name=exp_name,
        config=config,
        log_dir=config.get("log_dir", "./experiments"),
        use_wandb=config.get("use_wandb", False),
        use_tensorboard=config.get("use_tensorboard", True),
        wandb_project=config.get("wandb_project", "nmt-research")
    )
    
    return tracker
