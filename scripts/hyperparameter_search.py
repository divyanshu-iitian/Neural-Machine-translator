"""
Hyperparameter optimization for NMT models using Optuna.

Supports:
- Automated hyperparameter search
- Multiple optimization strategies (Grid, Random, Bayesian)
- Distributed optimization
- Pruning of unpromising trials
- Visualization of optimization results
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

import torch
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NMTHyperparameterSearch:
    """Hyperparameter optimization for NMT models."""
    
    def __init__(
        self,
        study_name: str = "nmt_optimization",
        storage: Optional[str] = None,
        direction: str = "minimize",
        sampler: str = "tpe",
        pruner: str = "median"
    ):
        """
        Initialize hyperparameter search.
        
        Args:
            study_name: Name of the optimization study
            storage: Database URL for distributed optimization (None for in-memory)
            direction: Optimization direction ('minimize' or 'maximize')
            sampler: Sampling strategy ('tpe', 'random', 'grid')
            pruner: Pruning strategy ('median', 'percentile', 'hyperband')
        """
        self.study_name = study_name
        self.direction = direction
        
        # Create sampler
        if sampler == "tpe":
            sampler_obj = optuna.samplers.TPESampler()
        elif sampler == "random":
            sampler_obj = optuna.samplers.RandomSampler()
        elif sampler == "grid":
            # Grid search requires search_space parameter
            sampler_obj = optuna.samplers.GridSampler({})
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # Create pruner
        if pruner == "median":
            pruner_obj = optuna.pruners.MedianPruner()
        elif pruner == "percentile":
            pruner_obj = optuna.pruners.PercentilePruner(percentile=25.0)
        elif pruner == "hyperband":
            pruner_obj = optuna.pruners.HyperbandPruner()
        else:
            pruner_obj = optuna.pruners.MedianPruner()
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            sampler=sampler_obj,
            pruner=pruner_obj,
            load_if_exists=True
        )
        
        logger.info(f"Created study: {study_name}")
        logger.info(f"Direction: {direction}, Sampler: {sampler}, Pruner: {pruner}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {
            # Model architecture
            "layers": trial.suggest_int("layers", 1, 4),
            "rnn_size": trial.suggest_categorical("rnn_size", [256, 512, 1024]),
            "word_vec_size": trial.suggest_categorical("word_vec_size", [256, 512]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            
            # Training
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "adagrad"]),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 1.0, 10.0),
            
            # Regularization
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-4),
            
            # Architecture type
            "model_type": trial.suggest_categorical("model_type", ["lstm", "transformer"]),
        }
        
        # Transformer-specific parameters
        if params["model_type"] == "transformer":
            params["num_heads"] = trial.suggest_categorical("num_heads", [4, 8, 16])
            params["d_ff"] = trial.suggest_categorical("d_ff", [1024, 2048, 4096])
        
        return params
    
    def objective(
        self,
        trial: optuna.Trial,
        train_fn,
        eval_fn,
        data_path: str,
        max_epochs: int = 20
    ) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial
            train_fn: Training function
            eval_fn: Evaluation function
            data_path: Path to training data
            max_epochs: Maximum training epochs
            
        Returns:
            Validation metric value
        """
        # Get hyperparameters
        params = self.suggest_hyperparameters(trial)
        
        logger.info(f"Trial {trial.number}: {params}")
        
        try:
            # Train model with suggested parameters
            model, metrics = train_fn(params, data_path, max_epochs, trial)
            
            # Evaluate
            val_metric = eval_fn(model, data_path)
            
            return val_metric
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
    def optimize(
        self,
        train_fn,
        eval_fn,
        data_path: str,
        n_trials: int = 50,
        max_epochs: int = 20,
        timeout: Optional[int] = None
    ):
        """
        Run hyperparameter optimization.
        
        Args:
            train_fn: Function to train model with given parameters
            eval_fn: Function to evaluate model
            data_path: Path to training data
            n_trials: Number of optimization trials
            max_epochs: Maximum epochs per trial
            timeout: Timeout in seconds (None for no limit)
        """
        logger.info(f"Starting optimization with {n_trials} trials")
        
        # Create objective with fixed parameters
        def objective_wrapper(trial):
            return self.objective(trial, train_fn, eval_fn, data_path, max_epochs)
        
        # Run optimization
        self.study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Log best results
        logger.info("=" * 80)
        logger.info("Optimization Complete!")
        logger.info(f"Best trial: {self.study.best_trial.number}")
        logger.info(f"Best value: {self.study.best_value:.4f}")
        logger.info(f"Best parameters:")
        for key, value in self.study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
    
    def save_results(self, output_dir: str = "./optimization_results"):
        """
        Save optimization results.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        best_params_path = output_path / "best_parameters.json"
        with open(best_params_path, 'w') as f:
            json.dump(self.study.best_params, f, indent=2)
        
        logger.info(f"Best parameters saved to {best_params_path}")
        
        # Save all trials
        trials_path = output_path / "all_trials.json"
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            })
        
        with open(trials_path, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        logger.info(f"All trials saved to {trials_path}")
        
        # Generate visualizations if plotly is available
        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(str(output_path / "optimization_history.html"))
            
            # Parameter importances
            fig = plot_param_importances(self.study)
            fig.write_html(str(output_path / "param_importances.html"))
            
            # Parallel coordinate plot
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(str(output_path / "parallel_coordinate.html"))
            
            # Slice plot
            fig = plot_slice(self.study)
            fig.write_html(str(output_path / "slice_plot.html"))
            
            logger.info(f"Visualizations saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best hyperparameter configuration."""
        return self.study.best_params


def example_train_function(params: Dict[str, Any], data_path: str, max_epochs: int, trial: optuna.Trial):
    """
    Example training function for hyperparameter search.
    
    This is a template - replace with actual training logic.
    
    Args:
        params: Hyperparameter dictionary
        data_path: Path to training data
        max_epochs: Maximum training epochs
        trial: Optuna trial for intermediate reporting
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    logger.info(f"Training with params: {params}")
    
    # Pseudo-training loop
    for epoch in range(max_epochs):
        # Simulate training
        train_loss = 5.0 - epoch * 0.2  # Decreasing loss
        
        # Report intermediate value for pruning
        trial.report(train_loss, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Return dummy model and metrics
    return None, {"final_loss": train_loss}


def example_eval_function(model, data_path: str) -> float:
    """
    Example evaluation function.
    
    This is a template - replace with actual evaluation logic.
    
    Args:
        model: Trained model
        data_path: Path to evaluation data
        
    Returns:
        Evaluation metric value
    """
    # Return dummy validation loss
    return 3.5


def main():
    """CLI for hyperparameter search."""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for NMT")
    
    parser.add_argument("-data", required=True, help="Path to preprocessed data")
    parser.add_argument("-study_name", default="nmt_optimization", help="Study name")
    parser.add_argument("-n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("-max_epochs", type=int, default=20, help="Max epochs per trial")
    parser.add_argument("-timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("-output_dir", default="./optimization_results", help="Output directory")
    parser.add_argument("-sampler", default="tpe", choices=["tpe", "random", "grid"],
                       help="Sampling strategy")
    parser.add_argument("-pruner", default="median", choices=["median", "percentile", "hyperband"],
                       help="Pruning strategy")
    parser.add_argument("-direction", default="minimize", choices=["minimize", "maximize"],
                       help="Optimization direction")
    
    args = parser.parse_args()
    
    # Create search object
    search = NMTHyperparameterSearch(
        study_name=args.study_name,
        direction=args.direction,
        sampler=args.sampler,
        pruner=args.pruner
    )
    
    # Run optimization
    logger.info("Starting hyperparameter search...")
    logger.info("NOTE: Using example train/eval functions. Replace with actual implementations.")
    
    search.optimize(
        train_fn=example_train_function,
        eval_fn=example_eval_function,
        data_path=args.data,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        timeout=args.timeout
    )
    
    # Save results
    search.save_results(args.output_dir)
    
    # Print best configuration
    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    for key, value in search.get_best_config().items():
        print(f"{key:20s}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
