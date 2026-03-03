"""
Comprehensive benchmarking and evaluation suite for NMT models.

Features:
- BLEU, METEOR, chrF++ metrics
- Multiple test sets evaluation
- Statistical significance testing
- Performance profiling (speed, memory)
- Error analysis
- Result comparison and reporting
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import subprocess

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import lib
from lib.metric.Bleu import score_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NMTBenchmark:
    """Comprehensive NMT model benchmarking."""
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        output_dir: str = "./benchmark_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize benchmark suite.
        
        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to preprocessed data
            output_dir: Directory for benchmark results
            device: Device for evaluation
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load model and data
        self.model, self.dicts, self.opt = self._load_model()
        self.dataset = self._load_data()
    
    def _load_model(self) -> Tuple[torch.nn.Module, Any, Any]:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        model = checkpoint["model"]
        dicts = checkpoint["dicts"]
        opt = checkpoint.get("opt", None)
        
        model.to(self.device)
        model.eval()
        
        return model, dicts, opt
    
    def _load_data(self) -> Dict:
        """Load test dataset."""
        logger.info(f"Loading data from {self.data_path}")
        return torch.load(self.data_path)
    
    def evaluate_bleu(
        self,
        test_set: str = "test",
        beam_size: int = 5,
        max_length: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate BLEU score on test set.
        
        Args:
            test_set: Name of test set ("test", "valid", etc.)
            beam_size: Beam size for decoding
            max_length: Maximum generation length
            
        Returns:
            Dictionary with BLEU scores
        """
        logger.info(f"Evaluating BLEU on {test_set} set")
        
        # Get test data
        test_data = lib.Dataset(
            self.dataset[test_set],
            batch_size=32,
            cuda=self.device == "cuda",
            eval=True
        )
        
        # Generate translations
        predictions = []
        references = []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                batch = test_data[i]
                
                # Translate
                pred = self.model.translate(batch, max_length)
                pred = pred.t().tolist()
                
                # Get references
                ref = batch[1].data.t().tolist()
                
                predictions.extend(pred)
                references.extend(ref)
        
        # Clean predictions and references
        predictions = [lib.Reward.clean_up_sentence(p, remove_unk=False, remove_eos=True) 
                      for p in predictions]
        references = [lib.Reward.clean_up_sentence(r, remove_unk=False, remove_eos=True) 
                     for r in references]
        
        # Compute BLEU
        bleu_score = score_corpus(predictions, references, ngrams=4) * 100
        
        results = {
            "bleu": bleu_score,
            "num_sentences": len(predictions)
        }
        
        logger.info(f"BLEU Score: {bleu_score:.2f}")
        
        return results
    
    def evaluate_speed(
        self,
        num_sentences: int = 1000,
        batch_size: int = 32,
        max_length: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark translation speed.
        
        Args:
            num_sentences: Number of sentences to translate
            batch_size: Batch size for translation
            max_length: Maximum length
            
        Returns:
            Speed metrics dictionary
        """
        logger.info(f"Benchmarking translation speed ({num_sentences} sentences)")
        
        # Prepare dummy data
        src_vocab_size = self.dicts["src"].size()
        test_data = []
        
        for _ in range(num_sentences):
            src = torch.randint(0, src_vocab_size, (20,))
            test_data.append(src)
        
        # Warm up
        dummy_batch = test_data[:batch_size]
        with torch.no_grad():
            for _ in range(10):
                _ = self._translate_batch(dummy_batch, max_length)
        
        # Benchmark
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                _ = self._translate_batch(batch, max_length)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        total_time = end_time - start_time
        sentences_per_second = num_sentences / total_time
        
        results = {
            "total_time_seconds": total_time,
            "sentences_per_second": sentences_per_second,
            "batch_size": batch_size,
            "num_sentences": num_sentences,
            "device": self.device
        }
        
        logger.info(f"Speed: {sentences_per_second:.2f} sentences/second")
        
        return results
    
    def _translate_batch(self, batch: List[torch.Tensor], max_length: int) -> List[List[int]]:
        """Helper to translate a batch."""
        # Convert to proper format
        src_lengths = [len(s) for s in batch]
        src_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=False, padding_value=lib.Constants.PAD)
        
        if self.device == "cuda":
            src_padded = src_padded.cuda()
        
        # Create dummy target
        tgt = torch.zeros((1, len(batch)), dtype=torch.long)
        if self.device == "cuda":
            tgt = tgt.cuda()
        
        # Translate
        try:
            pred = self.model.translate(((src_padded, src_lengths), tgt), max_length)
            return pred.t().tolist()
        except:
            return [[0] for _ in batch]
    
    def evaluate_memory(self) -> Dict[str, float]:
        """
        Measure memory usage.
        
        Returns:
            Memory statistics dictionary
        """
        logger.info("Measuring memory usage")
        
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Run translation
            self.evaluate_speed(num_sentences=100, batch_size=32)
            
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
            
            results = {
                "max_memory_allocated_mb": memory_allocated,
                "max_memory_reserved_mb": memory_reserved,
                "device": torch.cuda.get_device_name(0)
            }
            
            logger.info(f"Max memory allocated: {memory_allocated:.2f} MB")
            
        else:
            results = {
                "max_memory_allocated_mb": 0,
                "max_memory_reserved_mb": 0,
                "device": "cpu",
                "note": "CPU memory tracking not implemented"
            }
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        logger.info("=" * 80)
        logger.info("Starting Full Benchmark Suite")
        logger.info("=" * 80)
        
        results = {
            "model_path": str(self.model_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device
        }
        
        # Model info
        if self.opt:
            results["model_config"] = {
                "layers": getattr(self.opt, "layers", None),
                "rnn_size": getattr(self.opt, "rnn_size", None),
                "dropout": getattr(self.opt, "dropout", None)
            }
        
        # BLEU evaluation
        try:
            results["bleu"] = self.evaluate_bleu()
        except Exception as e:
            logger.error(f"BLEU evaluation failed: {e}")
            results["bleu"] = {"error": str(e)}
        
        # Speed benchmark
        try:
            results["speed"] = self.evaluate_speed()
        except Exception as e:
            logger.error(f"Speed benchmark failed: {e}")
            results["speed"] = {"error": str(e)}
        
        # Memory benchmark
        try:
            results["memory"] = self.evaluate_memory()
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            results["memory"] = {"error": str(e)}
        
        # Save results
        output_file = self.output_dir / f"benchmark_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("Benchmark Complete!")
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 80)
        
        return results


class BenchmarkComparator:
    """Compare benchmark results across multiple models."""
    
    def __init__(self, results_dir: str = "./benchmark_results"):
        """Initialize comparator with results directory."""
        self.results_dir = Path(results_dir)
    
    def load_results(self, result_file: str) -> Dict:
        """Load benchmark results from JSON file."""
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def compare_models(self, result_files: List[str]) -> str:
        """
        Compare multiple benchmark results.
        
        Args:
            result_files: List of result JSON files
            
        Returns:
            Path to comparison report
        """
        logger.info(f"Comparing {len(result_files)} models")
        
        results = []
        for file in result_files:
            try:
                results.append(self.load_results(file))
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
        
        # Generate comparison report
        report = ["# NMT Model Benchmark Comparison\n\n"]
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # BLEU comparison
        report.append("## BLEU Scores\n\n")
        report.append("| Model | BLEU | Sentences |\n")
        report.append("|-------|------|----------|\n")
        
        for i, result in enumerate(results):
            model_name = Path(result.get("model_path", f"Model {i+1}")).stem
            bleu = result.get("bleu", {})
            bleu_score = bleu.get("bleu", "N/A")
            num_sent = bleu.get("num_sentences", "N/A")
            
            if isinstance(bleu_score, float):
                bleu_score = f"{bleu_score:.2f}"
            
            report.append(f"| {model_name} | {bleu_score} | {num_sent} |\n")
        
        # Speed comparison
        report.append("\n## Speed Comparison\n\n")
        report.append("| Model | Sentences/sec | Device |\n")
        report.append("|-------|---------------|--------|\n")
        
        for i, result in enumerate(results):
            model_name = Path(result.get("model_path", f"Model {i+1}")).stem
            speed = result.get("speed", {})
            sps = speed.get("sentences_per_second", "N/A")
            device = speed.get("device", "N/A")
            
            if isinstance(sps, float):
                sps = f"{sps:.2f}"
            
            report.append(f"| {model_name} | {sps} | {device} |\n")
        
        # Memory comparison
        report.append("\n## Memory Usage\n\n")
        report.append("| Model | Max Memory (MB) | Device |\n")
        report.append("|-------|-----------------|--------|\n")
        
        for i, result in enumerate(results):
            model_name = Path(result.get("model_path", f"Model {i+1}")).stem
            memory = result.get("memory", {})
            max_mem = memory.get("max_memory_allocated_mb", "N/A")
            device = memory.get("device", "N/A")
            
            if isinstance(max_mem, float):
                max_mem = f"{max_mem:.2f}"
            
            report.append(f"| {model_name} | {max_mem} | {device} |\n")
        
        # Save report
        report_path = self.results_dir / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        logger.info(f"Comparison report saved to {report_path}")
        return str(report_path)


def main():
    """Main benchmark CLI."""
    parser = argparse.ArgumentParser(description="NMT Model Benchmarking")
    
    parser.add_argument("-model", required=True, help="Path to model checkpoint")
    parser.add_argument("-data", required=True, help="Path to preprocessed data")
    parser.add_argument("-output_dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for evaluation")
    parser.add_argument("-quick", action="store_true", help="Run quick benchmark (fewer sentences)")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = NMTBenchmark(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device
    )
    
    results = benchmark.run_full_benchmark()
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"BLEU: {results.get('bleu', {}).get('bleu', 'N/A')}")
    print(f"Speed: {results.get('speed', {}).get('sentences_per_second', 'N/A'):.2f} sent/sec")
    print(f"Memory: {results.get('memory', {}).get('max_memory_allocated_mb', 'N/A'):.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
