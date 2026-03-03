"""
Model export utilities for deployment and inference.

Supports:
- ONNX export for cross-platform deployment
- TorchScript export for production
- Model quantization for efficiency
- Model pruning
- Format conversion
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

import torch
import torch.nn as nn
import torch.quantization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("ONNX not installed. Install with: pip install onnx onnxruntime")


class ModelExporter:
    """Export NMT models to various formats."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./exported_models"
    ):
        """
        Initialize model exporter.
        
        Args:
            model_path: Path to PyTorch model checkpoint
            output_dir: Directory for exported models
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = checkpoint["model"]
        self.dicts = checkpoint["dicts"]
        self.opt = checkpoint.get("opt", None)
        
        self.model.eval()
    
    def export_torchscript(
        self,
        output_name: str = "model_torchscript.pt",
        optimize: bool = True
    ) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            output_name: Output filename
            optimize: Whether to optimize for inference
            
        Returns:
            Path to exported model
        """
        logger.info("Exporting to TorchScript...")
        
        try:
            # Use scripting (tracing may not work for all models)
            scripted_model = torch.jit.script(self.model)
            
            # Optimize for inference
            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            # Save
            output_path = self.output_dir / output_name
            torch.jit.save(scripted_model, str(output_path))
            
            logger.info(f"TorchScript model saved to {output_path}")
            
            # Save metadata
            metadata = {
                "format": "torchscript",
                "original_model": str(self.model_path),
                "optimized": optimize
            }
            
            metadata_path = self.output_dir / f"{Path(output_name).stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            logger.info("Falling back to trace mode...")
            
            try:
                # Create example input
                batch_size = 1
                seq_len = 10
                src = torch.randint(0, 1000, (seq_len, batch_size))
                tgt = torch.randint(0, 1000, (seq_len, batch_size))
                
                # Trace model
                traced_model = torch.jit.trace(self.model, ((src, [seq_len]), tgt))
                
                output_path = self.output_dir / output_name
                torch.jit.save(traced_model, str(output_path))
                
                logger.info(f"Traced model saved to {output_path}")
                return str(output_path)
                
            except Exception as e2:
                logger.error(f"Trace export also failed: {e2}")
                return None
    
    def export_onnx(
        self,
        output_name: str = "model.onnx",
        opset_version: int = 14,
        dynamic_axes: bool = True
    ) -> Optional[str]:
        """
        Export model to ONNX format.
        
        Args:
            output_name: Output filename
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes for variable lengths
            
        Returns:
            Path to exported model or None if failed
        """
        if not HAS_ONNX:
            logger.error("ONNX export requires onnx and onnxruntime packages")
            return None
        
        logger.info("Exporting to ONNX...")
        
        try:
            # Create dummy input
            batch_size = 1
            seq_len = 10
            src = torch.randint(0, 1000, (seq_len, batch_size))
            tgt = torch.randint(0, 1000, (seq_len, batch_size))
            dummy_input = ((src, [seq_len]), tgt)
            
            # Define dynamic axes if requested
            dynamic_axes_dict = None
            if dynamic_axes:
                dynamic_axes_dict = {
                    'input_0': {0: 'seq_len', 1: 'batch_size'},
                    'input_1': {0: 'seq_len', 1: 'batch_size'},
                    'output': {0: 'seq_len', 1: 'batch_size'}
                }
            
            # Export
            output_path = self.output_dir / output_name
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['source', 'target'],
                output_names=['output'],
                dynamic_axes=dynamic_axes_dict,
                verbose=False
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"ONNX model saved to {output_path}")
            logger.info(f"ONNX model verified successfully")
            
            # Save metadata
            metadata = {
                "format": "onnx",
                "opset_version": opset_version,
                "original_model": str(self.model_path),
                "dynamic_axes": dynamic_axes
            }
            
            metadata_path = self.output_dir / f"{Path(output_name).stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Test ONNX runtime
            self._test_onnx_runtime(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _test_onnx_runtime(self, onnx_path: str):
        """Test ONNX model with ONNX Runtime."""
        try:
            logger.info("Testing ONNX Runtime inference...")
            
            # Create session
            session = ort.InferenceSession(onnx_path)
            
            # Create test input
            src = torch.randint(0, 1000, (10, 1)).numpy()
            tgt = torch.randint(0, 1000, (10, 1)).numpy()
            
            # Run inference
            outputs = session.run(
                None,
                {'source': src, 'target': tgt}
            )
            
            logger.info(f"ONNX Runtime test passed. Output shape: {outputs[0].shape}")
            
        except Exception as e:
            logger.warning(f"ONNX Runtime test failed: {e}")
    
    def quantize_model(
        self,
        output_name: str = "model_quantized.pt",
        quantization_type: str = "dynamic"
    ) -> str:
        """
        Quantize model for reduced size and faster inference.
        
        Args:
            output_name: Output filename
            quantization_type: 'dynamic' or 'static'
            
        Returns:
            Path to quantized model
        """
        logger.info(f"Quantizing model ({quantization_type})...")
        
        try:
            if quantization_type == "dynamic":
                # Dynamic quantization (easier, no calibration needed)
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear, nn.LSTM, nn.LSTMCell},
                    dtype=torch.qint8
                )
            else:
                logger.warning("Static quantization not fully implemented")
                quantized_model = self.model
            
            # Save quantized model
            output_path = self.output_dir / output_name
            checkpoint = {
                "model": quantized_model,
                "dicts": self.dicts,
                "opt": self.opt,
                "quantized": True,
                "quantization_type": quantization_type
            }
            torch.save(checkpoint, output_path)
            
            # Compare sizes
            original_size = Path(self.model_path).stat().st_size / 1024**2  # MB
            quantized_size = output_path.stat().st_size / 1024**2  # MB
            reduction = (1 - quantized_size / original_size) * 100
            
            logger.info(f"Quantized model saved to {output_path}")
            logger.info(f"Size reduction: {reduction:.1f}% ({original_size:.1f}MB → {quantized_size:.1f}MB)")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return None
    
    def export_vocabulary(self, output_name: str = "vocabulary.json") -> str:
        """
        Export vocabulary for deployment.
        
        Args:
            output_name: Output filename
            
        Returns:
            Path to vocabulary file
        """
        logger.info("Exporting vocabulary...")
        
        vocab_data = {
            "source": {
                "size": self.dicts["src"].size(),
                "itos": {} if not hasattr(self.dicts["src"], "idxToLabel") else {
                    str(i): label for i, label in enumerate(self.dicts["src"].idxToLabel)
                }
            },
            "target": {
                "size": self.dicts["tgt"].size(),
                "itos": {} if not hasattr(self.dicts["tgt"], "idxToLabel") else {
                    str(i): label for i, label in enumerate(self.dicts["tgt"].idxToLabel)
                }
            }
        }
        
        output_path = self.output_dir / output_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vocabulary saved to {output_path}")
        return str(output_path)
    
    def export_config(self, output_name: str = "model_config.json") -> str:
        """
        Export model configuration.
        
        Args:
            output_name: Output filename
            
        Returns:
            Path to config file
        """
        logger.info("Exporting model configuration...")
        
        config = {}
        if self.opt:
            config = vars(self.opt) if hasattr(self.opt, '__dict__') else {}
        
        # Add model info
        config["model_type"] = type(self.model).__name__
        config["source_vocab_size"] = self.dicts["src"].size()
        config["target_vocab_size"] = self.dicts["tgt"].size()
        
        output_path = self.output_dir / output_name
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
        return str(output_path)
    
    def export_all(self) -> Dict[str, str]:
        """
        Export model in all available formats.
        
        Returns:
            Dictionary mapping format names to file paths
        """
        logger.info("=" * 80)
        logger.info("Exporting model in all formats...")
        logger.info("=" * 80)
        
        results = {}
        
        # TorchScript
        torchscript_path = self.export_torchscript()
        if torchscript_path:
            results["torchscript"] = torchscript_path
        
        # ONNX
        if HAS_ONNX:
            onnx_path = self.export_onnx()
            if onnx_path:
                results["onnx"] = onnx_path
        
        # Quantized
        quantized_path = self.quantize_model()
        if quantized_path:
            results["quantized"] = quantized_path
        
        # Vocabulary
        vocab_path = self.export_vocabulary()
        results["vocabulary"] = vocab_path
        
        # Config
        config_path = self.export_config()
        results["config"] = config_path
        
        logger.info("=" * 80)
        logger.info("Export complete!")
        logger.info(f"All files saved to: {self.output_dir}")
        logger.info("=" * 80)
        
        return results


def main():
    """CLI for model export."""
    parser = argparse.ArgumentParser(description="Export NMT models for deployment")
    
    parser.add_argument("-model", required=True, help="Path to PyTorch model checkpoint")
    parser.add_argument("-output_dir", default="./exported_models", help="Output directory")
    parser.add_argument("-format", default="all", 
                       choices=["all", "torchscript", "onnx", "quantized"],
                       help="Export format")
    parser.add_argument("-optimize", action="store_true", help="Optimize for inference")
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ModelExporter(args.model, args.output_dir)
    
    # Export
    if args.format == "all":
        results = exporter.export_all()
        
        print("\n" + "=" * 80)
        print("EXPORT SUMMARY")
        print("=" * 80)
        for format_name, path in results.items():
            print(f"{format_name:15s}: {path}")
        print("=" * 80)
        
    elif args.format == "torchscript":
        exporter.export_torchscript(optimize=args.optimize)
        
    elif args.format == "onnx":
        exporter.export_onnx()
        
    elif args.format == "quantized":
        exporter.quantize_model()


if __name__ == "__main__":
    main()
