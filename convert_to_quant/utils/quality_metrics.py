import torch
from torch import Tensor
import math
from typing import Dict, Optional

def compute_mse(original: Tensor, quantized: Tensor) -> float:
    """Compute Mean Squared Error between original and quantized tensors."""
    # Ensure both are on CPU for metric computation to avoid device mismatch
    orig_f = original.to(device="cpu", dtype=torch.float32)
    quant_f = quantized.to(device="cpu", dtype=torch.float32)
    return torch.mean((orig_f - quant_f) ** 2).item()

def compute_sqnr(original: Tensor, quantized: Tensor) -> float:
    """
    Compute Signal-to-Quantization-Noise Ratio in dB.
    SQNR = 10 * log10(var(original) / var(original - quantized))
    """
    # Ensure both are on CPU for metric computation to avoid device mismatch
    orig_f = original.to(device="cpu", dtype=torch.float32)
    quant_f = quantized.to(device="cpu", dtype=torch.float32)
    
    noise = orig_f - quant_f
    
    signal_var = torch.var(orig_f)
    noise_var = torch.var(noise)
    
    if noise_var == 0:
        return float('inf')
    
    sqnr = 10 * torch.log10(signal_var / noise_var)
    return sqnr.item()

def compute_max_error(original: Tensor, quantized: Tensor) -> float:
    """Compute maximum absolute error."""
    # Ensure both are on CPU for metric computation to avoid device mismatch
    orig_f = original.to(device="cpu", dtype=torch.float32)
    quant_f = quantized.to(device="cpu", dtype=torch.float32)
    return torch.max(torch.abs(orig_f - quant_f)).item()

def compute_quality_metrics(original: Tensor, quantized: Tensor) -> Dict[str, float]:
    """Compute all quality metrics for a tensor pair."""
    return {
        "mse": compute_mse(original, quantized),
        "sqnr": compute_sqnr(original, quantized),
        "max_error": compute_max_error(original, quantized)
    }

class QualityReporter:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold
        self.reports = {}
        self.summary = {
            "mean_sqnr": 0.0,
            "below_threshold": 0,
            "total_layers": 0
        }

    def add_layer(self, name: str, original: Tensor, quantized: Tensor):
        metrics = compute_quality_metrics(original, quantized)
        self.reports[name] = metrics
        
        if metrics["sqnr"] < self.threshold:
            self.summary["below_threshold"] += 1
        
        self.summary["total_layers"] += 1
        
        # Update running mean SQNR
        n = self.summary["total_layers"]
        self.summary["mean_sqnr"] = (self.summary["mean_sqnr"] * (n - 1) + metrics["sqnr"]) / n

    def get_report_string(self) -> str:
        if not self.reports:
            return "No quality metrics collected."

        lines = [
            "Quantization Quality Report:",
            f"{'Layer':<50} | {'MSE':<10} | {'SQNR (dB)':<10} | {'Max Error':<10}",
            "-" * 88
        ]
        
        for name, metrics in sorted(self.reports.items()):
            lines.append(f"{name:<50} | {metrics['mse']:.8f} | {metrics['sqnr']:.2f}      | {metrics['max_error']:.4f}")
        
        lines.append("-" * 88)
        lines.append("Summary:")
        lines.append(f"  Mean SQNR: {self.summary['mean_sqnr']:.2f} dB")
        lines.append(f"  Layers below {self.threshold} dB: {self.summary['below_threshold']} / {self.summary['total_layers']}")
        
        return "\n".join(lines)
