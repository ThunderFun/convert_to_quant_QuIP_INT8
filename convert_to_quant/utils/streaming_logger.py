"""Streaming activity logger for monitoring CPU/GPU offloading decisions."""
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field


@dataclass
class StreamingDecision:
    """Record of a single streaming decision."""
    layer_name: str
    operation: str  # 'hadamard', 'hessian', 'ldlq', 'ortho'
    device: str     # 'cpu' or 'gpu'
    tensor_shape: tuple
    threshold: Union[int, float]
    reason: str
    timestamp: float = field(default_factory=time.time)


class StreamingLogger:
    """Logs streaming decisions for analysis and reporting."""
    
    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.decisions: List[StreamingDecision] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
    
    def start(self):
        """Start timing the streaming session."""
        self._start_time = time.time()
    
    def end(self):
        """End timing the streaming session."""
        self._end_time = time.time()
    
    def log_decision(self, decision: StreamingDecision):
        """Log a streaming decision."""
        if not self.enabled:
            return
        self.decisions.append(decision)
        
        if self.verbose:
            print(f"[Streaming] {decision.layer_name}: {decision.operation} -> {decision.device} "
                  f"({decision.reason})")
    
    def log_decision_simple(
        self,
        layer_name: str,
        operation: str,
        device: str,
        tensor_shape: tuple,
        threshold: Union[int, float],
        reason: str
    ):
        """Convenience method to create and log a decision."""
        self.log_decision(StreamingDecision(
            layer_name=layer_name,
            operation=operation,
            device=device,
            tensor_shape=tensor_shape,
            threshold=threshold,
            reason=reason
        ))
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.decisions:
            return {
                'total_decisions': 0,
                'gpu_operations': 0,
                'cpu_operations': 0,
                'by_operation': {},
                'duration_seconds': 0.0
            }
        
        total = len(self.decisions)
        gpu_count = sum(1 for d in self.decisions if d.device == 'gpu')
        cpu_count = total - gpu_count
        
        by_operation: Dict[str, Dict[str, int]] = {}
        for d in self.decisions:
            by_operation.setdefault(d.operation, {'gpu': 0, 'cpu': 0})
            by_operation[d.operation][d.device] += 1
        
        duration = 0.0
        if self._start_time is not None:
            end = self._end_time if self._end_time is not None else time.time()
            duration = end - self._start_time
        
        return {
            'total_decisions': total,
            'gpu_operations': gpu_count,
            'cpu_operations': cpu_count,
            'by_operation': by_operation,
            'duration_seconds': duration
        }
    
    def get_report_string(self) -> str:
        """Generate formatted report string."""
        summary = self.get_summary()
        if summary['total_decisions'] == 0:
            return "No streaming activity recorded."
        
        lines = [
            "\nStreaming Activity Summary:",
            "=" * 50,
            f"Total operations: {summary['total_decisions']}",
            f"GPU operations: {summary['gpu_operations']}",
            f"CPU operations: {summary['cpu_operations']}",
        ]
        
        if summary['duration_seconds'] > 0:
            lines.append(f"Duration: {summary['duration_seconds']:.1f}s")
        
        lines.append("\nBy operation:")
        for op, counts in summary['by_operation'].items():
            total_op = counts['gpu'] + counts['cpu']
            gpu_pct = (counts['gpu'] / total_op * 100) if total_op > 0 else 0
            lines.append(f"  {op:12s}: GPU={counts['gpu']:3d} ({gpu_pct:5.1f}%), CPU={counts['cpu']:3d}")
        
        return "\n".join(lines)
    
    def print_report(self):
        """Print formatted report to stdout."""
        print(self.get_report_string())
    
    def clear(self):
        """Clear all logged decisions."""
        self.decisions.clear()
        self._start_time = None
        self._end_time = None
