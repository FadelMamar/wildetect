"""
Model performance metrics and statistics tracking.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from statistics import mean, median
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Track model performance metrics."""
    
    inference_times: List[float] = field(default_factory=list)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    
    def record_inference_time(self, inference_time: float) -> None:
        """Record inference time for a prediction."""
        self.inference_times.append(inference_time)
        self.total_predictions += 1
    
    def record_success(self) -> None:
        """Record a successful prediction."""
        self.successful_predictions += 1
    
    def record_failure(self) -> None:
        """Record a failed prediction."""
        self.failed_predictions += 1
    
    def get_average_inference_time(self) -> float:
        """Get average inference time."""
        return mean(self.inference_times) if self.inference_times else 0.0
    
    def get_median_inference_time(self) -> float:
        """Get median inference time."""
        return median(self.inference_times) if self.inference_times else 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        total = self.successful_predictions + self.failed_predictions
        return (self.successful_predictions / total * 100) if total > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'success_rate': self.get_success_rate(),
            'average_inference_time': self.get_average_inference_time(),
            'median_inference_time': self.get_median_inference_time(),
            'min_inference_time': min(self.inference_times) if self.inference_times else 0.0,
            'max_inference_time': max(self.inference_times) if self.inference_times else 0.0,
            'accuracy_metrics': self.accuracy_metrics
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.inference_times.clear()
        self.accuracy_metrics.clear()
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0


class MetricsTracker:
    """Context manager for tracking inference metrics."""
    
    def __init__(self, metrics: ModelMetrics):
        self.metrics = metrics
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            inference_time = time.time() - self.start_time
            self.metrics.record_inference_time(inference_time)
            
            if exc_type is None:
                self.metrics.record_success()
            else:
                self.metrics.record_failure()
                logger.error(f"Prediction failed: {exc_val}") 