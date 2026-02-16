"""
Backward compatibility wrapper for metrics collection.
This module re-exports from src.monitoring.metrics to maintain backward compatibility.
"""

# Re-export metrics collector from the new location
from src.monitoring.metrics import metrics_collector

__all__ = ['metrics_collector']