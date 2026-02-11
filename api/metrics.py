"""
Metrics collection for monitoring API performance.
"""
import time
from typing import Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Container for metric data."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: int = 0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, duration: float, error: bool = False):
        """Add a measurement."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)
        if error:
            self.errors += 1
    
    @property
    def avg_time(self) -> float:
        """Average time."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def p95_time(self) -> float:
        """95th percentile time."""
        if not self.recent_times:
            return 0.0
        sorted_times = sorted(self.recent_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'count': self.count,
            'avg_time': round(self.avg_time, 4),
            'min_time': round(self.min_time, 4) if self.min_time != float('inf') else 0,
            'max_time': round(self.max_time, 4),
            'p95_time': round(self.p95_time, 4),
            'errors': self.errors,
            'error_rate': round(self.errors / self.count, 4) if self.count > 0 else 0
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.endpoints: Dict[str, MetricData] = defaultdict(MetricData)
        self.start_times: Dict[str, float] = {}
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
    
    def start_request(self, endpoint: str):
        """Mark start of request."""
        key = self._normalize_endpoint(endpoint)
        self.start_times[key] = time.time()
    
    def end_request(self, endpoint: str, error: bool = False):
        """Mark end of request."""
        key = self._normalize_endpoint(endpoint)
        start = self.start_times.pop(key, None)
        
        if start:
            duration = time.time() - start
            self.endpoints[key].add(duration, error)
            
            with self._lock:
                self.total_requests += 1
                if error:
                    self.total_errors += 1
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint path (remove IDs, etc)."""
        # Simple normalization - in production use more sophisticated routing
        import re
        # Replace UUIDs and IDs with placeholders
        normalized = re.sub(r'/[0-9a-fA-F-]{36}', '/{id}', endpoint)
        normalized = re.sub(r'/\d+', '/{id}', normalized)
        return normalized
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        uptime = time.time() - self.start_time
        
        with self._lock:
            metrics = {
                'uptime_seconds': round(uptime, 2),
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'error_rate': round(self.total_errors / self.total_requests, 4) if self.total_requests > 0 else 0,
                'requests_per_second': round(self.total_requests / uptime, 2) if uptime > 0 else 0,
                'endpoints': {}
            }
            
            for endpoint, data in self.endpoints.items():
                metrics['endpoints'][endpoint] = data.to_dict()
            
            return metrics


# Global metrics collector
metrics_collector = MetricsCollector()


def track_metrics(endpoint: str):
    """Decorator to track metrics for endpoint."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            metrics_collector.start_request(endpoint)
            try:
                result = await func(*args, **kwargs)
                metrics_collector.end_request(endpoint, error=False)
                return result
            except Exception as e:
                metrics_collector.end_request(endpoint, error=True)
                raise
        return wrapper
    return decorator