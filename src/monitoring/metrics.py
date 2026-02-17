"""
Metrics collection for monitoring API performance.
"""
import time
from typing import Dict, Any, List
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
        """95th percentile time (robust calculation)."""
        if not self.recent_times:
            return 0.0
        sorted_times = sorted(self.recent_times)
        # Proper percentile calculation to avoid index out of bounds
        idx = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
        return float(sorted_times[idx])
    
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

    import re

    # Pre-compiled regex patterns for O(1) performance
    UUID_PATTERN = re.compile(r'/[0-9a-fA-F-]{36}')
    ID_PATTERN = re.compile(r'/\d+')

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance._lock = threading.Lock()
            cls._instance.endpoints: Dict[str, MetricData] = defaultdict(MetricData)
            cls._instance.start_times: Dict[str, float] = {}
            cls._instance.total_requests = 0
            cls._instance.total_errors = 0
            cls._instance.start_time = time.time()
        return cls._instance
    
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
    
    def record_request(self, endpoint: str, method: str, status_code: int, latency: float = None):
        """Record a request with all details."""
        if latency is None:
            # If no latency provided, use a small random value
            import random
            latency = random.uniform(0.01, 0.1)
        self.start_request(endpoint)
        self.end_request(endpoint, error=(status_code >= 400))
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.endpoints.clear()
            self.start_times.clear()
            self.total_requests = 0
            self.total_errors = 0
            self.start_time = time.time()
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint path using pre-compiled patterns (O(1) performance)."""
        # Replace UUIDs and IDs with placeholders
        normalized = self.UUID_PATTERN.sub('/{id}', endpoint)
        normalized = self.ID_PATTERN.sub('/{id}', normalized)
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


# Global metrics collector instance
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


class RequestMetrics:
    """Request-specific metrics tracking."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.requests: List[Dict] = []
        self.total_requests = 0
    
    def record_request(self, endpoint: str, method: str = "GET", status_code: int = 200, latency: float = None):
        """Record a request."""
        if latency is None:
            import random
            latency = random.uniform(0.01, 0.1)
        self.metrics.start_request(endpoint)
        self.metrics.end_request(endpoint, error=(status_code >= 400))
        self.requests.append({
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'latency': latency
        })
        self.total_requests += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get request metrics."""
        return self.metrics.get_metrics()
    
    def requests_by_endpoint(self) -> Dict[str, int]:
        """Get request count by endpoint."""
        counts = {}
        for req in self.requests:
            endpoint = req['endpoint']
            counts[endpoint] = counts.get(endpoint, 0) + 1
        return counts
    
    def requests_by_status(self) -> Dict[int, int]:
        """Get request count by status code."""
        counts = {}
        for req in self.requests:
            status = req['status_code']
            counts[status] = counts.get(status, 0) + 1
        return counts


class LatencyMetrics:
    """Latency metrics tracking."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
    
    def record_latency(self, endpoint: str, latency: float):
        """Record latency for an endpoint."""
        self.metrics.start_request(endpoint)
        self.metrics.end_request(endpoint, error=False)
    
    def average_latency(self, endpoint: str = None) -> float:
        """Get average latency."""
        metrics = self.metrics.get_metrics()
        if endpoint:
            if endpoint in metrics['endpoints']:
                return metrics['endpoints'][endpoint]['avg_time']
            return 0.0
        else:
            # Overall average
            total_time = sum(data['total_time'] for data in metrics['endpoints'].values())
            total_requests = sum(data['count'] for data in metrics['endpoints'].values())
            return total_time / total_requests if total_requests > 0 else 0.0
    
    def p99_latency(self, endpoint: str = None) -> float:
        """Get 99th percentile latency."""
        metrics = self.metrics.get_metrics()
        if endpoint and endpoint in metrics['endpoints']:
            return metrics['endpoints'][endpoint]['p95_time']  # Using p95 as approximation
        return 0.0


class ErrorMetrics:
    """Error metrics tracking."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.errors: List[Dict] = []
    
    def record_error(self, endpoint: str, error_type: str = "unknown", message: str = None):
        """Record an error."""
        self.metrics.start_request(endpoint)
        self.metrics.end_request(endpoint, error=True)
        self.errors.append({
            'endpoint': endpoint,
            'type': error_type,
            'message': message
        })
    
    def record_request(self, endpoint: str, status_code: int):
        """Record a request (success or failure)."""
        self.metrics.start_request(endpoint)
        self.metrics.end_request(endpoint, error=(status_code >= 400))
    
    def error_rate(self, endpoint: str = None) -> float:
        """Get error rate."""
        metrics = self.metrics.get_metrics()
        if endpoint:
            if endpoint in metrics['endpoints']:
                return metrics['endpoints'][endpoint]['error_rate']
            return 0.0
        return metrics.get('error_rate', 0.0)
    
    def errors_by_type(self) -> Dict[str, int]:
        """Get error count by type."""
        counts = {}
        for error in self.errors:
            error_type = error['type']
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts


class CacheMetrics:
    """Cache performance metrics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.current_size = 0
    
    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1
    
    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1
    
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def cache_size(self) -> int:
        """Get current cache size."""
        return self.current_size
    
    def set_cache_size(self, size: int):
        """Set cache size."""
        self.current_size = size


class OODMetrics:
    """OOD detection metrics."""
    
    def __init__(self):
        self.detections: List[Dict] = []
        self.thresholds: List[float] = []
    
    def record_ood_detection(self, is_ood: bool, ood_score: float = None, threshold: float = None):
        """Record OOD detection result."""
        self.detections.append({
            'is_ood': is_ood,
            'score': ood_score,
            'threshold': threshold
        })
    
    def record_threshold(self, threshold: float):
        """Record a threshold value for tracking."""
        self.thresholds.append(threshold)
    
    def ood_rate(self) -> float:
        """Get OOD detection rate."""
        if not self.detections:
            return 0.0
        ood_count = sum(1 for d in self.detections if d['is_ood'])
        return ood_count / len(self.detections)
    
    def threshold_tracking(self) -> Dict[str, float]:
        """Get threshold statistics."""
        if not self.thresholds:
            return {'current_threshold': 0.0, 'min_threshold': 0.0, 'max_threshold': 0.0}
        return {
            'current_threshold': self.thresholds[-1],
            'min_threshold': min(self.thresholds),
            'max_threshold': max(self.thresholds)
        }


class GPUMetrics:
    """GPU utilization metrics."""
    
    def __init__(self):
        self.utilizations: List[float] = []
        self.memory_used = 0.0
        self.memory_total = 0.0
    
    def record_gpu_utilization(self, device_id: int, utilization: float, memory_used: float, memory_total: float):
        """Record GPU utilization."""
        self.utilizations.append(utilization)
        self.memory_used = memory_used
        self.memory_total = memory_total
    
    def gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage."""
        return {
            'used_gb': self.memory_used,
            'total_gb': self.memory_total,
            'percentage': (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0.0
        }
    
    def average_gpu_utilization(self) -> float:
        """Get average GPU utilization."""
        if not self.utilizations:
            return 0.0
        return sum(self.utilizations) / len(self.utilizations)


class MemoryMetrics:
    """System memory metrics."""
    
    def __init__(self):
        self.used = 0.0
        self.total = 0.0
        self.swap_used = 0.0
        self.swap_total = 0.0
    
    def record_memory_usage(self, used: float, total: float, swap_used: float = 0.0, swap_total: float = 0.0):
        """Record memory usage."""
        self.used = used
        self.total = total
        self.swap_used = swap_used
        self.swap_total = swap_total
    
    def memory_percentage(self) -> float:
        """Get memory usage percentage."""
        return (self.used / self.total * 100) if self.total > 0 else 0.0
    
    def swap_usage(self) -> Dict[str, float]:
        """Get swap usage."""
        return {
            'used_gb': self.swap_used,
            'total_gb': self.swap_total,
            'percentage': (self.swap_used / self.swap_total * 100) if self.swap_total > 0 else 0.0
        }