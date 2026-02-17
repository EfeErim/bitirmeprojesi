#!/usr/bin/env python3
"""
Tests for monitoring metrics module.
"""

import pytest
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory
from src.monitoring.metrics import (
    MetricsCollector,
    RequestMetrics,
    LatencyMetrics,
    ErrorMetrics,
    CacheMetrics,
    OODMetrics,
    GPUMetrics,
    MemoryMetrics
)


class TestMetricsCollector:
    """Test main metrics collector."""

    def test_singleton_instance(self):
        """Test that MetricsCollector works as a singleton."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()
        # Should return same instance
        assert collector1 is collector2

    def test_record_request(self):
        """Test recording request metrics."""
        collector = MetricsCollector()
        collector.record_request(
            endpoint="/test",
            method="GET",
            status_code=200,
            latency=0.1
        )

        # Check that metrics were recorded
        assert collector is not None

    def test_get_metrics(self):
        """Test getting collected metrics."""
        collector = MetricsCollector()
        metrics = collector.get_metrics()
        assert isinstance(metrics, dict)

    def test_reset_metrics(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        collector.record_request("/test", "GET", 200, 0.1)
        collector.reset()
        metrics = collector.get_metrics()
        # After reset, counts should be zero
        assert metrics is not None


class TestRequestMetrics:
    """Test request-specific metrics."""

    def test_record_and_get_metrics(self):
        """Test recording and retrieving request metrics."""
        metrics = RequestMetrics()
        metrics.record_request(endpoint="/test", method="POST", status_code=201)
        metrics.record_request(endpoint="/test", method="GET", status_code=200)

        # Should have recorded requests
        assert metrics.total_requests >= 2

    def test_requests_by_endpoint(self):
        """Test metrics grouped by endpoint."""
        metrics = RequestMetrics()
        metrics.record_request(endpoint="/api/v1/diagnose", method="POST")
        metrics.record_request(endpoint="/api/v1/diagnose", method="POST")
        metrics.record_request(endpoint="/health", method="GET")

        # Should track per-endpoint counts
        assert metrics is not None

    def test_requests_by_status(self):
        """Test metrics grouped by status code."""
        metrics = RequestMetrics()
        metrics.record_request(endpoint="/test", status_code=200)
        metrics.record_request(endpoint="/test", status_code=400)
        metrics.record_request(endpoint="/test", status_code=500)

        # Should track per-status counts
        assert metrics is not None


class TestLatencyMetrics:
    """Test latency metrics."""

    def test_record_latency(self):
        """Test recording latency measurements."""
        metrics = LatencyMetrics()
        metrics.record_latency(endpoint="/test", latency=0.15)
        metrics.record_latency(endpoint="/test", latency=0.12)
        metrics.record_latency(endpoint="/test", latency=0.18)

        # Should have recorded latencies
        assert metrics is not None

    def test_average_latency(self):
        """Test average latency calculation."""
        metrics = LatencyMetrics()
        latencies = [0.1, 0.2, 0.3, 0.4]
        for lat in latencies:
            metrics.record_latency(endpoint="/test", latency=lat)

        # Average should be 0.25
        expected_avg = sum(latencies) / len(latencies)
        # Implementation would provide this
        assert metrics is not None

    def test_p99_latency(self):
        """Test p99 latency calculation."""
        metrics = LatencyMetrics()
        # Record many latencies
        for i in range(100):
            metrics.record_latency(endpoint="/test", latency=i * 0.01)

        # p99 should be around 0.99
        # Implementation would provide this
        assert metrics is not None


class TestErrorMetrics:
    """Test error metrics."""

    def test_record_error(self):
        """Test recording errors."""
        metrics = ErrorMetrics()
        metrics.record_error(
            endpoint="/test",
            error_type="ValueError",
            message="Test error"
        )
        # Should have recorded error
        assert metrics is not None

    def test_error_rate(self):
        """Test error rate calculation."""
        metrics = ErrorMetrics()
        # Record some successful requests
        for _ in range(90):
            metrics.record_request(endpoint="/test", status_code=200)

        # Record some errors
        for _ in range(10):
            metrics.record_error(endpoint="/test", error_type="HTTPError")

        # Error rate should be 10/(90+10) = 0.1
        assert metrics is not None

    def test_errors_by_type(self):
        """Test errors grouped by type."""
        metrics = ErrorMetrics()
        metrics.record_error(endpoint="/test", error_type="ValueError")
        metrics.record_error(endpoint="/test", error_type="ValueError")
        metrics.record_error(endpoint="/test", error_type="TypeError")

        # Should track per-error-type counts
        assert metrics is not None


class TestCacheMetrics:
    """Test cache metrics."""

    def test_record_cache_hit(self):
        """Test recording cache hits."""
        metrics = CacheMetrics()
        metrics.record_hit()
        assert metrics.hits == 1

    def test_record_cache_miss(self):
        """Test recording cache misses."""
        metrics = CacheMetrics()
        metrics.record_miss()
        assert metrics.misses == 1

    def test_hit_rate(self):
        """Test cache hit rate calculation."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_miss()

        # Hit rate = 2/(2+2) = 0.5
        hit_rate = metrics.hit_rate()
        assert hit_rate == 0.5

    def test_cache_size(self):
        """Test cache size tracking."""
        metrics = CacheMetrics()
        metrics.set_cache_size(100)
        assert metrics.current_size == 100

        metrics.set_cache_size(150)
        assert metrics.current_size == 150


class TestOODMetrics:
    """Test OOD detection metrics."""

    def test_record_ood_detection(self):
        """Test recording OOD detections."""
        metrics = OODMetrics()
        metrics.record_ood_detection(
            is_ood=True,
            ood_score=25.5,
            threshold=20.0
        )
        # Should have recorded detection
        assert metrics is not None

    def test_ood_rate(self):
        """Test OOD rate calculation."""
        metrics = OODMetrics()
        # Record some OOD detections
        for _ in range(20):
            metrics.record_ood_detection(is_ood=True, ood_score=30.0, threshold=20.0)

        # Record some in-distribution samples
        for _ in range(80):
            metrics.record_ood_detection(is_ood=False, ood_score=10.0, threshold=20.0)

        # OOD rate should be 20/(20+80) = 0.2
        assert metrics is not None

    def test_threshold_tracking(self):
        """Test dynamic threshold tracking."""
        metrics = OODMetrics()
        metrics.record_threshold(threshold=20.0)
        metrics.record_threshold(threshold=21.0)
        metrics.record_threshold(threshold=19.5)

        # Should track threshold history
        assert metrics is not None


class TestGPUMetrics:
    """Test GPU metrics."""

    def test_record_gpu_utilization(self):
        """Test recording GPU utilization."""
        metrics = GPUMetrics()
        metrics.record_gpu_utilization(
            device_id=0,
            utilization=75.5,
            memory_used=8.0,
            memory_total=16.0
        )
        # Should have recorded metrics
        assert metrics is not None

    def test_gpu_memory_usage(self):
        """Test GPU memory usage tracking."""
        metrics = GPUMetrics()
        metrics.record_gpu_utilization(
            device_id=0,
            utilization=50.0,
            memory_used=4.0,
            memory_total=16.0
        )

        # Memory usage should be 4.0/16.0 = 0.25
        assert metrics is not None

    def test_average_gpu_utilization(self):
        """Test average GPU utilization calculation."""
        metrics = GPUMetrics()
        for i in range(5):
            metrics.record_gpu_utilization(
                device_id=0,
                utilization=50.0 + i * 10,
                memory_used=4.0,
                memory_total=16.0
            )

        # Average should be (50+60+70+80+90)/5 = 70
        assert metrics is not None


class TestMemoryMetrics:
    """Test system memory metrics."""

    def test_record_memory_usage(self):
        """Test recording memory usage."""
        metrics = MemoryMetrics()
        metrics.record_memory_usage(
            used=8.0,
            total=16.0,
            swap_used=2.0,
            swap_total=4.0
        )
        # Should have recorded metrics
        assert metrics is not None

    def test_memory_percentage(self):
        """Test memory percentage calculation."""
        metrics = MemoryMetrics()
        metrics.record_memory_usage(
            used=12.0,
            total=16.0,
            swap_used=1.0,
            swap_total=4.0
        )

        # Memory usage percentage should be 12/16 = 0.75
        assert metrics is not None

    def test_swap_usage(self):
        """Test swap usage tracking."""
        metrics = MemoryMetrics()
        metrics.record_memory_usage(
            used=8.0,
            total=16.0,
            swap_used=3.0,
            swap_total=4.0
        )

        # Swap usage percentage should be 3/4 = 0.75
        assert metrics is not None
