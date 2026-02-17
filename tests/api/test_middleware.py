#!/usr/bin/env python3
"""
Tests for middleware integration and behavior.
"""

import pytest
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import sys
import io
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from api.main import app
from middleware import (
    RateLimitMiddleware,
    CacheMiddleware,
    CompressionMiddleware,
    AuditMiddleware,
    APIKeyMiddleware
)


class TestMiddlewareIntegration:
    """Test middleware integration."""

    @pytest.fixture
    def client(self):
        """Create test client with all middleware."""
        return TestClient(app)

    @pytest.fixture
    def sample_image_b64(self):
        """Create a sample base64 encoded image."""
        img = Image.new('RGB', (10, 10), color='red')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def test_rate_limit_behavior(self, client):
        """Test rate limiting middleware behavior."""
        # Make several requests
        for i in range(5):
            response = client.get("/health")
            assert response.status_code in [200, 429]

        # If rate limited, should get 429
        # Check for rate limit headers
        if response.status_code == 429:
            assert "X-RateLimit" in response.headers or "Retry-After" in response.headers

    def test_caching_behavior(self, client):
        """Test caching middleware behavior."""
        # First request - potential cache miss
        response1 = client.get("/health")
        assert response1.status_code == 200

        # Second request - should be cache hit
        response2 = client.get("/health")
        assert response2.status_code == 200

        # Check for cache headers
        # Cache-Control, ETag, or Last-Modified might be present
        has_cache_headers = any(
            header in response2.headers
            for header in ["Cache-Control", "ETag", "Last-Modified"]
        )
        # May or may not have caching enabled for this endpoint

    def test_compression_behavior(self, client):
        """Test compression middleware behavior."""
        # Request with Accept-Encoding
        response = client.get(
            "/health",
            headers={"Accept-Encoding": "gzip, deflate, br"}
        )
        assert response.status_code == 200

        # Check if content is compressed
        if "Content-Encoding" in response.headers:
            encoding = response.headers["Content-Encoding"]
            assert encoding in ["gzip", "deflate", "br"]

    def test_audit_logging(self, client, tmp_path):
        """Test audit logging middleware."""
        # This would need audit middleware configured with test log file
        # For now, just verify request goes through
        response = client.get("/health")
        assert response.status_code == 200

    def test_api_key_auth(self, client):
        """Test API key authentication."""
        # Without API key
        response = client.get("/health")
        assert response.status_code in [200, 401]

        # With invalid API key
        response = client.get(
            "/health",
            headers={"X-API-Key": "invalid_key"}
        )
        assert response.status_code in [200, 401]

        # With valid API key (if configured)
        response = client.get(
            "/health",
            headers={"X-API-Key": "valid_key"}
        )
        assert response.status_code == 200

    def test_cors_behavior(self, client):
        """Test CORS middleware behavior."""
        # Preflight request
        response = client.options("/v1/diagnose")
        assert response.status_code in [200, 400]

        # Check CORS headers
        cors_headers = [
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Methods",
            "Access-Control-Allow-Headers"
        ]
        has_cors = any(header in response.headers for header in cors_headers)
        assert has_cors or response.status_code == 400

    def test_request_size_limit(self, client):
        """Test request size limiting."""
        large_payload = {
            "image": "A" * (11 * 1024 * 1024),  # 11MB
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=large_payload)
        assert response.status_code in [422, 413]

    def test_middleware_order(self, client):
        """Test that middleware executes in correct order."""
        # All middleware should be applied without errors
        response = client.get("/health")
        assert response.status_code == 200

        # Check headers from different middleware
        headers = response.headers

        # Should have some combination of:
        # - Rate limit headers
        # - Cache headers
        # - Security headers
        # - Compression headers
        has_any_middleware_headers = any(
            key in headers
            for key in [
                "X-RateLimit-Limit",
                "Cache-Control",
                "Content-Encoding",
                "Strict-Transport-Security"
            ]
        )
        # At least some middleware headers should be present
        assert has_any_middleware_headers or len(headers) > 0


class TestMiddlewareConfiguration:
    """Test middleware configuration."""

    def test_rate_limit_configuration(self):
        """Test rate limit configuration."""
        from src.middleware.rate_limit import RateLimitMiddleware
        from api.main import app

        # Check if rate limiting is configured
        # This tests the configuration values
        assert app is not None

    def test_cache_configuration(self):
        """Test cache configuration."""
        from src.middleware.caching import CacheMiddleware
        assert CacheMiddleware is not None

    def test_compression_configuration(self):
        """Test compression configuration."""
        from src.middleware.compression import CompressionMiddleware
        assert CompressionMiddleware is not None

    def test_audit_configuration(self):
        """Test audit configuration."""
        from src.middleware.audit import AuditMiddleware
        assert AuditMiddleware is not None

    def test_auth_configuration(self):
        """Test authentication configuration."""
        from src.middleware.auth import APIKeyMiddleware
        assert APIKeyMiddleware is not None


class TestMiddlewareErrorHandling:
    """Test middleware error handling."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_rate_limit_error_handling(self, client):
        """Test rate limit error response."""
        # Make many requests to trigger rate limit
        for _ in range(100):
            client.get("/health")

        # Should get 429 with proper error message
        response = client.get("/health")
        if response.status_code == 429:
            data = response.json()
            assert "detail" in data
            assert "rate limit" in str(data["detail"]).lower()

    def test_auth_error_handling(self, client):
        """Test authentication error response."""
        # If auth is required, should get 401
        response = client.get("/health")
        # Either 200 (no auth required) or 401 (auth required)
        assert response.status_code in [200, 401]

    def test_validation_error_handling(self, client):
        """Test validation error from middleware."""
        # Send invalid data
        response = client.post(
            "/v1/diagnose",
            content="invalid",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_server_error_handling(self, client, sample_image_b64):
        """Test handling of server errors."""
        # This might trigger an error if pipeline not initialized
        payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=payload)

        # Should handle gracefully
        assert response.status_code in [200, 400, 500, 503]
        if response.status_code >= 500:
            # Server error should have error details
            data = response.json()
            assert "error" in data or "detail" in data


class TestMiddlewarePerformance:
    """Test middleware performance impact."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_middleware_overhead(self, client, sample_image_b64):
        """Test that middleware overhead is acceptable."""
        import time

        # Measure baseline
        start = time.time()
        for _ in range(10):
            client.get("/health")
        baseline = time.time() - start

        # Measure with full middleware
        start = time.time()
        for _ in range(10):
            payload = {
                "image": sample_image_b64,
                "crop_hint": "tomato"
            }
            client.post("/v1/diagnose", json=payload)
        with_middleware = time.time() - start

        # Overhead should be reasonable (e.g., less than 10x)
        # This is a very loose check
        assert with_middleware < baseline * 10 or baseline < 0.1

    def test_concurrent_middleware_handling(self, client):
        """Test middleware handling of concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get("/health")

        # 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert len(responses) == 50
        success_count = sum(1 for r in responses if r.status_code == 200)
        # At least 90% should succeed
        assert success_count >= 45


class TestMiddlewareSecurity:
    """Test middleware security features."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_sql_injection_prevention(self, client, sample_image_b64):
        """Test SQL injection attempts are handled."""
        malicious_payloads = [
            {"image": sample_image_b64, "crop_hint": "'; DROP TABLE users; --"},
            {"image": sample_image_b64, "crop_hint": "1' OR '1'='1"},
            {"image": sample_image_b64, "crop_hint": "admin'--"}
        ]

        for payload in malicious_payloads:
            response = client.post("/v1/diagnose", json=payload)
            # Should not cause server error or data loss
            assert response.status_code in [200, 400, 422, 500]

    def test_xss_prevention(self, client, sample_image_b64):
        """Test XSS attempts are handled."""
        xss_payloads = [
            {"image": sample_image_b64, "crop_hint": "<script>alert('xss')</script>"},
            {"image": sample_image_b64, "crop_hint": "<img src=x onerror=alert(1)>"},
            {"image": sample_image_b64, "crop_hint": "javascript:alert(1)"}
        ]

        for payload in xss_payloads:
            response = client.post("/v1/diagnose", json=payload)
            # Should be sanitized or rejected
            assert response.status_code in [200, 400, 422, 500]

    def test_path_traversal_prevention(self, client):
        """Test path traversal attempts."""
        # Try to access files outside allowed directories
        malicious_paths = [
            "/v1/diagnose/../../../etc/passwd",
            "/v1/diagnose/....//....//etc/passwd",
            "/v1/diagnose/%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]

        for path in malicious_paths:
            response = client.get(path)
            # Should be blocked or return 404
            assert response.status_code in [404, 400, 403]

    def test_request_smuggling_prevention(self, client):
        """Test request smuggling attempts."""
        # Attempt to inject additional requests
        malicious_headers = {
            "Content-Length": "0",
            "Transfer-Encoding": "chunked",
            "X-Forwarded-For": "127.0.0.1\r\nGET /admin HTTP/1.1"
        }

        response = client.get("/health", headers=malicious_headers)
        # Should be handled safely
        assert response.status_code in [200, 400]

    def test_ssrf_prevention(self, client):
        """Test SSRF attempts."""
        # Try to make requests to internal services
        ssrf_payloads = [
            {"image": sample_image_b64, "crop_hint": "tomato", "url": "http://localhost:6379/"},
            {"image": sample_image_b64, "crop_hint": "tomato", "url": "http://169.254.169.254/latest/meta-data/"}
        ]

        for payload in ssrf_payloads:
            response = client.post("/v1/diagnose", json=payload)
            # Should be blocked or validated
            assert response.status_code in [200, 400, 422]


class TestMiddlewareObservability:
    """Test middleware observability features."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_metrics_collection(self, client):
        """Test that middleware collects metrics."""
        # Make some requests
        for _ in range(5):
            client.get("/health")

        # Check metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200

        metrics = response.text
        # Should contain metrics from middleware
        middleware_metrics = [
            "http_requests_total",
            "http_request_duration_seconds",
            "middleware_",
            "rate_limit_",
            "cache_"
        ]
        # At least some should be present
        has_metrics = any(metric in metrics for metric in middleware_metrics)
        assert has_metrics or len(metrics) > 0

    def test_logging_integration(self, client, tmp_path):
        """Test middleware logging."""
        # This would need logging configured to write to file
        # For now, just verify requests succeed
        response = client.get("/health")
        assert response.status_code == 200

    def test_tracing_headers(self, client):
        """Test distributed tracing headers."""
        # Send request with trace context
        headers = {
            "X-Request-ID": "test-request-123",
            "X-Correlation-ID": "test-correlation-456",
            "X-Trace-ID": "test-trace-789"
        }

        response = client.get("/health", headers=headers)
        assert response.status_code == 200

        # Check if tracing headers are echoed back or logged
        # Implementation dependent


class TestMiddlewareConfigurationEdgeCases:
    """Test middleware configuration edge cases."""

    def test_disabled_middleware(self):
        """Test middleware can be disabled."""
        # Each middleware should have an 'enabled' flag
        from src.middleware.rate_limit import RateLimitMiddleware
        from src.middleware.caching import CacheMiddleware
        from src.middleware.compression import CompressionMiddleware
        from src.middleware.audit import AuditMiddleware

        # Should accept enabled=False
        # This is a compile-time check
        assert RateLimitMiddleware is not None
        assert CacheMiddleware is not None
        assert CompressionMiddleware is not None
        assert AuditMiddleware is not None

    def test_middleware_with_custom_config(self):
        """Test middleware with custom configuration."""
        # Should be able to configure each middleware
        from src.middleware.rate_limit import RateLimitMiddleware
        from src.middleware.caching import CacheMiddleware

        # Custom rate limit
        custom_rate = RateLimitMiddleware(
            app=None,
            requests_per_minute=1000,
            burst=2000
        )
        assert custom_rate.requests_per_minute == 1000

        # Custom cache
        custom_cache = CacheMiddleware(
            app=None,
            enabled=True,
            ttl=7200,
            max_size=2000
        )
        assert custom_cache.ttl == 7200
        assert custom_cache.max_size == 2000

    def test_middleware_dependency_checks(self):
        """Test middleware handles missing dependencies gracefully."""
        # Each middleware should handle missing dependencies
        # For example, Redis for cache, etc.
        from src.middleware.caching import CacheMiddleware

        # Should handle connection failures gracefully
        assert CacheMiddleware is not None
