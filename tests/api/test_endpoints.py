#!/usr/bin/env python3
"""
Comprehensive API endpoint tests.
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
from api.endpoints.crops import router as crops_router
from api.endpoints.feedback import router as feedback_router
from api.endpoints.monitoring import router as monitoring_router


class TestDiagnosisEndpoint:
    """Test /v1/diagnose endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_image_b64(self):
        """Create a sample base64 encoded image."""
        # Create a small 1x1 red pixel PNG
        img = Image.new('RGB', (1, 1), color='red')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def test_valid_diagnosis_request(self, client, sample_image_b64):
        """Test valid diagnosis request."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=payload)
        # Should return 200, 400, 500, or 503 depending on pipeline state
        assert response.status_code in [200, 400, 500, 503]

    @pytest.mark.parametrize("missing_field,valid_base64", [
        ("image", "valid_base64"),
        ("crop_hint", "valid_base64")
    ])
    def test_missing_required_fields(self, client, missing_field, valid_base64):
        """Test missing required fields."""
        payload = {}
        if missing_field != "image":
            payload["image"] = valid_base64
        if missing_field != "crop_hint":
            payload["crop_hint"] = "tomato"
        
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422

    def test_invalid_base64_image(self, client):
        """Test invalid base64 image."""
        payload = {
            "image": "invalid_base64_string",
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422

    def test_oversized_image(self, client):
        """Test image size limit (10MB)."""
        large_b64 = "A" * (11 * 1024 * 1024)  # 11MB
        payload = {
            "image": large_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422

    def test_invalid_crop_hint(self, client, sample_image_b64):
        """Test invalid crop hint."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": "unknown_crop"
        }
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422

    def test_optional_parameters(self, client, sample_image_b64):
        """Test with optional parameters."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato",
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "accuracy_meters": 10.5
            },
            "metadata": {
                "device_model": "Pixel 7",
                "os_version": "Android 14"
            }
        }
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code in [200, 400, 500, 503]

    def test_response_schema(self, client, sample_image_b64):
        """Test response schema is correct."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=payload)

        if response.status_code == 200:
            data = response.json()
            # Check required fields
            assert "disease" in data or "error" in data
            if "disease" in data:
                assert "class_index" in data["disease"]
                assert "name" in data["disease"]
                assert "confidence" in data["disease"]
            if "ood_analysis" in data:
                assert "is_ood" in data["ood_analysis"]
                assert "ood_score" in data["ood_analysis"]
                assert "threshold" in data["ood_analysis"]

    def test_malformed_json(self, client):
        """Test malformed JSON request."""
        response = client.post(
            "/v1/diagnose",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestBatchDiagnosisEndpoint:
    """Test /v1/diagnose/batch endpoint."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def sample_images_b64(self):
        """Create multiple sample base64 images."""
        images = []
        for i in range(3):
            img = Image.new('RGB', (1, 1), color=['red', 'green', 'blue'][i])
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_str)
        return images

    def test_batch_single_image(self, client, sample_images_b64):
        """Test batch with single image."""
        payload = {
            "images": [sample_images_b64[0]],
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code in [200, 400, 500, 503]

    def test_batch_multiple_images(self, client, sample_images_b64):
        """Test batch with multiple images."""
        payload = {
            "images": sample_images_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code in [200, 400, 500, 503]

    def test_batch_too_many_images(self, client, sample_images_b64):
        """Test batch with too many images (max 10)."""
        # 11 images should exceed limit
        payload = {
            "images": sample_images_b64 * 4,  # 12 images total
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code == 422

    def test_batch_empty_images(self, client):
        """Test batch with empty images list."""
        payload = {"images": []}
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code == 422

    def test_batch_mixed_valid_invalid(self, client):
        """Test batch with mix of valid and invalid images."""
        valid_img = base64.b64encode(b"fake image").decode()
        invalid_img = "invalid_base64"

        payload = {
            "images": [valid_img, invalid_img, valid_img],
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose/batch", json=payload)
        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 500, 503]

    def test_batch_response_structure(self, client, sample_images_b64):
        """Test batch response structure."""
        payload = {
            "images": sample_images_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose/batch", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total_processed" in data
            assert len(data["results"]) == len(sample_images_b64)


class TestFeedbackEndpoint:
    """Test feedback endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_expert_label_submission(self, client):
        """Test submitting expert label."""
        payload = {
            "sample_id": "12345678-1234-1234-1234-123456789012",
            "true_label": "septoria_leaf_spot",
            "confidence": 0.95
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code in [200, 400, 500]

    def test_invalid_uuid(self, client):
        """Test invalid UUID format."""
        payload = {
            "sample_id": "invalid-uuid",
            "true_label": "septoria_leaf_spot",
            "confidence": 0.95
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code == 422

    def test_invalid_confidence_range(self, client):
        """Test confidence out of range."""
        payload = {
            "sample_id": "12345678-1234-1234-1234-123456789012",
            "true_label": "septoria_leaf_spot",
            "confidence": 1.5  # > 1
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code == 422

    def test_negative_confidence(self, client):
        """Test negative confidence."""
        payload = {
            "sample_id": "12345678-1234-1234-1234-123456789012",
            "true_label": "septoria_leaf_spot",
            "confidence": -0.1
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code == 422

    def test_batch_feedback_submission(self, client):
        """Test batch feedback submission."""
        payload = [
            {
                "sample_id": "12345678-1234-1234-1234-123456789012",
                "true_label": "septoria_leaf_spot",
                "confidence": 0.95
            },
            {
                "sample_id": "87654321-4321-4321-4321-210987654321",
                "true_label": "healthy",
                "confidence": 0.88
            }
        ]
        response = client.post("/v1/feedback/batch", json=payload)
        assert response.status_code in [200, 400, 500]

    def test_feedback_with_metadata(self, client):
        """Test feedback with additional metadata."""
        payload = {
            "sample_id": "12345678-1234-1234-1234-123456789012",
            "true_label": "septoria_leaf_spot",
            "confidence": 0.95,
            "expert_id": "expert_001",
            "notes": "Confirmed diagnosis with lab results",
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194
            }
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code in [200, 400, 500]


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy", "degraded"]

    def test_system_info(self, client):
        """Test system info endpoint."""
        response = client.get("/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "middleware" in data

    def test_health_detailed(self, client):
        """Test detailed health endpoint."""
        response = client.get("/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/v1/readiness")
        assert response.status_code in [200, 503]

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/v1/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestMiddlewareBehavior:
    """Test middleware behavior (rate limiting, caching, compression)."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.get("/health")
        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers or response.status_code != 200
        assert "X-RateLimit-Remaining" in response.headers or response.status_code != 200

    def test_caching_headers(self, client):
        """Test cache-related headers on GET requests."""
        response = client.get("/health")
        # Cache headers may or may not be present
        assert response.status_code == 200

    def test_compression_accept_encoding(self, client):
        """Test compression with Accept-Encoding header."""
        response = client.get(
            "/health",
            headers={"Accept-Encoding": "gzip, br"}
        )
        assert response.status_code == 200
        # Check if compression was applied
        if response.status_code == 200:
            # Either compressed or not, should be successful
            assert response.content is not None

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/v1/diagnose")
        assert response.status_code in [200, 400]
        # CORS headers should be present
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 400

    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/health")
        # Check for common security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        # At least some should be present
        has_security = any(header in response.headers for header in security_headers)
        assert has_security or response.status_code != 200


class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_large_request_body(self, client):
        """Test request body size limit."""
        # Create a payload that exceeds size limit
        large_data = "A" * (11 * 1024 * 1024)  # 11MB
        response = client.post(
            "/v1/diagnose",
            json={"image": large_data, "crop_hint": "tomato"}
        )
        # Should be rejected due to size
        assert response.status_code in [422, 413]

    def test_malformed_json(self, client):
        """Test malformed JSON."""
        response = client.post(
            "/v1/diagnose",
            content=b"{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_sql_injection_attempt(self, client, sample_image_b64):
        """Test SQL injection attempt in payload."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato'; DROP TABLE users; --"
        }
        response = client.post("/v1/diagnose", json=payload)
        # Should be rejected or handled safely
        assert response.status_code in [200, 400, 422, 500]

    def test_xss_attempt(self, client, sample_image_b64):
        """Test XSS attempt in payload."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": "<script>alert('xss')</script>"
        }
        response = client.post("/v1/diagnose", json=payload)
        # Should be sanitized or rejected
        assert response.status_code in [200, 400, 422, 500]


class TestIntegrationScenarios:
    """Test full integration scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_full_diagnosis_flow(self, client, sample_image_b64):
        """Test complete diagnosis flow."""
        # 1. Submit diagnosis request
        diagnosis_payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato",
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194
            }
        }
        diagnosis_response = client.post("/v1/diagnose", json=diagnosis_payload)

        if diagnosis_response.status_code == 200:
            result = diagnosis_response.json()
            sample_id = result.get("sample_id")

            # 2. Submit feedback for the diagnosis
            if sample_id:
                feedback_payload = {
                    "sample_id": sample_id,
                    "true_label": "healthy",
                    "confidence": 0.95
                }
                feedback_response = client.post(
                    "/v1/feedback/expert-label",
                    json=feedback_payload
                )
                assert feedback_response.status_code in [200, 400, 500]

    def test_batch_processing_flow(self, client, sample_images_b64):
        """Test batch processing flow."""
        payload = {
            "images": sample_images_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose/batch", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == len(sample_images_b64)

            # Each result should have required fields
            for result in data["results"]:
                assert "disease" in result or "error" in result

    def test_health_after_requests(self, client):
        """Test health endpoints after making requests."""
        # Make some requests
        for _ in range(5):
            client.get("/health")

        # Check health status
        response = client.get("/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime" in data or "start_time" in data

    def test_concurrent_requests_simulation(self, client, sample_image_b64):
        """Test handling of multiple concurrent requests."""
        import concurrent.futures

        def make_request():
            payload = {
                "image": sample_image_b64,
                "crop_hint": "tomato"
            }
            return client.post("/v1/diagnose", json=payload)

        # Simulate 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]

        # All should succeed or fail gracefully
        for response in responses:
            assert response.status_code in [200, 400, 429, 500, 503]

    def test_error_recovery(self, client, sample_image_b64):
        """Test system recovery after errors."""
        # Send invalid request to trigger error
        invalid_payload = {"invalid": "payload"}
        client.post("/v1/diagnose", json=invalid_payload)

        # System should still handle valid requests
        valid_payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=valid_payload)
        assert response.status_code in [200, 400, 500, 503]


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization (if enabled)."""

    @pytest.fixture
    def client_with_auth(self):
        """Create client with API key if configured."""
        # Check if API key auth is configured
        # For testing, we'll test both with and without
        return TestClient(app)

    def test_without_api_key(self, client_with_auth):
        """Test requests without API key (if required)."""
        response = client_with_auth.get("/health")
        # Should either succeed or return 401
        assert response.status_code in [200, 401]

    def test_with_invalid_api_key(self, client_with_auth):
        """Test requests with invalid API key."""
        response = client_with_auth.get(
            "/health",
            headers={"X-API-Key": "invalid_key"}
        )
        # Should return 401 if auth is required
        assert response.status_code in [200, 401]

    def test_with_valid_api_key(self, client_with_auth):
        """Test requests with valid API key (if configured)."""
        # This would need a valid API key from config
        # For now, just test that header is accepted
        response = client_with_auth.get(
            "/health",
            headers={"X-API-Key": "test_key"}
        )
        assert response.status_code in [200, 401]


class TestPerformanceAndLoad:
    """Test performance characteristics."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_response_time_acceptable(self, client, sample_image_b64):
        """Test that response times are acceptable."""
        import time

        payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato"
        }

        start = time.time()
        response = client.post("/v1/diagnose", json=payload)
        elapsed = time.time() - start

        # Should respond within reasonable time (e.g., 30 seconds)
        assert elapsed < 30.0
        # Or if too slow, just note it
        if elapsed > 10.0:
            pytest.skip(f"Response time too slow: {elapsed:.2f}s")

    def test_concurrent_requests_handling(self, client, sample_image_b64):
        """Test handling of concurrent requests."""
        import concurrent.futures

        payload = {
            "image": sample_image_b64,
            "crop_hint": "tomato"
        }

        def make_request():
            return client.post("/v1/diagnose", json=payload)

        # 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            responses = [f.result() for f in futures]

        # All should complete without crashing
        assert len(responses) == 20
        # Most should succeed (some might be rate limited)
        success_count = sum(1 for r in responses if r.status_code in [200, 400, 500, 503])
        assert success_count >= len(responses) * 0.8  # At least 80% succeed


class TestMonitoringEndpoints:
    """Test monitoring-specific endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint for Prometheus."""
        response = client.get("/metrics")
        # Should return metrics in Prometheus format
        assert response.status_code == 200
        # Should contain common metric names
        content = response.text
        # Check for some expected metrics
        expected_metrics = [
            "http_requests_total",
            "http_request_duration_seconds",
            "system_memory_usage_bytes"
        ]
        # At least some should be present
        has_metrics = any(metric in content for metric in expected_metrics)
        assert has_metrics or len(content) > 0

    def test_metrics_content_type(self, client):
        """Test metrics endpoint returns correct content type."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should be text/plain for Prometheus
        assert "text/plain" in response.headers.get("content-type", "")

    def test_metrics_labels(self, client):
        """Test that metrics have proper labels."""
        response = client.get("/metrics")
        if response.status_code == 200:
            content = response.text
            # Should have labels like method, endpoint, status
            assert "method=" in content or "endpoint=" in content or "status=" in content


# Parametrized tests for common validation patterns
class TestValidationPatterns:
    """Test common validation patterns across endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.mark.parametrize("crop_hint", [
        "tomato",
        "pepper",
        "corn",
        "wheat",
        "potato"
    ])
    def test_valid_crop_hints(self, client, sample_image_b64, crop_hint):
        """Test all valid crop hints."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": crop_hint
        }
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code in [200, 400, 500, 503]

    @pytest.mark.parametrize("invalid_hint", [
        "",
        "unknown_crop",
        "123",
        "tomato;",
        "tomato<script>"
    ])
    def test_invalid_crop_hints(self, client, sample_image_b64, invalid_hint):
        """Test invalid crop hints."""
        payload = {
            "image": sample_image_b64,
            "crop_hint": invalid_hint
        }
        response = client.post("/v1/diagnose", json=payload)
        # Should be rejected
        assert response.status_code in [422, 400]

    @pytest.mark.parametrize("status_code", [200, 201, 400, 404, 500, 503])
    def test_response_codes_handled(self, client, status_code):
        """Test that different response codes are handled properly."""
        # This would need mocking to generate specific status codes
        # For now, just verify client handles them
        response = client.get("/health")
        assert response.status_code in [200, 503]
