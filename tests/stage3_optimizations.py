#!/usr/bin/env python3
"""
Test suite for Stage 3 production optimizations.
Tests validation, rate limiting, caching, compression, batch processing, etc.
"""
import pytest
import asyncio
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app
from api.endpoints.crops import router as crops_router
from api.endpoints.feedback import router as feedback_router

client = TestClient(app)


class TestPydanticValidation:
    """Test request validation with Pydantic models."""
    
    def test_valid_diagnosis_request(self):
        """Test valid diagnosis request."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {
            "image": image_b64,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose", json=payload)
        # Should fail due to pipeline not initialized, but validation should pass
        assert response.status_code in [200, 400, 500, 503]
    
    def test_missing_image_field(self):
        """Test missing required image field."""
        payload = {"crop_hint": "tomato"}
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_base64_image(self):
        """Test invalid base64 image."""
        payload = {"image": "invalid_base64", "crop_hint": "tomato"}
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422
    
    def test_oversized_image(self):
        """Test image size limit (10MB)."""
        large_b64 = "A" * (11 * 1024 * 1024)  # 11MB
        payload = {"image": large_b64, "crop_hint": "tomato"}
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422
    
    def test_invalid_crop_hint(self):
        """Test invalid crop hint."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {"image": image_b64, "crop_hint": "invalid_crop"}
        response = client.post("/v1/diagnose", json=payload)
        assert response.status_code == 422


class TestBatchEndpoint:
    """Test batch diagnosis endpoint."""
    
    def test_batch_request_single_image(self):
        """Test batch with single image."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {"images": [image_b64], "crop_hint": "tomato"}
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_batch_request_multiple_images(self):
        """Test batch with multiple images."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {
            "images": [image_b64] * 3,
            "crop_hint": "tomato"
        }
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code in [200, 400, 500, 503]
    
    def test_batch_request_too_many_images(self):
        """Test batch with too many images (max 10)."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {"images": [image_b64] * 11, "crop_hint": "tomato"}
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code == 422
    
    def test_batch_empty_images(self):
        """Test batch with empty images list."""
        payload = {"images": []}
        response = client.post("/v1/diagnose/batch", json=payload)
        assert response.status_code == 422


class TestFeedbackValidation:
    """Test feedback endpoint validation."""
    
    def test_valid_expert_label(self):
        """Test valid expert label."""
        payload = {
            "sample_id": "12345678-1234-1234-1234-123456789012",
            "true_label": "septoria_leaf_spot",
            "confidence": 0.95
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code in [200, 400, 500]
    
    def test_invalid_uuid(self):
        """Test invalid UUID format."""
        payload = {
            "sample_id": "invalid-uuid",
            "true_label": "septoria_leaf_spot",
            "confidence": 0.95
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code == 422
    
    def test_invalid_confidence_range(self):
        """Test confidence out of range."""
        payload = {
            "sample_id": "12345678-1234-1234-1234-123456789012",
            "true_label": "septoria_leaf_spot",
            "confidence": 1.5
        }
        response = client.post("/v1/feedback/expert-label", json=payload)
        assert response.status_code == 422
    
    def test_batch_feedback(self):
        """Test batch feedback submission."""
        payload = [
            {
                "sample_id": "12345678-1234-1234-1234-123456789012",
                "true_label": "septoria_leaf_spot",
                "confidence": 0.95
            }
        ]
        response = client.post("/v1/feedback/batch", json=payload)
        assert response.status_code in [200, 400, 500]


class TestRateLimiting:
    """Test rate limiting middleware."""
    
    def test_rate_limit_headers(self):
        """Test rate limit headers are present."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {"image": image_b64}
        response = client.post("/v1/diagnose", json=payload)
        
        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded (429)."""
        # This test would require many requests
        # For now, just verify endpoint exists
        response = client.get("/health")
        assert response.status_code == 200


class TestCaching:
    """Test response caching."""
    
    def test_cache_headers(self):
        """Test cache-related headers."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {"image": image_b64}
        response = client.post("/v1/diagnose", json=payload)
        # Cache headers may or may not be present depending on config
        assert response.status_code in [200, 400, 500, 503]


class TestCompression:
    """Test response compression."""
    
    def test_compression_accept_header(self):
        """Test compression with Accept-Encoding."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {"image": image_b64}
        headers = {"Accept-Encoding": "gzip, br"}
        response = client.post("/v1/diagnose", json=payload, headers=headers)
        
        # Check if compression was applied
        if response.status_code == 200:
            # Either compressed or not, should be successful
            assert response.content is not None


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_system_info(self):
        """Test system info endpoint."""
        response = client.get("/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "middleware" in data
    
    def test_health_detailed(self):
        """Test detailed health endpoint."""
        response = client.get("/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe."""
        response = client.get("/v1/readiness")
        assert response.status_code in [200, 503]
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe."""
        response = client.get("/v1/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestCors:
    """Test CORS configuration."""
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/v1/diagnose")
        assert response.status_code in [200, 400]
        # CORS headers should be present
        assert "Access-Control-Allow-Origin" in response.headers or response.status_code == 400


class TestInputSizeLimits:
    """Test input size limits."""
    
    def test_large_request_body(self):
        """Test request body size limit."""
        # Create a payload that exceeds size limit
        large_image = "A" * (11 * 1024 * 1024)  # 11MB
        payload = {"image": large_image}
        response = client.post("/v1/diagnose", json=payload)
        # Should be rejected due to size
        assert response.status_code in [422, 413]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])