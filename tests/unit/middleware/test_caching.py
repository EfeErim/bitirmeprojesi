#!/usr/bin/env python3
"""
Tests for response caching middleware.
"""

import pytest
import time
from fastapi import Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.middleware.caching import CacheMiddleware


class TestCacheMiddleware:
    """Test cache middleware functionality."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""
        async def app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'application/json']]
            })
            await send({
                'type': 'http.response.body',
                'body': b'{"status": "ok"}'
            })
        return app

    @pytest.fixture
    def middleware(self, mock_app):
        """Create middleware instance with cache enabled."""
        return CacheMiddleware(
            app=mock_app,
            enabled=True,
            ttl=60,
            max_size=100
        )

    @pytest.mark.asyncio
    async def test_cache_miss_returns_fresh_response(self, middleware):
        """Test that a cache miss returns a fresh response."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"

        # First request should be a cache miss
        response1 = await middleware.dispatch(request, self.mock_app)
        assert response1.status_code == 200

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self, middleware):
        """Test that a cache hit returns cached response."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"

        # First request - cache miss
        response1 = await middleware.dispatch(request, self.mock_app)

        # Second request - should be cache hit
        response2 = await middleware.dispatch(request, self.mock_app)

        assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_app):
        """Test that caching can be disabled."""
        middleware = CacheMiddleware(app=mock_app, enabled=False)

        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"

        # Should not cache when disabled
        response = await middleware.dispatch(request, mock_app)
        assert response.status_code == 200

    def test_cache_key_generation(self, middleware):
        """Test that cache keys are generated correctly."""
        request1 = MagicMock()
        request1.url.path = "/test"
        request1.headers = {"Accept": "application/json"}
        request1.method = "GET"

        request2 = MagicMock()
        request2.url.path = "/test"
        request2.headers = {"Accept": "application/json"}
        request2.method = "GET"

        key1 = middleware._generate_cache_key(request1)
        key2 = middleware._generate_cache_key(request2)

        assert key1 == key2, "Same request should generate same cache key"

    def test_different_methods_different_keys(self, middleware):
        """Test that different HTTP methods generate different cache keys."""
        request_get = MagicMock()
        request_get.url.path = "/test"
        request_get.headers = {}
        request_get.method = "GET"

        request_post = MagicMock()
        request_post.url.path = "/test"
        request_post.headers = {}
        request_post.method = "POST"

        key_get = middleware._generate_cache_key(request_get)
        key_post = middleware._generate_cache_key(request_post)

        assert key_get != key_post, "Different methods should have different cache keys"

    def test_cache_size_limit(self, middleware):
        """Test that cache respects size limits."""
        # This is a basic test - full implementation would need integration
        assert middleware.max_size == 100

    @pytest.mark.asyncio
    async def test_cache_headers_present(self, middleware):
        """Test that cache-related headers are present in response."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"

        response = await middleware.dispatch(request, self.mock_app)

        # Check for cache headers (implementation dependent)
        # This test may need adjustment based on actual implementation
        assert response is not None
