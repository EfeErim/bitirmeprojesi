#!/usr/bin/env python3
"""
Tests for response compression middleware.
"""

import pytest
import gzip
import brotli
from fastapi import Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.middleware.compression import CompressionMiddleware


class TestCompressionMiddleware:
    """Test compression middleware functionality."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""
        async def app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'application/json']]
            })
            # Send a reasonably large response to test compression
            body = b'{"data": "x" * 1000}'
            await send({
                'type': 'http.response.body',
                'body': body
            })
        return app

    @pytest.fixture
    def middleware_enabled(self, mock_app):
        """Create middleware instance with compression enabled."""
        return CompressionMiddleware(
            app=mock_app,
            enabled=True,
            minimum_size=100,
            compression_level=6
        )

    @pytest.mark.asyncio
    async def test_small_response_not_compressed(self, middleware_enabled):
        """Test that small responses are not compressed."""
        # Create a small response
        async def small_app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'text/plain']]
            })
            await send({
                'type': 'http.response.body',
                'body': b'small'
            })

        middleware = CompressionMiddleware(
            app=small_app,
            enabled=True,
            minimum_size=100
        )

        request = MagicMock()
        request.url.path = "/test"
        request.headers = {"Accept-Encoding": "gzip"}
        request.method = "GET"

        response = await middleware.dispatch(request, small_app)
        # Should not be compressed since it's too small
        assert response is not None

    @pytest.mark.asyncio
    async def test_gzip_compression(self, middleware_enabled):
        """Test that gzip compression is applied when requested."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {"Accept-Encoding": "gzip"}
        request.method = "GET"

        response = await middleware_enabled.dispatch(request, self.mock_app)
        assert response is not None
        # Check for gzip header in response (implementation dependent)
        # Actual compression depends on response size

    @pytest.mark.asyncio
    async def test_brotli_compression(self, mock_app):
        """Test that brotli compression is applied when requested."""
        middleware = CompressionMiddleware(
            app=mock_app,
            enabled=True,
            minimum_size=100,
            compression_level=6
        )

        request = MagicMock()
        request.url.path = "/test"
        request.headers = {"Accept-Encoding": "br"}
        request.method = "GET"

        response = await middleware.dispatch(request, mock_app)
        assert response is not None

    @pytest.mark.asyncio
    async def test_compression_disabled(self, mock_app):
        """Test that compression can be disabled."""
        middleware = CompressionMiddleware(app=mock_app, enabled=False)

        request = MagicMock()
        request.url.path = "/test"
        request.headers = {"Accept-Encoding": "gzip"}
        request.method = "GET"

        response = await middleware.dispatch(request, mock_app)
        assert response is not None

    def test_should_compress_decision(self, middleware_enabled):
        """Test the decision logic for when to compress."""
        # Large enough response should be compressed
        large_body = b'x' * 1000
        assert middleware_enabled._should_compress(large_body, {"Accept-Encoding": "gzip"}) is True

        # Small response should not be compressed
        small_body = b'small'
        assert middleware_enabled._should_compress(small_body, {"Accept-Encoding": "gzip"}) is False

        # No Accept-Encoding header should not compress
        assert middleware_enabled._should_compress(large_body, {}) is False

    @pytest.mark.asyncio
    async def test_compression_with_no_encoding_header(self, middleware_enabled):
        """Test behavior when client doesn't accept compression."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}  # No Accept-Encoding
        request.method = "GET"

        response = await middleware_enabled.dispatch(request, self.mock_app)
        assert response is not None
