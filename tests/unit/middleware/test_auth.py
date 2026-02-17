#!/usr/bin/env python3
"""
Tests for API Key authentication middleware.
"""

import pytest
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.middleware.auth import APIKeyMiddleware


class TestAPIKeyMiddleware:
    """Test API key authentication middleware."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""
        async def app(scope, receive, send):
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [[b'content-type', b'text/plain']]
            })
            await send({
                'type': 'http.response.body',
                'body': b'OK'
            })
        return app

    @pytest.fixture
    def middleware(self, mock_app):
        """Create middleware instance with test API keys."""
        return APIKeyMiddleware(
            app=mock_app,
            api_keys=["test_key_1", "test_key_2"],
            exempt_paths=["/health", "/docs"]
        )

    @pytest.mark.asyncio
    async def test_exempt_path_skips_auth(self, middleware):
        """Test that exempt paths skip authentication."""
        # Create mock request for health endpoint
        request = MagicMock()
        request.url.path = "/health"
        request.headers = {}

        # Create mock call_next
        async def call_next(request):
            return MagicMock(status_code=200)

        # Should not raise exception
        response = await middleware.dispatch(request, call_next)
        assert response is not None

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_exception(self, middleware):
        """Test that missing API key raises HTTPException."""
        request = MagicMock()
        request.url.path = "/diagnose"
        request.headers = {}

        async def call_next(request):
            return MagicMock(status_code=200)

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)

        assert exc_info.value.status_code == 401
        assert "API key required" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_exception(self, middleware):
        """Test that invalid API key raises HTTPException."""
        request = MagicMock()
        request.url.path = "/diagnose"
        request.headers = {"X-API-Key": "invalid_key"}

        async def call_next(request):
            return MagicMock(status_code=200)

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_valid_api_key_passes(self, middleware):
        """Test that valid API key allows request to proceed."""
        request = MagicMock()
        request.url.path = "/diagnose"
        request.headers = {"X-API-Key": "test_key_1"}

        async def call_next(request):
            return MagicMock(status_code=200)

        # Should not raise exception
        response = await middleware.dispatch(request, call_next)
        assert response is not None

    @pytest.mark.asyncio
    async def test_no_api_keys_configured_allows_all(self, mock_app):
        """Test that if no API keys configured, all requests pass."""
        middleware = APIKeyMiddleware(app=mock_app, api_keys=[])

        request = MagicMock()
        request.url.path = "/diagnose"
        request.headers = {}

        async def call_next(request):
            return MagicMock(status_code=200)

        # Should not raise exception
        response = await middleware.dispatch(request, call_next)
        assert response is not None

    def test_default_exempt_paths(self, mock_app):
        """Test that default exempt paths are set."""
        middleware = APIKeyMiddleware(app=mock_app, api_keys=["key"])
        expected_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
        assert middleware.exempt_paths == expected_paths

    def test_custom_exempt_paths(self, mock_app):
        """Test custom exempt paths."""
        custom_paths = ["/custom", "/test"]
        middleware = APIKeyMiddleware(
            app=mock_app,
            api_keys=["key"],
            exempt_paths=custom_paths
        )
        assert middleware.exempt_paths == custom_paths
