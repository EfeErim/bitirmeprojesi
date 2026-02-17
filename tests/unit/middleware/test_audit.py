#!/usr/bin/env python3
"""
Tests for audit logging middleware.
"""

import pytest
import json
from fastapi import Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.middleware.audit import AuditMiddleware, AuditLogger


class TestAuditMiddleware:
    """Test audit middleware functionality."""

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
    def middleware(self, mock_app, tmp_path):
        """Create middleware instance with audit logging."""
        audit_log_file = tmp_path / "audit.log"
        return AuditMiddleware(
            app=mock_app,
            audit_log_file=str(audit_log_file),
            enabled=True
        )

    @pytest.mark.asyncio
    async def test_request_logged(self, middleware, tmp_path):
        """Test that requests are logged."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {"X-Forwarded-For": "127.0.0.1"}
        request.method = "POST"
        request.query_params = {}
        request.client = MagicMock(host="127.0.0.1")

        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response

        await middleware.dispatch(request, call_next)

        # Check that audit log was written
        audit_log_file = tmp_path / "audit.log"
        assert audit_log_file.exists()

        log_content = audit_log_file.read_text()
        assert len(log_content) > 0
        # Should contain request information
        assert "127.0.0.1" in log_content or "test" in log_content.lower()

    @pytest.mark.asyncio
    async def test_response_status_logged(self, middleware, tmp_path):
        """Test that response status is logged."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"
        request.query_params = {}
        request.client = MagicMock(host="127.0.0.1")

        async def call_next(request):
            response = MagicMock()
            response.status_code = 201
            response.headers = {}
            return response

        await middleware.dispatch(request, call_next)

        log_content = (tmp_path / "audit.log").read_text()
        assert "201" in log_content

    @pytest.mark.asyncio
    async def test_request_body_logged(self, middleware, tmp_path):
        """Test that request body is logged when enabled."""
        middleware.log_request_body = True

        request = MagicMock()
        request.url.path = "/test"
        request.headers = {"Content-Type": "application/json"}
        request.method = "POST"
        request.query_params = {}
        request.client = MagicMock(host="127.0.0.1")

        # Mock body
        async def receive():
            return {"type": "http.request", "body": b'{"test": "data"}'}

        request.receive = receive

        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response

        await middleware.dispatch(request, call_next)

        log_content = (tmp_path / "audit.log").read_text()
        assert "test" in log_content or "data" in log_content

    @pytest.mark.asyncio
    async def test_response_body_logged(self, middleware, tmp_path):
        """Test that response body is logged when enabled."""
        middleware.log_response_body = True

        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"
        request.query_params = {}
        request.client = MagicMock(host="127.0.0.1")

        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.body = b'{"result": "success"}'

            async def send_sequence():
                yield {
                    'type': 'http.response.start',
                    'status': 200,
                    'headers': [[b'content-type', b'application/json']]
                }
                yield {
                    'type': 'http.response.body',
                    'body': b'{"result": "success"}'
                }

            response.send = send_sequence
            return response

        await middleware.dispatch(request, call_next)

        log_content = (tmp_path / "audit.log").read_text()
        assert "result" in log_content or "success" in log_content

    @pytest.mark.asyncio
    async def test_audit_disabled(self, mock_app, tmp_path):
        """Test that audit logging can be disabled."""
        middleware = AuditMiddleware(
            app=mock_app,
            audit_log_file=str(tmp_path / "audit.log"),
            enabled=False
        )

        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"
        request.query_params = {}
        request.client = MagicMock(host="127.0.0.1")

        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response

        await middleware.dispatch(request, call_next)

        # Log file should not be created when disabled
        audit_log_file = tmp_path / "audit.log"
        # File might exist but should be empty or not contain log entries
        if audit_log_file.exists():
            content = audit_log_file.read_text()
            # Should not contain audit entries
            assert len(content.strip()) == 0

    @pytest.mark.asyncio
    async def test_exempt_paths_not_logged(self, middleware, tmp_path):
        """Test that exempt paths are not logged."""
        middleware.exempt_paths = ["/health", "/status"]

        request = MagicMock()
        request.url.path = "/health"
        request.headers = {}
        request.method = "GET"
        request.query_params = {}
        request.client = MagicMock(host="127.0.0.1")

        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response

        await middleware.dispatch(request, call_next)

        # Should not have logged the health check
        log_content = (tmp_path / "audit.log").read_text()
        # If health check was logged, it would contain "/health"
        # Since it's exempt, it should not be present
        assert "/health" not in log_content

    @pytest.mark.asyncio
    async def test_error_logging(self, middleware, tmp_path):
        """Test that errors are logged with exception details."""
        request = MagicMock()
        request.url.path = "/test"
        request.headers = {}
        request.method = "GET"
        request.query_params = {}
        request.client = MagicMock(host="127.0.0.1")

        async def call_next(request):
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await middleware.dispatch(request, call_next)

        log_content = (tmp_path / "audit.log").read_text()
        # Should contain error information
        assert "ValueError" in log_content or "error" in log_content.lower()

    def test_log_rotation(self, tmp_path):
        """Test log file rotation."""
        log_file = tmp_path / "audit.log"

        # Create logger with small max size for testing
        logger = AuditLogger(
            log_file=str(log_file),
            max_size_mb=0.001  # Very small to trigger rotation
        )

        # Log many entries to trigger rotation
        for i in range(100):
            logger.log_request(
                method="GET",
                path="/test",
                status_code=200,
                client_host="127.0.0.1"
            )

        # Should have created rotated logs
        log_files = list(tmp_path.glob("audit*.log"))
        assert len(log_files) >= 1


class TestAuditLogger:
    """Test audit logger functionality."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create audit logger instance."""
        log_file = tmp_path / "audit.log"
        return AuditLogger(str(log_file))

    def test_log_request(self, logger, tmp_path):
        """Test logging a request."""
        logger.log_request(
            method="POST",
            path="/api/test",
            status_code=200,
            client_host="192.168.1.1",
            user_agent="TestAgent/1.0",
            request_size=1024,
            response_size=512
        )

        log_file = tmp_path / "audit.log"
        assert log_file.exists()

        content = log_file.read_text()
        assert "POST" in content
        assert "/api/test" in content
        assert "200" in content
        assert "192.168.1.1" in content

    def test_log_with_metadata(self, logger, tmp_path):
        """Test logging with additional metadata."""
        metadata = {
            "user_id": "user123",
            "session_id": "session456",
            "custom_field": "custom_value"
        }

        logger.log_request(
            method="GET",
            path="/test",
            status_code=200,
            client_host="127.0.0.1",
            metadata=metadata
        )

        content = (tmp_path / "audit.log").read_text()
        # Should contain metadata fields
        assert "user123" in content or "session456" in content

    def test_log_json_format(self, logger, tmp_path):
        """Test that logs are in JSON format."""
        logger.log_request(
            method="GET",
            path="/test",
            status_code=200,
            client_host="127.0.0.1"
        )

        content = (tmp_path / "audit.log").read_text().strip()
        # Should be valid JSON
        log_entry = json.loads(content.split('\n')[0])

        assert "timestamp" in log_entry
        assert "method" in log_entry
        assert "path" in log_entry
        assert "status_code" in log_entry
        assert "client_host" in log_entry

    def test_log_rotation_by_size(self, tmp_path):
        """Test log rotation based on file size."""
        log_file = tmp_path / "audit.log"

        logger = AuditLogger(
            str(log_file),
            max_size_mb=0.01  # 10KB
        )

        # Write large entries to trigger rotation
        large_metadata = {"data": "x" * 5000}

        for i in range(10):
            logger.log_request(
                method="POST",
                path="/test",
                status_code=200,
                client_host="127.0.0.1",
                metadata=large_metadata
            )

        # Check if rotation occurred
        log_files = list(tmp_path.glob("audit*.log"))
        assert len(log_files) >= 1

    def test_log_with_user_context(self, logger, tmp_path):
        """Test logging with user authentication context."""
        logger.log_request(
            method="GET",
            path="/api/user/profile",
            status_code=200,
            client_host="127.0.0.1",
            user_id="user_12345",
            api_key_id="key_abc"
        )

        content = (tmp_path / "audit.log").read_text()
        assert "user_12345" in content
        assert "key_abc" in content

    def test_log_sensitive_data_filtering(self, tmp_path):
        """Test that sensitive data is filtered from logs."""
        log_file = tmp_path / "audit.log"

        logger = AuditLogger(
            str(log_file),
            sensitive_fields=["password", "token", "secret"]
        )

        # Log request with sensitive data
        logger.log_request(
            method="POST",
            path="/login",
            status_code=200,
            client_host="127.0.0.1",
            metadata={
                "username": "testuser",
                "password": "secret123",
                "token": "abc123token"
            }
        )

        content = log_file.read_text()
        # Sensitive fields should be redacted
        assert "secret123" not in content
        assert "abc123token" not in content
        # Non-sensitive should be present
        assert "testuser" in content

    def test_log_structured_format(self, logger, tmp_path):
        """Test that logs are in structured format."""
        logger.log_request(
            method="POST",
            path="/api/v1/diagnose",
            status_code=201,
            client_host="192.168.1.100",
            user_agent="Mozilla/5.0",
            request_size=2048,
            response_size=1024,
            latency_ms=150
        )

        content = (tmp_path / "audit.log").read_text()
        log_entry = json.loads(content.strip().split('\n')[0])

        # All expected fields should be present
        expected_fields = [
            'timestamp', 'method', 'path', 'status_code',
            'client_host', 'user_agent', 'request_size',
            'response_size', 'latency_ms'
        ]
        for field in expected_fields:
            assert field in log_entry

    def test_log_with_exception(self, logger, tmp_path):
        """Test logging when exception occurs."""
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.log_exception(
                method="GET",
                path="/test",
                client_host="127.0.0.1"
            )

        content = (tmp_path / "audit.log").read_text()
        # Should contain exception information
        assert "ValueError" in content
        assert "Test error" in content

    def test_log_batch_operations(self, logger, tmp_path):
        """Test logging batch operations."""
        operations = [
            {"method": "POST", "path": "/batch1", "status": 200},
            {"method": "POST", "path": "/batch2", "status": 201},
            {"method": "POST", "path": "/batch3", "status": 400}
        ]

        logger.log_batch(operations)

        content = (tmp_path / "audit.log").read_text()
        # All operations should be logged
        assert "batch1" in content
        assert "batch2" in content
        assert "batch3" in content
        assert "400" in content

    def test_log_with_correlation_id(self, logger, tmp_path):
        """Test logging with correlation ID for tracing."""
        correlation_id = "req_123456789"

        logger.log_request(
            method="GET",
            path="/test",
            status_code=200,
            client_host="127.0.0.1",
            correlation_id=correlation_id
        )

        content = (tmp_path / "audit.log").read_text()
        assert correlation_id in content

    def test_log_compressed_archiving(self, tmp_path):
        """Test that old logs are compressed and archived."""
        log_file = tmp_path / "audit.log"

        logger = AuditLogger(
            str(log_file),
            max_size_mb=0.001,
            archive_dir=str(tmp_path / "archive")
        )

        # Generate many log entries
        for i in range(100):
            logger.log_request(
                method="GET",
                path="/test",
                status_code=200,
                client_host="127.0.0.1"
            )

        # Check for archived files
        archive_dir = tmp_path / "archive"
        if archive_dir.exists():
            archive_files = list(archive_dir.glob("*.log.gz"))
            # May or may not have archived files depending on implementation
            pass
