#!/usr/bin/env python3
"""
Tests for security module.
"""

import pytest
import hashlib
import hmac
import secrets
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory
from src.security.security import (
    APIKeyValidator,
    PasswordHasher,
    TokenManager,
    InputValidator,
    SecurityHeaders
)


class TestAPIKeyValidator:
    """Test API key validation."""

    @pytest.mark.parametrize("api_key,expected", [
        ("key1", True),
        ("key2", True)
    ])
    def test_valid_api_key(self, api_key, expected):
        """Test that valid API key is accepted."""
        validator = APIKeyValidator(api_keys=["key1", "key2"])
        assert validator.validate(api_key) is expected

    def test_invalid_api_key(self):
        """Test that invalid API key is rejected."""
        validator = APIKeyValidator(api_keys=["key1", "key2"])
        assert validator.validate("invalid") is False

    def test_no_api_keys_configured(self):
        """Test behavior when no API keys are configured."""
        validator = APIKeyValidator(api_keys=[])
        # Should allow all when no keys configured
        assert validator.validate("any_key") is True

    @pytest.mark.parametrize("api_key,expected", [
        ("", False),
        (None, False)
    ])
    def test_empty_api_key(self, api_key, expected):
        """Test that empty API key is rejected."""
        validator = APIKeyValidator(api_keys=["key1"])
        assert validator.validate(api_key) is expected


class TestPasswordHasher:
    """Test password hashing functionality."""

    def test_hash_password(self):
        """Test that password is hashed correctly."""
        hasher = PasswordHasher()
        password = "secure_password_123"
        hashed = hasher.hash(password)

        # Hash should be different from original password
        assert hashed != password
        # Hash should be a string
        assert isinstance(hashed, str)
        # Hash should contain algorithm identifier
        assert "argon2" in hashed or "bcrypt" in hashed or "pbkdf2" in hashed

    def test_verify_password(self):
        """Test that hashed password can be verified."""
        hasher = PasswordHasher()
        password = "secure_password_123"
        hashed = hasher.hash(password)

        assert hasher.verify(password, hashed) is True
        assert hasher.verify("wrong_password", hashed) is False

    def test_different_passwords_different_hashes(self):
        """Test that different passwords produce different hashes."""
        hasher = PasswordHasher()
        hash1 = hasher.hash("password1")
        hash2 = hasher.hash("password2")

        assert hash1 != hash2

    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        hasher = PasswordHasher()
        password = "same_password"
        hash1 = hasher.hash(password)
        hash2 = hasher.hash(password)

        # Should be different due to random salt
        assert hash1 != hash2


class TestTokenManager:
    """Test token management."""

    def test_generate_token(self):
        """Test token generation."""
        manager = TokenManager(secret_key="test_secret")
        token = manager.generate_token("user_id_123")

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_token(self):
        """Test token validation."""
        manager = TokenManager(secret_key="test_secret")
        token = manager.generate_token("user_id_123")

        payload = manager.validate_token(token)
        assert payload is not None
        assert payload.get("user_id") == "user_id_123"

    def test_invalid_token(self):
        """Test that invalid token is rejected."""
        manager = TokenManager(secret_key="test_secret")

        assert manager.validate_token("invalid_token") is None
        assert manager.validate_token("") is None

    def test_token_expiration(self):
        """Test token expiration."""
        manager = TokenManager(
            secret_key="test_secret",
            expiration_seconds=1  # Very short expiration
        )

        token = manager.generate_token("user_id_123")
        # Token should be valid immediately
        payload = manager.validate_token(token)
        assert payload is not None

        # Wait for expiration
        import time
        time.sleep(1.1)

        # Token should be expired
        payload = manager.validate_token(token)
        assert payload is None

    def test_refresh_token(self):
        """Test refresh token generation."""
        manager = TokenManager(secret_key="test_secret")
        refresh_token = manager.generate_refresh_token("user_id_123")

        assert refresh_token is not None
        assert isinstance(refresh_token, str)

        # Should be able to validate refresh token
        payload = manager.validate_token(refresh_token)
        assert payload is not None


class TestInputValidator:
    """Test input validation."""

    def test_sanitize_string(self):
        """Test string sanitization."""
        validator = InputValidator()
        # Test basic sanitization (implementation dependent)
        dirty = "<script>alert('xss')</script>"
        clean = validator.sanitize(dirty)
        # Should remove or escape dangerous characters
        assert "<script>" not in clean

    def test_validate_email(self):
        """Test email validation."""
        validator = InputValidator()

        assert validator.validate_email("test@example.com") is True
        assert validator.validate_email("invalid-email") is False
        assert validator.validate_email("") is False

    def test_validate_url(self):
        """Test URL validation."""
        validator = InputValidator()

        assert validator.validate_url("https://example.com") is True
        assert validator.validate_url("http://test.com/path") is True
        assert validator.validate_url("not-a-url") is False

    def test_validate_uuid(self):
        """Test UUID validation."""
        validator = InputValidator()

        valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
        assert validator.validate_uuid(valid_uuid) is True

        invalid_uuid = "not-a-uuid"
        assert validator.validate_uuid(invalid_uuid) is False

    def test_validate_file_size(self):
        """Test file size validation."""
        validator = InputValidator()

        # 1MB should be valid if limit is 10MB
        assert validator.validate_file_size(1024 * 1024, max_size_mb=10) is True

        # 11MB should be invalid if limit is 10MB
        assert validator.validate_file_size(11 * 1024 * 1024, max_size_mb=10) is False

    def test_validate_image_dimensions(self):
        """Test image dimension validation."""
        validator = InputValidator()

        # Valid dimensions
        assert validator.validate_image_dimensions(1920, 1080, max_dimensions=4096) is True
        assert validator.validate_image_dimensions(100, 100, max_dimensions=4096) is True

        # Invalid dimensions (too large)
        assert validator.validate_image_dimensions(5000, 5000, max_dimensions=4096) is False


class TestSecurityHeaders:
    """Test security headers middleware."""

    def test_hsts_header(self):
        """Test HSTS header is set."""
        headers = SecurityHeaders.get_default_headers()
        header_dict = dict(headers)

        assert "Strict-Transport-Security" in header_dict or "strict-transport-security" in header_dict

    def test_csp_header(self):
        """Test CSP header is set."""
        headers = SecurityHeaders.get_default_headers()
        header_dict = dict(headers)

        assert "Content-Security-Policy" in header_dict or "content-security-policy" in header_dict

    def test_x_frame_options(self):
        """Test X-Frame-Options header is set."""
        headers = SecurityHeaders.get_default_headers()
        header_dict = dict(headers)

        assert "X-Frame-Options" in header_dict or "x-frame-options" in header_dict

    def test_x_content_type_options(self):
        """Test X-Content-Type-Options header is set."""
        headers = SecurityHeaders.get_default_headers()
        header_dict = dict(headers)

        assert "X-Content-Type-Options" in header_dict or "x-content-type-options" in header_dict

    def test_custom_headers(self):
        """Test custom security headers."""
        custom_headers = {
            "X-Custom-Header": "custom-value"
        }
        headers = SecurityHeaders.get_default_headers(custom_headers)
        header_dict = dict(headers)

        assert "X-Custom-Header" in header_dict or "x-custom-header" in header_dict
