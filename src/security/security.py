"""Lightweight security helpers used by unit tests and local tooling."""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


class APIKeyValidator:
    """Validate request API keys against an allow-list."""

    def __init__(self, api_keys: Optional[Iterable[str]] = None):
        self.api_keys = {key for key in (api_keys or []) if key}

    def validate(self, api_key: Optional[str]) -> bool:
        if not self.api_keys:
            return True
        if not api_key:
            return False
        return api_key in self.api_keys


class PasswordHasher:
    """PBKDF2-based password hasher with per-password random salt."""

    def __init__(self, iterations: int = 120_000):
        self.iterations = int(iterations)

    def hash(self, password: str) -> str:
        if not isinstance(password, str) or not password:
            raise ValueError("password must be a non-empty string")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.iterations,
        )
        return "pbkdf2_sha256${iter}${salt}${digest}".format(
            iter=self.iterations,
            salt=base64.urlsafe_b64encode(salt).decode("ascii"),
            digest=base64.urlsafe_b64encode(digest).decode("ascii"),
        )

    def verify(self, password: str, hashed: str) -> bool:
        try:
            scheme, iter_raw, salt_raw, digest_raw = hashed.split("$", 3)
            if scheme != "pbkdf2_sha256":
                return False
            iterations = int(iter_raw)
            salt = base64.urlsafe_b64decode(salt_raw.encode("ascii"))
            expected = base64.urlsafe_b64decode(digest_raw.encode("ascii"))
            computed = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                iterations,
            )
            return hmac.compare_digest(computed, expected)
        except Exception:
            return False


class TokenManager:
    """Signed token generator/validator using HMAC-SHA256."""

    def __init__(self, secret_key: str, expiration_seconds: int = 3600):
        if not secret_key:
            raise ValueError("secret_key is required")
        self.secret_key = secret_key.encode("utf-8")
        self.expiration_seconds = int(expiration_seconds)

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

    @staticmethod
    def _b64decode(data: str) -> bytes:
        padding = "=" * ((4 - len(data) % 4) % 4)
        return base64.urlsafe_b64decode((data + padding).encode("ascii"))

    def _sign(self, payload_b64: str) -> str:
        sig = hmac.new(self.secret_key, payload_b64.encode("ascii"), hashlib.sha256).digest()
        return self._b64(sig)

    def _encode_payload(self, payload: Dict[str, object]) -> str:
        payload_raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = self._b64(payload_raw)
        return f"{payload_b64}.{self._sign(payload_b64)}"

    def _decode_payload(self, token: str) -> Optional[Dict[str, object]]:
        try:
            payload_b64, sig = token.split(".", 1)
            expected = self._sign(payload_b64)
            if not hmac.compare_digest(sig, expected):
                return None
            payload_raw = self._b64decode(payload_b64)
            payload = json.loads(payload_raw.decode("utf-8"))
            if not isinstance(payload, dict):
                return None
            return payload
        except Exception:
            return None

    def generate_token(self, user_id: str) -> str:
        now = int(time.time())
        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + self.expiration_seconds,
            "type": "access",
        }
        return self._encode_payload(payload)

    def generate_refresh_token(self, user_id: str) -> str:
        now = int(time.time())
        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + (self.expiration_seconds * 24),
            "type": "refresh",
        }
        return self._encode_payload(payload)

    def validate_token(self, token: Optional[str]) -> Optional[Dict[str, object]]:
        if not token:
            return None
        payload = self._decode_payload(token)
        if not payload:
            return None
        exp = payload.get("exp")
        if not isinstance(exp, int):
            return None
        if int(time.time()) >= exp:
            return None
        return payload


class InputValidator:
    """Basic input validation helpers."""

    _EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    def sanitize(self, value: str) -> str:
        if not isinstance(value, str):
            return ""
        # Remove script tags then escape remaining HTML.
        no_script = re.sub(r"(?is)<\s*script[^>]*>.*?<\s*/\s*script\s*>", "", value)
        return html.escape(no_script, quote=True)

    def validate_email(self, value: str) -> bool:
        return bool(value and self._EMAIL_PATTERN.match(value))

    def validate_url(self, value: str) -> bool:
        if not value:
            return False
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def validate_uuid(self, value: str) -> bool:
        if not value:
            return False
        try:
            uuid.UUID(value)
            return True
        except Exception:
            return False

    def validate_file_size(self, size_bytes: int, max_size_mb: int = 10) -> bool:
        if size_bytes < 0:
            return False
        return size_bytes <= int(max_size_mb) * 1024 * 1024

    def validate_image_dimensions(self, width: int, height: int, max_dimensions: int = 4096) -> bool:
        if width <= 0 or height <= 0:
            return False
        return width <= max_dimensions and height <= max_dimensions


class SecurityHeaders:
    """Security header defaults."""

    @staticmethod
    def get_default_headers(custom_headers: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
        headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
        }
        if custom_headers:
            headers.update(custom_headers)
        return list(headers.items())
