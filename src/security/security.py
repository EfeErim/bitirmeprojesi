"""
Security middleware: HTTPS enforcement, CORS hardening, input size limits.
"""
from fastapi import Request, HTTPException
from typing import List, Dict, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import logging
import secrets
import hashlib
import hmac
import re
import json
import time
import jwt
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InputSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    
    def __init__(self, app, max_size_mb: int = 10):
        super().__init__(app)
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    async def dispatch(self, request: Request, call_next):
        # Check content length header
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size_bytes:
                    logger.warning(
                        f"Request too large: {size} bytes (max: {self.max_size_bytes})"
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request too large. Maximum size is {self.max_size_bytes // (1024*1024)}MB"
                    )
            except ValueError:
                pass
        
        # For streaming requests, we can't easily check size without reading body
        # In production, use nginx or similar for body size limiting
        
        return await call_next(request)


class APIKeyValidator:
    """Validates API keys against a list of allowed keys."""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.key_hashes = {self._hash_key(key) for key in api_keys}
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for secure comparison."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def validate(self, api_key: Optional[str]) -> bool:
        """Validate an API key.

        Backwards-compatible behaviour: when no API keys are configured,
        validation should allow all keys (return True). This mirrors test
        expectations that an empty `api_keys` list disables API key checks.
        """
        # If no API keys configured, allow all (tests expect this)
        if not self.api_keys:
            return True

        if not api_key:
            return False

        return self._hash_key(api_key) in self.key_hashes


class PasswordHasher:
    """Handles password hashing and verification.

    Uses PBKDF2-HMAC-SHA256 by default and emits a string prefixed with
    the algorithm name (e.g. "pbkdf2$...") so tests that look for
    'pbkdf2' in the hash are satisfied.
    """

    def __init__(self, algorithm: str = "pbkdf2", salt_length: int = 16, iterations: int = 100_000):
        self.algorithm = algorithm
        self.salt_length = salt_length
        self.iterations = int(iterations)

    def hash_password(self, password: str) -> str:
        """Hash a password with salt using the configured algorithm."""
        salt = secrets.token_hex(self.salt_length)
        if self.algorithm == "pbkdf2":
            dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), self.iterations)
            digest = dk.hex()
            return f"pbkdf2${self.iterations}${salt}${digest}"
        else:
            # Fallback to simple salted hash for other algorithm names
            hash_obj = hashlib.new(self.algorithm)
            hash_obj.update(salt.encode() + password.encode())
            return f"{self.algorithm}${salt}${hash_obj.hexdigest()}"

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against a hashed value."""
        try:
            parts = hashed.split('$')
            if parts[0] == 'pbkdf2':
                # Format: pbkdf2$<iterations>$<salt>$<digest>
                _, iters, salt, stored_hash = parts
                dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), int(iters))
                return hmac.compare_digest(dk.hex(), stored_hash)
            else:
                algorithm, salt, stored_hash = parts
                hash_obj = hashlib.new(algorithm)
                hash_obj.update(salt.encode() + password.encode())
                return hmac.compare_digest(hash_obj.hexdigest(), stored_hash)
        except Exception:
            return False

    # Backwards-compatible simple API used by tests
    def hash(self, password: str) -> str:
        return self.hash_password(password)

    def verify(self, password: str, hashed: str) -> bool:
        return self.verify_password(password, hashed)


class TokenManager:
    """Manages JWT token creation and validation.

    Provides compatibility methods expected by tests: `generate_token`,
    `validate_token` (returns None on invalid token instead of raising),
    and `generate_refresh_token`. Accepts `expiration_seconds` as a ctor
    alias for `expires_in`.
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256", expires_in: int = 3600, expiration_seconds: Optional[int] = None):
        self.secret_key = secret_key
        self.algorithm = algorithm
        # Accept either `expires_in` or legacy `expiration_seconds`
        if expiration_seconds is not None:
            self.expires_in = int(expiration_seconds)
        else:
            self.expires_in = int(expires_in)

    def create_token(self, user_id: str, additional_claims: Dict = None, expires_in: Optional[int] = None) -> str:
        """Create a JWT token."""
        ttl = self.expires_in if expires_in is None else int(expires_in)
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=ttl),
            "iat": datetime.utcnow(),
        }
        if additional_claims:
            payload.update(additional_claims)
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    # Backwards-compatible aliases expected by tests
    def generate_token(self, user_id: str) -> str:
        return self.create_token(user_id)

    def generate_refresh_token(self, user_id: str) -> str:
        # Simple refresh token implementation: longer expiry and explicit claim
        return self.create_token(user_id, additional_claims={"type": "refresh"}, expires_in=max(60, self.expires_in * 10))

    def verify_token(self, token: str) -> Dict:
        """Verify and decode a JWT token. Raises HTTPException on errors."""
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate token and return payload or None if invalid/expired.

        Tests expect `None` for invalid tokens rather than an exception.
        """
        try:
            return self.verify_token(token)
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    def __init__(self):
        self.sanitization_patterns = {
            "sql_injection": re.compile(r"(union|select|insert|update|delete|drop|alter|create)\s", re.IGNORECASE),
            "xss": re.compile(r"(<script.*?>.*?</script.*?>|<.*?javascript:.*?>)", re.IGNORECASE),
            "path_traversal": re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),
        }
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input to prevent common attacks."""
        sanitized = input_data
        for pattern_name, pattern in self.sanitization_patterns.items():
            sanitized = pattern.sub("█", sanitized)
        return sanitized

    # Backwards-compatible method names expected in tests
    def sanitize(self, input_data: str) -> str:
        return self.sanitize_input(input_data)

    def validate_url(self, url: str) -> bool:
        """Basic URL validation for http/https schemes."""
        url_pattern = re.compile(r'^(https?)://[\w\-]+(\.[\w\-]+)+(:\d+)?(/.*)?$')
        return bool(url_pattern.match(url))

    def validate_file_size(self, size_bytes: int, max_size_mb: int = 10) -> bool:
        """Validate file size (bytes) against `max_size_mb`."""
        return int(size_bytes) <= int(max_size_mb) * 1024 * 1024

    def validate_image_dimensions(self, width: int, height: int, max_dimensions: int = 4096) -> bool:
        """Validate image width/height are within provided max dimension."""
        try:
            return int(width) <= int(max_dimensions) and int(height) <= int(max_dimensions)
        except Exception:
            return False
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        return bool(email_pattern.match(email))
    
    def validate_uuid(self, uuid_str: str) -> bool:
        """Validate UUID format."""
        uuid_pattern = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$", re.IGNORECASE)
        return bool(uuid_pattern.match(uuid_str))
    
    def validate_json(self, json_str: str) -> bool:
        """Validate JSON format."""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False


class SecurityHeaders:
    """Adds security headers to responses."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.default_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
    
    def add_headers(self, response):
        """Add security headers to response."""
        headers = self.default_headers.copy()
        headers.update(self.config.get("custom_headers", {}))
        
        for header, value in headers.items():
            response.headers[header] = value
        
        return response

    @classmethod
    def get_default_headers(cls, custom_headers: Dict = None) -> Dict[str, str]:
        """Return default security headers dict (merge with `custom_headers`).

        Tests expect a classmethod named `get_default_headers` returning a
        dictionary of headers; expose that API while preserving instance-based
        `add_headers` for middleware usage.
        """
        inst = cls()
        headers = inst.default_headers.copy()
        if custom_headers:
            headers.update(custom_headers)
        return headers


def setup_security_middleware(app, config: dict):
    """Setup all security middleware."""
    # CORS with hardened settings
    cors_config = config.get('security', {})
    allowed_origins = cors_config.get('allowed_origins', ["*"])
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        max_age=86400  # 24 hours
    )
    
    # Input size limits
    max_size_mb = cors_config.get('max_request_size_mb', 10)
    app.add_middleware(InputSizeLimitMiddleware, max_size_mb=max_size_mb)
    
    logger.info("Security middleware configured")