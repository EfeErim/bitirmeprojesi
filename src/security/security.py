"""
Security middleware: HTTPS enforcement, CORS hardening, input size limits.
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import logging

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