"""
API Key authentication middleware.
"""
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""
    
    def __init__(
        self, 
        app, 
        api_keys: list = None,
        exempt_paths: list = None
    ):
        super().__init__(app)
        self.api_keys = set(api_keys or [])
        self.exempt_paths = exempt_paths or [
            "/health", 
            "/docs", 
            "/redoc", 
            "/openapi.json"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Skip if no API keys configured
        if not self.api_keys:
            return await call_next(request)
        
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            logger.warning(f"Missing API key for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        # Validate API key
        if api_key not in self.api_keys:
            logger.warning(f"Invalid API key for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        return await call_next(request)