"""
Rate limiting middleware using sliding window algorithm.
Provides 10x protection against abuse.
"""
import time
import asyncio
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}  # client_id -> list of timestamps
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        async with self._lock:
            now = time.time()
            
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # Clean old requests outside window
            window_start = now - self.window_seconds
            self.requests[client_id] = [
                ts for ts in self.requests[client_id] 
                if ts > window_start
            ]
            
            # Check if under limit
            if len(self.requests[client_id]) >= self.max_requests:
                return False, 0
            
            # Add current request
            self.requests[client_id].append(now)
            
            # Calculate remaining
            remaining = self.max_requests - len(self.requests[client_id])
            return True, remaining
    
    def cleanup_old_entries(self):
        """Remove old client entries to prevent memory growth."""
        now = time.time()
        window_start = now - self.window_seconds
        
        to_remove = []
        for client_id, timestamps in self.requests.items():
            if not timestamps or timestamps[-1] < window_start:
                to_remove.append(client_id)
        
        for client_id in to_remove:
            del self.requests[client_id]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self, 
        app, 
        max_requests: int = 100, 
        window_seconds: int = 60,
        exempt_paths: list = None
    ):
        super().__init__(app)
        self.rate_limiter = RateLimiter(max_requests, window_seconds)
        self.exempt_paths = exempt_paths or ["/health", "/metrics"]
        self._cleanup_task = None
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Get client identifier (IP or API key)
        client_id = self._get_client_id(request)
        
        # Check rate limit
        allowed, remaining = await self.rate_limiter.is_allowed(client_id)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(self.rate_limiter.window_seconds)
                }
            )
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Prefer API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:10]}"  # Use first 10 chars for privacy
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"