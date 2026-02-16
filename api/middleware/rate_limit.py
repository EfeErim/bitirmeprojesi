"""
Backward compatibility wrapper for rate limiting middleware.
This module re-exports from src.middleware.rate_limit to maintain backward compatibility.
"""

from src.middleware.rate_limit import RateLimitMiddleware, RateLimiter

__all__ = ['RateLimitMiddleware', 'RateLimiter']