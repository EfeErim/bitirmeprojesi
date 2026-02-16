"""
Backward compatibility wrapper for caching middleware.
This module re-exports from src.middleware.caching to maintain backward compatibility.
"""

from src.middleware.caching import CacheMiddleware, RedisCache

__all__ = ['CacheMiddleware', 'RedisCache']