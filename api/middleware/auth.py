"""
Backward compatibility wrapper for auth middleware.
This module re-exports from src.middleware.auth to maintain backward compatibility.
"""

from src.middleware.auth import APIKeyMiddleware

__all__ = ['APIKeyMiddleware']