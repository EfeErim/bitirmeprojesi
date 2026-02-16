"""
Backward compatibility wrapper for security middleware.
This module re-exports from src.security.security to maintain backward compatibility.
"""

from src.security.security import InputSizeLimitMiddleware, setup_security_middleware

__all__ = ['InputSizeLimitMiddleware', 'setup_security_middleware']