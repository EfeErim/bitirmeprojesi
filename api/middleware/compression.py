"""
Backward compatibility wrapper for compression middleware.
This module re-exports from src.middleware.compression to maintain backward compatibility.
"""

from src.middleware.compression import CompressionMiddleware

__all__ = ['CompressionMiddleware']