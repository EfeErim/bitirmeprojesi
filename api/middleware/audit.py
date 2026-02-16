"""
Backward compatibility wrapper for audit middleware.
This module re-exports from src.middleware.audit to maintain backward compatibility.
"""

from src.middleware.audit import AuditMiddleware, AuditLogger

__all__ = ['AuditMiddleware', 'AuditLogger']