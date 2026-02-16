"""
Audit logging middleware for tracking all requests.
"""
import time
import json
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any
import uuid

logger = logging.getLogger(__name__)


class AuditLogger:
    """Handles audit logging."""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.audit_logger = logging.getLogger("audit")
        
        # Configure audit logger if not already configured
        if not self.audit_logger.handlers:
            handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
            self.audit_logger.setLevel(logging.INFO)
    
    def log_request(
        self, 
        request: Request, 
        response: Response, 
        duration: float,
        extra: Dict[str, Any] = None
    ):
        """Log request details."""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "request_id": str(uuid.uuid4()),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", ""),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "content_length": int(response.headers.get("Content-Length", 0))
        }
        
        # Add API key info (masked)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            log_entry["api_key"] = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        
        # Add extra info
        if extra:
            log_entry.update(extra)
        
        self.audit_logger.info(json.dumps(log_entry, default=str))


class AuditMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for audit logging."""
    
    def __init__(self, app, log_file: str = None):
        super().__init__(app)
        self.audit_logger = AuditLogger(log_file)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful request
            self.audit_logger.log_request(request, response, duration)
            
            return response
        except Exception as exc:
            duration = time.time() - start_time
            
            # Log failed request
            error_response = Response(
                content=str(exc),
                status_code=500
            )
            self.audit_logger.log_request(
                request, 
                error_response, 
                duration,
                extra={"error": str(exc), "error_type": type(exc).__name__}
            )
            raise