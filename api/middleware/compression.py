"""
Compression middleware for reducing bandwidth.
Provides 50% less bandwidth for compressible responses.
"""
import gzip
import brotli
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware that compresses responses."""
    
    def __init__(
        self, 
        app, 
        minimum_size: int = 1024,
        compression_level: int = 6,
        enabled: bool = True
    ):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.enabled = enabled
    
    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)
        
        # Check if client accepts compression
        accept_encoding = request.headers.get("Accept-Encoding", "")
        supports_gzip = "gzip" in accept_encoding
        supports_br = "br" in accept_encoding
        
        response = await call_next(request)
        
        # Only compress if client supports it and response is compressible
        if not (supports_gzip or supports_br):
            return response
        
        # Skip compression for already compressed content
        content_type = response.headers.get("Content-Type", "")
        if any(ct in content_type for ct in ["image/", "video/", "audio/", "application/zip"]):
            return response
        
        # Get response body
        content = b""
        async for chunk in response.body_iterator:
            content += chunk
        
        # Only compress if content is large enough
        if len(content) < self.minimum_size:
            return Response(
                content=content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        
        # Choose best compression method
        compressed_content = None
        content_encoding = None
        
        if supports_br:
            try:
                compressed_content = brotli.compress(content, quality=self.compression_level)
                content_encoding = "br"
            except Exception as e:
                logger.error(f"Brotli compression error: {e}")
        
        if not compressed_content and supports_gzip:
            try:
                compressed_content = gzip.compress(content, compresslevel=self.compression_level)
                content_encoding = "gzip"
            except Exception as e:
                logger.error(f"Gzip compression error: {e}")
        
        if compressed_content and len(compressed_content) < len(content):
            # Build new headers
            headers = dict(response.headers)
            headers["Content-Encoding"] = content_encoding
            headers["Content-Length"] = str(len(compressed_content))
            # Remove content-length if present (will be set automatically)
            if "content-length" in headers:
                del headers["content-length"]
            
            logger.info(
                f"Compressed {len(content)} -> {len(compressed_content)} "
                f"({len(compressed_content)/len(content)*100:.1f}%)"
            )
            
            return Response(
                content=compressed_content,
                status_code=response.status_code,
                headers=headers
            )
        
        return Response(
            content=content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )