#!/usr/bin/env python3
"""
Error Handling Module for Google Colab Data Pipeline
Custom exceptions and error handling utilities.
"""

import logging
import traceback
import time
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context for error handling."""
    
    operation: str
    component: str
    severity: str
    error_code: str
    message: str
    details: Dict[str, Any] = None
    traceback: str = ""
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ColabDataPipelineError(Exception):
    """Base exception for Colab data pipeline."""
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.context = context or ErrorContext(
            operation="unknown",
            component="unknown",
            severity="error",
            error_code="UNKNOWN_ERROR",
            message=message
        )
        self.cause = cause
        
        if cause:
            self.context.traceback = self._format_traceback(cause)
    
    def _format_traceback(self, exc: Exception) -> str:
        """Format exception traceback."""
        return traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error": {
                "message": self.context.message,
                "operation": self.context.operation,
                "component": self.context.component,
                "severity": self.context.severity,
                "error_code": self.context.error_code,
                "details": self.context.details,
                "traceback": self.context.traceback
            }
        }


class DownloadError(ColabDataPipelineError):
    """Exception for download operations."""
    
    def __init__(
        self,
        message: str,
        file_id: Optional[str] = None,
        destination: Optional[str] = None,
        attempt: Optional[int] = None,
        max_retries: Optional[int] = None,
        cause: Optional[Exception] = None
    ):
        context = ErrorContext(
            operation="download",
            component="downloader",
            severity="error",
            error_code="DOWNLOAD_FAILED",
            message=message,
            details={
                "file_id": file_id,
                "destination": destination,
                "attempt": attempt,
                "max_retries": max_retries
            }
        )
        super().__init__(message, context, cause)


class CacheError(ColabDataPipelineError):
    """Exception for cache operations."""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        operation: str = "cache",
        cause: Optional[Exception] = None
    ):
        context = ErrorContext(
            operation=operation,
            component="cache",
            severity="error",
            error_code="CACHE_ERROR",
            message=message,
            details={
                "cache_key": cache_key,
                "cache_dir": cache_dir
            }
        )
        super().__init__(message, context, cause)


class DataLoaderError(ColabDataPipelineError):
    """Exception for DataLoader operations."""
    
    def __init__(
        self,
        message: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        operation: str = "data_loading",
        cause: Optional[Exception] = None
    ):
        context = ErrorContext(
            operation=operation,
            component="data_loader",
            severity="error",
            error_code="DATA_LOADER_ERROR",
            message=message,
            details={
                "batch_size": batch_size,
                "num_workers": num_workers
            }
        )
        super().__init__(message, context, cause)


class DatasetError(ColabDataPipelineError):
    """Exception for dataset operations."""
    
    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        operation: str = "dataset_operation",
        cause: Optional[Exception] = None
    ):
        context = ErrorContext(
            operation=operation,
            component="dataset",
            severity="error",
            error_code="DATASET_ERROR",
            message=message,
            details={
                "dataset_name": dataset_name
            }
        )
        super().__init__(message, context, cause)


class ColabDataPipelineErrorHandler:
    """Centralized error handler for Colab data pipeline."""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True
    ) -> None:
        """Handle an exception with context."""
        if isinstance(exception, ColabDataPipelineError):
            self._handle_pipeline_error(exception, reraise)
        else:
            self._handle_generic_error(exception, context, reraise)
    
    def _handle_pipeline_error(self, error: ColabDataPipelineError, reraise: bool) -> None:
        """Handle a pipeline-specific error."""
        logger.error(f"[{error.context.error_code}] {error.context.message}")
        
        if error.context.details:
            logger.debug(f"Error details: {error.context.details}")
        
        if error.context.traceback:
            logger.debug(f"Traceback: {error.context.traceback}")
        
        if reraise:
            raise error
    
    def _handle_generic_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext],
        reraise: bool
    ) -> None:
        """Handle a generic error."""
        error_message = f"Unexpected error: {str(exception)}"
        
        if context:
            error_message = f"[{context.error_code}] {context.message}"
        
        logger.error(error_message)
        logger.debug(traceback.format_exc())
        
        if reraise:
            raise ColabDataPipelineError(error_message, context, exception)
    
    def log_warning(
        self,
        message: str,
        context: Optional[ErrorContext] = None
    ) -> None:
        """Log a warning with context."""
        if context:
            logger.warning(f"[{context.error_code}] {message}")
            if context.details:
                logger.debug(f"Warning details: {context.details}")
        else:
            logger.warning(message)
    
    def log_info(
        self,
        message: str,
        context: Optional[ErrorContext] = None
    ) -> None:
        """Log information with context."""
        if context:
            logger.info(f"[{context.error_code}] {message}")
        else:
            logger.info(message)


class RetryHandler:
    """Retry mechanism with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                delay = min(
                    self.base_delay * (self.backoff_factor ** (attempt - 1)),
                    self.max_delay
                )
                
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed: {str(e)}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                
                time.sleep(delay)
        
        raise last_exception


class ResourceMonitor:
    """Monitor system resources for error prevention."""
    
    def __init__(self):
        self._last_memory_check = 0
        self._memory_threshold_gb = 0.5  # Warn when less than 500MB available
    
    def check_memory(self) -> bool:
        """Check available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < self._memory_threshold_gb:
                logger.warning(
                    f"Low memory warning: {available_gb:.1f}GB available. "
                    f"Consider reducing batch size or clearing cache."
                )
                return False
            
            return True
        except ImportError:
            logger.debug("psutil not available, skipping memory check")
            return True
    
    def check_disk_space(self, required_gb: float = 1.0) -> bool:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb < required_gb:
                logger.warning(
                    f"Low disk space warning: {free_gb:.1f}GB available. "
                    f"Need at least {required_gb}GB for operation."
                )
                return False
            
            return True
        except Exception:
            logger.debug("Could not check disk space")
            return True


def get_error_handler() -> ColabDataPipelineErrorHandler:
    """Get a configured error handler."""
    return ColabDataPipelineErrorHandler(log_level="INFO")


def get_retry_handler() -> RetryHandler:
    """Get a retry handler configured for Colab."""
    return RetryHandler(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0
    )


def get_resource_monitor() -> ResourceMonitor:
    """Get a resource monitor."""
    return ResourceMonitor()


if __name__ == "__main__":
    # Example usage
    handler = get_error_handler()
    
    try:
        # Simulate an error
        raise ValueError("Test error")
    except Exception as e:
        handler.handle_exception(e, context=ErrorContext(
            operation="test",
            component="example",
            severity="error",
            error_code="TEST_ERROR",
            message="This is a test error"
        ))