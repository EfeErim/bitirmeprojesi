"""
Monitoring endpoints for metrics and health checks.
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import psutil
import torch
import logging
import time
from typing import Dict, Any

from api.metrics import metrics_collector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["monitoring"])


@router.get("/metrics")
async def get_metrics():
    """
    Get API metrics in Prometheus format.
    
    Returns metrics including:
    - Request counts and rates
    - Response times (avg, p95, min, max)
    - Error rates
    """
    metrics = metrics_collector.get_metrics()
    
    # Convert to Prometheus text format
    lines = []
    
    # Help text
    lines.append("# HELP aads_ulora_requests_total Total number of requests")
    lines.append("# TYPE aads_ulora_requests_total counter")
    
    lines.append("# HELP aads_ulora_errors_total Total number of errors")
    lines.append("# TYPE aads_ulora_errors_total counter")
    
    lines.append("# HELP aads_ulora_request_duration_seconds Request duration in seconds")
    lines.append("# TYPE aads_ulora_request_duration_seconds gauge")
    
    lines.append("# HELP aads_ulora_error_rate Error rate")
    lines.append("# TYPE aads_ulora_error_rate gauge")
    
    # Metrics
    lines.append(f"aads_ulora_requests_total {metrics['total_requests']}")
    lines.append(f"aads_ulora_errors_total {metrics['total_errors']}")
    lines.append(f"aads_ulora_error_rate {metrics['error_rate']}")
    
    for endpoint, data in metrics['endpoints'].items():
        endpoint_name = endpoint.replace('/', '_').replace('{', '').replace('}', '')
        lines.append(f'aads_ulora_request_duration_seconds{{endpoint="{endpoint}"}} {data["avg_time"]}')
    
    return JSONResponse(
        content={"metrics": metrics},
        media_type="text/plain"
    )


@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """
    Detailed health check with system metrics.
    
    Returns:
    - API status
    - System resources (CPU, memory, disk)
    - GPU status (if available)
    - Pipeline status
    """
    health = {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "uptime": time.time() - psutil.boot_time()
    }
    
    # System metrics
    try:
        health["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "percent": psutil.disk_usage('/').percent
            }
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        health["system"] = {"error": str(e)}
    
    # GPU metrics
    try:
        if torch.cuda.is_available():
            health["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            }
        else:
            health["gpu"] = {"available": False}
    except Exception as e:
        logger.error(f"Error getting GPU metrics: {e}")
        health["gpu"] = {"available": False, "error": str(e)}
    
    # Pipeline status
    try:
        pipeline = request.app.state.pipeline
        if pipeline:
            health["pipeline"] = {
                "initialized": True,
                "device": str(pipeline.device),
                "router_loaded": pipeline.router is not None,
                "adapters_loaded": list(pipeline.adapters.keys()),
                "num_adapters": len(pipeline.adapters)
            }
        else:
            health["pipeline"] = {"initialized": False}
            health["status"] = "unhealthy"
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        health["pipeline"] = {"error": str(e)}
        health["status"] = "unhealthy"
    
    return health


@router.get("/readiness")
async def readiness_check(request: Request):
    """Kubernetes readiness probe."""
    try:
        pipeline = request.app.state.pipeline
        if pipeline and pipeline.router:
            return {"status": "ready"}
        else:
            return JSONResponse(
                content={"status": "not ready"},
                status_code=503
            )
    except Exception:
        return JSONResponse(
            content={"status": "not ready"},
            status_code=503
        )


@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}
