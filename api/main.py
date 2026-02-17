#!/usr/bin/env python3
"""
AADS-ULoRA v5.5.3 FastAPI Server
Main API entry point for crop disease diagnosis with production optimizations.
"""

import logging
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration management
from src.core.config_manager import config_manager, get_config, reload_configuration
from src.core.configuration_validator import ConfigurationError

# Load configuration at module level
logger.info("Loading configuration...")
try:
    # Load all configurations
    config_manager.load_base_config()
    config_manager.load_all_configs()
    
    # Get environment
    ENV = os.getenv('APP_ENV', 'development')
    
    # Load environment-specific config
    env_config = config_manager.get_environment_config(ENV)
    if env_config:
        logger.info(f"Loaded {ENV} environment configuration")
    
    # Get complete merged configuration
    CONFIG = get_config()
    
    # Apply environment-specific overrides
    if env_config:
        for key, value in env_config.items():
            if key in CONFIG and isinstance(value, dict):
                CONFIG[key].update(value)
            else:
                CONFIG[key] = value
    
    logger.info(f"Configuration loaded successfully (version: {CONFIG.get('version')})")
    
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    raise
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="AADS-ULoRA v5.5.3 API",
    description="Agricultural Disease Detection with Dynamic OOD - Production Optimized",
    version="5.5.3-performance",
    docs_url="/docs" if CONFIG.get('api', {}).get('reload', False) else None,
    redoc_url="/redoc" if CONFIG.get('api', {}).get('reload', False) else None,
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Health & System",
            "description": "Health checks and system information endpoints",
            "externalDocs": {
                "description": "Health endpoints",
                "url": "https://fastapi.tiangolo.com/tutorial/path-operation-configuration/"
            }
        },
        {
            "name": "Crops",
            "description": "Crop management and information endpoints",
            "externalDocs": {
                "description": "Crop endpoints",
                "url": "https://fastapi.tiangolo.com/tutorial/path-operation-configuration/"
            }
        },
        {
            "name": "Diagnosis",
            "description": "Disease diagnosis and analysis endpoints",
            "externalDocs": {
                "description": "Diagnosis endpoints",
                "url": "https://fastapi.tiangolo.com/tutorial/path-operation-configuration/"
            }
        },
        {
            "name": "Feedback",
            "description": "Expert feedback and label submission endpoints",
            "externalDocs": {
                "description": "Feedback endpoints",
                "url": "https://fastapi.tiangolo.com/tutorial/path-operation-configuration/"
            }
        },
        {
            "name": "Monitoring",
            "description": "System monitoring and metrics endpoints",
            "externalDocs": {
                "description": "Monitoring endpoints",
                "url": "https://fastapi.tiangolo.com/tutorial/path-operation-configuration/"
            }
        }
    ]
)

# Apply CORS middleware
cors_config = CONFIG.get('security', {}).get('cors', {})
allowed_origins = cors_config.get('allow_origins', ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=cors_config.get('allow_credentials', True),
    allow_methods=cors_config.get('allow_methods', ["*"]),
    allow_headers=cors_config.get('allow_headers', ["*"]),
    expose_headers=cors_config.get('expose_headers', ["X-Process-Time"]),
    max_age=cors_config.get('max_age', 600)
)

# Apply input size limit middleware
from src.security.security import InputSizeLimitMiddleware
max_size_mb = CONFIG.get('security', {}).get('input_validation', {}).get('max_request_size_mb', 10)
app.add_middleware(InputSizeLimitMiddleware, max_size_mb=max_size_mb)

# Initialize pipeline variable
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    global pipeline
    
    logger.info("Starting AADS-ULoRA v5.5.3...")
    
    # Initialize pipeline
    try:
        from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
        pipeline = IndependentMultiCropPipeline(CONFIG, device=CONFIG.get('ml', {}).get('device', 'cuda'))
        
        # Initialize router
        if CONFIG.get('router', {}).get('enabled', True):
            router_path = CONFIG.get('router', {}).get('model_path', None)
            success = pipeline.initialize_router(router_path)
            if success:
                logger.info("Router initialized successfully")
            else:
                logger.warning("Router initialization failed")
        
        logger.info("Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down AADS-ULoRA v5.5.3...")
    # Cleanup resources if needed

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_config = CONFIG.get('health', {})
    
    health_status = {
        "status": "healthy",
        "version": CONFIG.get('version', 'unknown'),
        "environment": os.getenv('APP_ENV', 'development'),
        "uptime": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    if health_config.get('detailed', False):
        # Add detailed health information
        health_status.update({
            "memory": get_memory_info(),
            "cuda": get_cuda_info() if torch.cuda.is_available() else None,
            "disk": get_disk_info()
        })
    
    return health_status

# Configuration reload endpoint (for development)
@app.post("/admin/reload-config")
async def reload_config():
    """Reload configuration (development only)."""
    global CONFIG
    if not CONFIG.get('api', {}).get('reload', False):
        raise HTTPException(status_code=403, detail="Configuration reload disabled in production")
    
    try:
        reload_configuration()
        CONFIG = get_config()
        return {"status": "success", "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload configuration: {e}")

# Error handling
@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    """Handle configuration errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Configuration error: {str(exc)}"}
    )

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get API configuration
    api_config = CONFIG.get('api', {})
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', False),
        workers=api_config.get('workers', 1),
        log_level=api_config.get('log_level', 'info').upper()
    )