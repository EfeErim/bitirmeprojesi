#!/usr/bin/env python3
"""
AADS-ULoRA v5.5.3 FastAPI Server
Main API entry point for crop disease diagnosis with production optimizations.
"""

import logging
import sys
import os
from pathlib import Path
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from src.utils.data_loader import preprocess_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'adapter_spec_v55.json'
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
    
    # Load environment-specific config
    import os
    ENV = os.getenv('APP_ENV', 'development')
    if ENV == 'production':
        prod_config_path = Path(__file__).parent.parent / 'config' / 'production.json'
        with open(prod_config_path, 'r') as f:
            PROD_CONFIG = json.load(f)
        # Merge configs (prod overrides base)
        CONFIG.update(PROD_CONFIG)
        logger.info(f"Loaded production configuration (version: {PROD_CONFIG.get('version')})")
    else:
        dev_config_path = Path(__file__).parent.parent / 'config' / 'development.json'
        if dev_config_path.exists():
            with open(dev_config_path, 'r') as f:
                DEV_CONFIG = json.load(f)
            CONFIG.update(DEV_CONFIG)
            logger.info(f"Loaded development configuration (version: {DEV_CONFIG.get('version')})")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="AADS-ULoRA v5.5.3 API",
    description="Agricultural Disease Detection with Dynamic OOD - Production Optimized",
    version="5.5.3-performance",
    docs_url="/docs" if CONFIG.get('api', {}).get('reload', False) else None,
    redoc_url="/redoc" if CONFIG.get('api', {}).get('reload', False) else None
)

# Import and setup middleware
from api.middleware.security import setup_security_middleware
from api.middleware.auth import APIKeyMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.caching import CacheMiddleware
from api.middleware.compression import CompressionMiddleware
from api.middleware.audit import AuditMiddleware

# Setup security middleware (CORS, input limits)
setup_security_middleware(app, CONFIG)

# API Key authentication (if enabled)
if CONFIG.get('security', {}).get('api_key_required', False):
    api_keys = CONFIG.get('security', {}).get('api_keys', [])
    app.add_middleware(
        APIKeyMiddleware,
        api_keys=api_keys,
        exempt_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
    )
    logger.info("API key authentication enabled")

# Rate limiting
rate_config = CONFIG.get('security', {})
app.add_middleware(
    RateLimitMiddleware,
    max_requests=rate_config.get('rate_limit_requests', 100),
    window_seconds=rate_config.get('rate_limit_window', 60),
    exempt_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
)
logger.info(f"Rate limiting: {rate_config.get('rate_limit_requests', 100)} requests per {rate_config.get('rate_limit_window', 60)}s")

# Response caching (if enabled)
if CONFIG.get('caching', {}).get('enabled', False):
    cache_config = CONFIG.get('caching', {})
    app.add_middleware(
        CacheMiddleware,
        redis_url=cache_config.get('redis_url', 'redis://localhost:6379'),
        ttl=cache_config.get('diagnosis_ttl', 1800),
        cacheable_paths=["/v1/diagnose", "/v1/crops"],
        max_size=cache_config.get('max_cache_size', 1000)
    )
    logger.info("Response caching enabled with Redis")

# Compression
comp_config = CONFIG.get('compression', {})
app.add_middleware(
    CompressionMiddleware,
    minimum_size=comp_config.get('minimum_size', 1024),
    compression_level=comp_config.get('compression_level', 6),
    enabled=comp_config.get('enabled', True)
)
logger.info("Response compression enabled")

# Audit logging
if CONFIG.get('monitoring', {}).get('audit_logging', False):
    log_file = f"logs/audit_{asyncio.current_task().get_name()}.log" if asyncio.current_task() else "logs/audit.log"
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    app.add_middleware(AuditMiddleware, log_file=log_file)
    logger.info("Audit logging enabled")

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    logger.info("Initializing AADS-ULoRA pipeline...")
    
    try:
        # Create pipeline from config
        pipeline = IndependentMultiCropPipeline(CONFIG, device='cuda')
        
        # Initialize router
        router_path = CONFIG.get('api', {}).get('router_checkpoint', './router/crop_router_best.pth')
        pipeline.initialize_router(router_path=router_path if Path(router_path).exists() else None)
        
        # Register adapters if available
        adapters_dir = Path(CONFIG.get('api', {}).get('adapters_dir', './adapters'))
        if adapters_dir.exists():
            for crop in CONFIG['data']['crops']:
                adapter_path = adapters_dir / crop
                if adapter_path.exists():
                    pipeline.register_crop(crop, str(adapter_path))
        
        # Store pipeline in app state for thread-safe access
        app.state.pipeline = pipeline
        
        logger.info("Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if hasattr(app.state, 'pipeline'):
        logger.info("Shutting down AADS-ULoRA pipeline...")
        try:
            # Clean up pipeline resources
            pipeline = app.state.pipeline
            if hasattr(pipeline, 'cleanup'):
                pipeline.cleanup()
            del app.state.pipeline
            logger.info("Pipeline shutdown completed")
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}")

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    pipeline = request.app.state.pipeline
    status = {
        "status": "healthy" if pipeline else "unhealthy",
        "router_loaded": pipeline.router is not None if pipeline else False,
        "adapters_loaded": list(pipeline.adapters.keys()) if pipeline else [],
        "device": str(pipeline.device) if pipeline else None
    }
    return status

@app.get("/v1/crops")
async def list_crops(request: Request):
    """List supported crops."""
    pipeline = request.app.state.pipeline
    return {
        "crops": CONFIG['data']['crops'],
        "router_accuracy_target": CONFIG['targets']['crop_routing_accuracy']
    }

@app.get("/v1/adapters/{crop}/status")
async def get_adapter_status(crop: str, request: Request):
    """Get status of a specific crop adapter."""
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if crop not in pipeline.adapters:
        raise HTTPException(status_code=404, detail=f"No adapter found for crop: {crop}")
    
    adapter = pipeline.adapters[crop]
    status = {
        "crop": crop,
        "is_trained": adapter.is_trained,
        "current_phase": adapter.current_phase,
        "num_classes": len(adapter.class_to_idx) if adapter.class_to_idx else 0,
        "classes": list(adapter.class_to_idx.keys()) if adapter.class_to_idx else [],
        "has_ood": adapter.ood_thresholds is not None
    }
    
    return status

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import re
import base64
from io import BytesIO
from PIL import Image

# Import shared validation utilities
from api.validation import (
    validate_base64_image,
    validate_location_data,
    validate_crop_hint,
    validate_metadata,
    validate_uuid,
    sanitize_input
)

# Pydantic models for request validation
class LocationData(BaseModel):
    """Validated location data model."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy_meters: Optional[float] = Field(None, ge=0)
    
    @validator('latitude', 'longitude')
    def validate_coordinates(cls, v):
        """Validate coordinate precision."""
        if not isinstance(v, (float, int)):
            raise ValueError("Coordinates must be numeric")
        return float(v)

class Metadata(BaseModel):
    """Validated request metadata model."""
    capture_timestamp: Optional[str] = None
    device_model: Optional[str] = None
    os_version: Optional[str] = None
    
    @validator('capture_timestamp')
    def validate_timestamp(cls, v):
        """Validate ISO 8601 timestamp format."""
        if v is None:
            return v
        try:
            from dateutil.parser import parse
            parse(v)
            return v
        except Exception:
            raise ValueError("Invalid timestamp format. Use ISO 8601 format")
    
    @validator('device_model', 'os_version')
    def validate_string_length(cls, v, field):
        """Validate string length constraints."""
        if v is None:
            return v
        max_length = 100 if field.name == 'device_model' else 50
        if len(v) > max_length:
            raise ValueError(f"{field.name} exceeds maximum length of {max_length} characters")
        return v

class DiagnosisRequest(BaseModel):
    """Validated diagnosis request model."""
    image: str = Field(..., min_length=1, max_length=10000000)  # Base64 string
    crop_hint: Optional[str] = None
    location: Optional[LocationData] = None
    metadata: Optional[Metadata] = None
    
    @validator('crop_hint')
    def validate_crop_hint(cls, v):
        """Validate crop hint against valid crops."""
        if v:
            valid_crops = CONFIG.get('data', {}).get('crops', [])
            return validate_crop_hint(v, valid_crops)
        return v
    
    @validator('image')
    def validate_image_b64(cls, v):
        """Validate base64 image string with comprehensive checks."""
        try:
            decoded, _ = validate_base64_image(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid image: {str(e)}")
    
    @validator('location')
    def validate_location(cls, v):
        """Validate location data."""
        if v:
            return validate_location_data(v.dict())
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata."""
        if v:
            return validate_metadata(v.dict())
        return v

class BatchDiagnosisRequest(BaseModel):
    """Validated batch diagnosis request model."""
    images: List[str] = Field(..., min_items=1, max_items=10)
    crop_hint: Optional[str] = None
    location: Optional[LocationData] = None
    metadata: Optional[Metadata] = None
    
    @validator('images')
    def validate_images(cls, v):
        """Validate batch images."""
        if len(v) > 10:
            raise ValueError("Maximum 10 images per batch request")
        return v
    
    @validator('images', each_item=True)
    def validate_each_image(cls, v):
        """Validate each image in batch."""
        try:
            decoded, _ = validate_base64_image(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid image in batch: {str(e)}")
    
    @validator('crop_hint')
    def validate_crop_hint(cls, v):
        """Validate crop hint against valid crops."""
        if v:
            valid_crops = CONFIG.get('data', {}).get('crops', [])
            return validate_crop_hint(v, valid_crops)
        return v
    
    @validator('location')
    def validate_location(cls, v):
        """Validate location data."""
        if v:
            return validate_location_data(v.dict())
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validate metadata."""
        if v:
            return validate_metadata(v.dict())
        return v

class ExpertLabelRequest(BaseModel):
    """Validated expert label request model."""
    sample_id: str = Field(..., min_length=1, max_length=100)
    true_label: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: Optional[str] = Field(None, max_length=1000)
    
    @validator('sample_id')
    def validate_sample_id(cls, v):
        """Validate UUID format."""
        validate_uuid(v)
        return v
    
    @validator('true_label')
    def validate_true_label(cls, v):
        """Validate true label format."""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("True label must contain only alphanumeric characters and underscores")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence bounds."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    @validator('notes')
    def validate_notes(cls, v):
        """Validate and sanitize notes."""
        if v:
            if len(v) > 1000:
                raise ValueError("Notes exceed maximum length of 1000 characters")
            return sanitize_input(v, max_length=1000)
        return v

class DiagnosisResponse(BaseModel):
    """Standardized diagnosis response model."""
    status: str
    request_id: str
    timestamp: str
    crop: Dict[str, Any]
    disease: Optional[Dict[str, Any]] = None
    ood_analysis: Dict[str, Any]
    recommendations: Optional[Dict[str, Any]] = None
    follow_up: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None

@app.post("/v1/diagnose", response_model=DiagnosisResponse)
async def diagnose(diagnosis_request: DiagnosisRequest, request: Request):
    """
    Main diagnosis endpoint with enhanced validation.
    
    Request body:
    {
        "image": "base64_encoded_jpeg_string",
        "crop_hint": "tomato" (optional),
        "location": {
            "latitude": 41.0082,
            "longitude": 28.9784,
            "accuracy_meters": 10.0
        },
        "metadata": {
            "capture_timestamp": "2026-03-15T14:30:00Z",
            "device_model": "iPhone14,2",
            "os_version": "iOS 17.4"
        }
    }
    """
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Decode image (already validated by Pydantic)
        image_data = base64.b64decode(diagnosis_request.image)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        img_tensor = preprocess_image(image).unsqueeze(0)
        
        # Process
        result = pipeline.process_image(
            img_tensor,
            metadata={
                'location': diagnosis_request.location.dict() if diagnosis_request.location else None,
                'request_metadata': diagnosis_request.metadata.dict() if diagnosis_request.metadata else None
            }
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result.get('message', 'Unknown error'))
        
        # Format response
        response = {
            'status': 'success',
            'request_id': result.get('request_id', ''),
            'timestamp': result.get('timestamp', ''),
            'crop': {
                'predicted': result.get('crop'),
                'confidence': result.get('crop_confidence', 0.0)
            },
            'ood_analysis': result.get('ood_analysis', {}),
        }
        
        if 'disease' in result:
            response['disease'] = result['disease']
        
        if 'recommendations' in result:
            response['recommendations'] = result['recommendations']
        
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Diagnosis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/v1/diagnose/batch", response_model=List[DiagnosisResponse])
async def batch_diagnose(batch_request: BatchDiagnosisRequest, request: Request):
    """
    Batch diagnosis endpoint for multiple images.
    
    Request body:
    {
        "images": ["base64_encoded_jpeg_string_1", "base64_encoded_jpeg_string_2"],
        "crop_hint": "tomato" (optional),
        "location": {
            "latitude": 41.0082,
            "longitude": 28.9784,
            "accuracy_meters": 10.0
        },
        "metadata": {
            "capture_timestamp": "2026-03-15T14:30:00Z",
            "device_model": "iPhone14,2",
            "os_version": "iOS 17.4"
        }
    }
    """
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    results = []
    
    for i, image_b64 in enumerate(batch_request.images):
        try:
            # Decode image (already validated by Pydantic)
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data)).convert('RGB')
            
            # Preprocess
            img_tensor = preprocess_image(image).unsqueeze(0)
            
            # Process
            result = pipeline.process_image(
                img_tensor,
                metadata={
                    'location': batch_request.location.dict() if batch_request.location else None,
                    'request_metadata': batch_request.metadata.dict() if batch_request.metadata else None
                }
            )
            
            if result.get('status') == 'error':
                raise HTTPException(status_code=400, detail=result.get('message', 'Unknown error'))
            
            # Format response
            response = {
                'status': 'success',
                'request_id': f"{result.get('request_id', '')}_{i}",
                'timestamp': result.get('timestamp', ''),
                'crop': {
                    'predicted': result.get('crop'),
                    'confidence': result.get('crop_confidence', 0.0)
                },
                'ood_analysis': result.get('ood_analysis', {}),
            }
            
            if 'disease' in result:
                response['disease'] = result['disease']
            
            if 'recommendations' in result:
                response['recommendations'] = result['recommendations']
            
            results.append(response)
            
        except Exception as e:
            logger.error(f"Batch diagnosis error for image {i}: {e}")
            results.append({
                'status': 'error',
                'request_id': f"batch_{i}",
                'timestamp': torch.datetime.now().isoformat(),
                'crop': {"predicted": None, "confidence": 0.0},
                'ood_analysis': {},
                'error': str(e)
            })
    
    return results

@app.post("/v1/feedback/expert-label", response_model=ExpertLabelRequest)
async def submit_expert_label(feedback: ExpertLabelRequest):
    """
    Submit expert label for OOD samples with enhanced validation.
    
    Request body:
    {
        "sample_id": "uuid",
        "true_label": "septoria_leaf_spot",
        "confidence": 0.95,
        "notes": "Additional notes" (optional)
    }
    """
    
    logger.info(f"Received expert label for sample {feedback.sample_id}: {feedback.true_label}")
    
    # In a full implementation:
    # 1. Store the feedback in database
    # 2. Queue for Phase 2 training if enough new disease samples
    # 3. Trigger adapter retraining
    
    return {
        "status": "accepted",
        "message": "Expert label received and queued for processing",
        "sample_id": feedback.sample_id
    }

@app.post("/v1/feedback/batch")
async def submit_batch_feedback(feedback_list: List[ExpertLabelRequest]):
    """
    Submit multiple expert labels.
    
    Request body:
    [
        {
            "sample_id": "uuid",
            "true_label": "septoria_leaf_spot",
            "confidence": 0.95,
            "notes": "Additional notes" (optional)
        }
    ]
    """
    
    results = []
    for feedback in feedback_list:
        logger.info(f"Received expert label for sample {feedback.sample_id}: {feedback.true_label}")
        results.append({
            "sample_id": feedback.sample_id,
            "status": "accepted",
            "message": "Expert label received and queued for processing"
        })
    
    return {
        "status": "success",
        "processed": len(results),
        "results": results
    }

@app.get("/v1/system/info")
async def system_info(request: Request):
    """Get system information."""
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "version": CONFIG.get('version', '5.5.0'),
        "architecture": CONFIG.get('architecture'),
        "crops": CONFIG['data']['crops'],
        "router_loaded": pipeline.router is not None,
        "adapters_loaded": list(pipeline.adapters.keys()),
        "device": str(pipeline.device),
        "targets": CONFIG['targets'],
        "middleware": {
            "rate_limiting": "enabled" if CONFIG.get('security', {}).get('rate_limit_requests') else "disabled",
            "caching": "enabled" if CONFIG.get('caching', {}).get('enabled') else "disabled",
            "compression": "enabled" if CONFIG.get('compression', {}).get('enabled') else "disabled",
            "auth": "enabled" if CONFIG.get('security', {}).get('api_key_required') else "disabled"
        }
    }

# Include monitoring endpoints
from api.endpoints.monitoring import router as monitoring_router
app.include_router(monitoring_router)

# Setup graceful shutdown
from api.graceful_shutdown import setup_shutdown_handlers
shutdown_handler = setup_shutdown_handlers(app)

# Initialize database if configured
if CONFIG.get('database', {}).get('url'):
    try:
        from api.database import init_database
        init_database(CONFIG)
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Handle specific exception types
    if isinstance(exc, ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Validation error: {str(exc)}"}
        )
    elif isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_code": "INTERNAL_SERVER_ERROR"}
        )

if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG.get('api', {}).get('host', '0.0.0.0')
    port = CONFIG.get('api', {}).get('port', 8000)
    workers = CONFIG.get('api', {}).get('workers', 1)
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level=CONFIG.get('api', {}).get('log_level', 'info')
    )