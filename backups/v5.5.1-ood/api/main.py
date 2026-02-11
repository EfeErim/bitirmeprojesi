#!/usr/bin/env python3
"""
AADS-ULoRA v5.5 FastAPI Server
Main API entry point for crop disease diagnosis.
"""

import logging
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
from pathlib import Path
import json
import base64
from io import BytesIO
from PIL import Image
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from src.utils.data_loader import preprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'adapter_spec_v55.json'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Initialize FastAPI app
app = FastAPI(
    title="AADS-ULoRA v5.5 API",
    description="Agricultural Disease Detection with Dynamic OOD",
    version="5.5.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    
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
        
        logger.info("Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy" if pipeline else "unhealthy",
        "router_loaded": pipeline.router is not None if pipeline else False,
        "adapters_loaded": list(pipeline.adapters.keys()) if pipeline else [],
        "device": str(pipeline.device) if pipeline else None
    }
    return status

@app.get("/v1/crops")
async def list_crops():
    """List supported crops."""
    return {
        "crops": CONFIG['data']['crops'],
        "router_accuracy_target": CONFIG['targets']['crop_routing_accuracy']
    }

@app.get("/v1/adapters/{crop}/status")
async def get_adapter_status(crop: str):
    """Get status of a specific crop adapter."""
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

@app.post("/v1/diagnose")
async def diagnose(request: dict):
    """
    Main diagnosis endpoint.
    
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
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Extract image from base64
        image_b64 = request.get('image')
        if not image_b64:
            raise HTTPException(status_code=400, detail="Image field is required")
        
        # Decode base64
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        img_tensor = preprocess_image(image).unsqueeze(0)
        
        # Get crop hint if provided
        crop_hint = request.get('crop_hint')
        if crop_hint and crop_hint not in CONFIG['data']['crops']:
            crop_hint = None
        
        # Process through pipeline
        result = pipeline.process_image(
            img_tensor,
            metadata={
                'location': request.get('location'),
                'request_metadata': request.get('metadata')
            }
        )
        
        # Add request_id
        import uuid
        result['request_id'] = str(uuid.uuid4())
        result['timestamp'] = torch.datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/feedback/expert-label")
async def submit_expert_label(feedback: dict):
    """
    Submit expert label for OOD samples.
    
    Request body:
    {
        "sample_id": "uuid",
        "true_label": "septoria_leaf_spot",
        "confidence": 0.95
    }
    """
    # In a full implementation, this would:
    # 1. Store the feedback in database
    # 2. Trigger Phase 2 training if enough new disease samples
    # 3. Update adapter version
    
    logger.info(f"Received expert label: {feedback}")
    
    return {
        "status": "accepted",
        "message": "Expert label received and queued for processing",
        "sample_id": feedback.get('sample_id')
    }

@app.get("/v1/system/info")
async def system_info():
    """Get system information."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "version": CONFIG.get('version', '5.5.0'),
        "architecture": CONFIG.get('architecture'),
        "crops": CONFIG['data']['crops'],
        "router_loaded": pipeline.router is not None,
        "adapters_loaded": list(pipeline.adapters.keys()),
        "device": str(pipeline.device),
        "targets": CONFIG['targets']
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
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
        reload=False
    )