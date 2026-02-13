from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import sys
import os
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

router = APIRouter(prefix="/v1", tags=["crops"])

class CropListResponse(BaseModel):
    """Response model for crop list."""
    crops: List[str]
    router_accuracy_target: float = Field(0.98, ge=0, le=1)

class AdapterStatusResponse(BaseModel):
    """Response model for adapter status."""
    crop: str
    is_trained: bool
    current_phase: str
    num_classes: int
    classes: List[str]
    has_ood: bool

@router.get("/crops", response_model=CropListResponse)
async def list_crops(request: Request) -> Dict[str, List[str]]:
    """List all supported crops."""
    pipeline = request.app.state.pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "crops": pipeline.crops,
        "router_accuracy_target": 0.98
    }

@router.get("/adapters/{crop}/status", response_model=AdapterStatusResponse)
async def get_adapter_status(crop: str, request: Request) -> Dict:
    """
    Get status of a specific crop adapter.
    
    Path parameters:
        crop: Crop name to check status for (alphanumeric, hyphens, underscores)
    
    Returns:
        Adapter status information
        
    Raises:
        400: Invalid crop parameter
        404: Crop not found or adapter not available
        503: Service not initialized
    """
    pipeline = request.app.state.pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Sanitize and validate crop parameter
    if not crop or not isinstance(crop, str):
        raise HTTPException(status_code=400, detail="Invalid crop parameter")
    
    # Sanitize input to prevent injection
    crop_sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', crop)
    if crop_sanitized != crop:
        raise HTTPException(status_code=400, detail="Crop name contains invalid characters")
    
    # Validate crop exists
    if crop not in pipeline.crops:
        raise HTTPException(status_code=404, detail=f"Invalid crop: {crop}")
    
    if crop not in pipeline.adapters:
        raise HTTPException(status_code=404, detail=f"No adapter found for crop: {crop}")
    
    adapter = pipeline.adapters[crop]
    
    return {
        "crop": crop,
        "is_trained": adapter.is_trained,
        "current_phase": adapter.current_phase,
        "num_classes": len(adapter.class_to_idx) if adapter.class_to_idx else 0,
        "classes": list(adapter.class_to_idx.keys()) if adapter.class_to_idx else [],
        "has_ood": adapter.ood_thresholds is not None
    }