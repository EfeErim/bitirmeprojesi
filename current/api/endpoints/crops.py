from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline

router = APIRouter(prefix="/v1", tags=["crops"])

pipeline = None

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
async def list_crops() -> Dict[str, List[str]]:
    """List all supported crops."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "crops": pipeline.crops,
        "router_accuracy_target": 0.98
    }

@router.get("/adapters/{crop}/status", response_model=AdapterStatusResponse)
async def get_adapter_status(crop: str) -> Dict:
    """Get status of a specific crop adapter."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
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