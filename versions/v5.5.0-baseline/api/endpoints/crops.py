from fastapi import APIRouter, HTTPException
from typing import Dict, List
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline

router = APIRouter(prefix="/v1", tags=["crops"])

pipeline = None

@router.get("/crops")
async def list_crops() -> Dict[str, List[str]]:
    """List all supported crops."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "crops": pipeline.crops,
        "router_accuracy_target": 0.98
    }

@router.get("/adapters/{crop}/status")
async def get_adapter_status(crop: str) -> Dict:
    """Get status of a specific crop adapter."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
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