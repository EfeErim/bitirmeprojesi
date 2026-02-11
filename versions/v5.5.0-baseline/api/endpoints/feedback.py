from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["feedback"])

class ExpertLabelRequest(BaseModel):
    sample_id: str
    true_label: str
    confidence: float = 1.0
    notes: Optional[str] = None

class ExpertLabelResponse(BaseModel):
    status: str
    message: str
    sample_id: str

@router.post("/feedback/expert-label", response_model=ExpertLabelResponse)
async def submit_expert_label(feedback: ExpertLabelRequest):
    """Submit expert label for OOD samples."""
    
    logger.info(f"Received expert label for sample {feedback.sample_id}: {feedback.true_label}")
    
    # In a full implementation:
    # 1. Store in database
    # 2. Queue for Phase 2 training if new disease
    # 3. Trigger adapter retraining
    
    return {
        "status": "accepted",
        "message": "Expert label received and queued for processing",
        "sample_id": feedback.sample_id
    }

@router.post("/feedback/batch")
async def submit_batch_feedback(feedback_list: list):
    """Submit multiple expert labels."""
    
    results = []
    for feedback in feedback_list:
        # Process each
        results.append({
            "sample_id": feedback.get('sample_id'),
            "status": "accepted"
        })
    
    return {
        "status": "success",
        "processed": len(results),
        "results": results
    }