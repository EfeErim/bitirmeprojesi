from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import sys
import os
import uuid
import logging
import re

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["feedback"])

class ExpertLabelRequest(BaseModel):
    """Validated expert label request."""
    sample_id: str = Field(..., min_length=1, max_length=100)
    true_label: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: Optional[str] = Field(None, max_length=1000)
    
    @validator('sample_id')
    def validate_sample_id(cls, v):
        """Validate UUID format."""
        if not re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', v):
            raise ValueError("Invalid UUID format")
        return v

class BatchFeedbackRequest(BaseModel):
    """Batch feedback request."""
    feedback_list: List[ExpertLabelRequest] = Field(..., min_items=1, max_items=100)

class ExpertLabelResponse(BaseModel):
    """Response model for expert label submission."""
    status: str
    message: str
    sample_id: str

class BatchFeedbackResponse(BaseModel):
    """Response model for batch feedback submission."""
    status: str
    processed: int
    results: List[Dict[str, Any]]

@router.post("/feedback/expert-label", response_model=ExpertLabelResponse)
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
    # 1. Store in database
    # 2. Queue for Phase 2 training if new disease
    # 3. Trigger adapter retraining
    
    return {
        "status": "accepted",
        "message": "Expert label received and queued for processing",
        "sample_id": feedback.sample_id
    }

@router.post("/feedback/batch", response_model=BatchFeedbackResponse)
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