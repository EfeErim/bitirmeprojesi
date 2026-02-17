from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

from src.utils.data_loader import preprocess_image

router = APIRouter(prefix="/v1", tags=["diagnosis"])

class DiagnosisRequest(BaseModel):
    image: str
    crop_hint: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class DiagnosisResponse(BaseModel):
    status: str
    request_id: str
    timestamp: str
    crop: Dict[str, Any]
    disease: Optional[Dict[str, Any]] = None
    ood_analysis: Dict[str, Any]
    recommendations: Optional[Dict[str, Any]] = None
    follow_up: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None

@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(diagnosis_request: DiagnosisRequest, request: Request):
    """Main diagnosis endpoint."""
    pipeline = getattr(request.app.state, 'pipeline', None)
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Validate image size before decoding (DOS protection)
        MAX_BASE64_SIZE = 50 * 1024 * 1024  # 50MB
        if len(diagnosis_request.image) > MAX_BASE64_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Request too large: image exceeds 50MB limit"
            )
        
        # Decode image
        image_data = base64.b64decode(diagnosis_request.image)
        
        # Validate decoded size
        MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
        if len(image_data) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image too large: decoded image exceeds 50MB limit"
            )
        
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        img_tensor = preprocess_image(image).unsqueeze(0)
        
        # Process
        result = pipeline.process_image(
            img_tensor,
            metadata={
                'location': diagnosis_request.location,
                'request_metadata': diagnosis_request.metadata
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))