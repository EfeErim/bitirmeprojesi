#!/usr/bin/env python3
"""
Quick test of VLM pipeline in Colab with GPU support and debug logging.
Designed to test image encoding and verify that BioCLIP is working correctly.
"""

import logging
import torch
import sys
from pathlib import Path
from PIL import Image
import urllib.request

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.router.vlm_pipeline import VLMPipeline

def test_vlm_pipeline():
    """Test VLM pipeline with a real image."""
    
    # Test configuration
    config = {
        'vlm_enabled': True,
        'vlm_confidence_threshold': 0.5,
        'vlm_max_detections': 10,
        'vlm_strict_model_loading': False,
        'vlm_model_source': 'huggingface'
    }
    
    # Initialize pipeline
    print("\n" + "="*80)
    print("INITIALIZING VLM PIPELINE")
    print("="*80)
    pipeline = VLMPipeline(config, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    pipeline.load_models()
    
    if not pipeline.is_ready():
        print("ERROR: Pipeline not ready after loading models")
        return
    
    print("\nPipeline is ready!")
    
    # Test with a downloaded image
    print("\n" + "="*80)
    print("DOWNLOADING TEST IMAGE")
    print("="*80)
    
    strawberry_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Fragaria_x_ananassa_Foto_by_CF_Weise.jpg/800px-Fragaria_x_ananassa_Foto_by_CF_Weise.jpg"
    image_path = "/tmp/test_strawberry.jpg"
    
    try:
        urllib.request.urlretrieve(strawberry_url, image_path)
        image = Image.open(image_path)
        print(f"Downloaded test image: {image.size}")
    except Exception as e:
        print(f"Failed to download image: {e}")
        # Create a simple test image instead
        image = Image.new('RGB', (224, 224), color=(200, 0, 0))  # Red image
        print(f"Using simple red test image: {image.size}")
    
    # Test crop classification
    print("\n" + "="*80)
    print("TESTING CROP CLASSIFICATION")
    print("="*80)
    print(f"Testing with crop labels: {pipeline.crop_labels}")
    
    crop_label, confidence = pipeline._classify_with_prompt_ensemble(image, 'crop')
    print(f"\nResult: {crop_label} with confidence {confidence:.1%}")
    
    # Test part classification if available
    if pipeline.part_labels:
        print("\n" + "="*80)
        print("TESTING PART CLASSIFICATION")
        print("="*80)
        print(f"Testing with part labels: {pipeline.part_labels}")
        
        part_label, confidence = pipeline._classify_with_prompt_ensemble(image, 'part')
        print(f"\nResult: {part_label} with confidence {confidence:.1%}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    test_vlm_pipeline()
