#!/usr/bin/env python3
"""Test SAM3 raw outputs without any filtering."""

import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_sam3_raw(image_path: str, prompt: str = "plant"):
    """Test SAM3 with minimal processing to see raw outputs."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading image: {image_path}")
    
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    print(f"\nLoading SAM3 model...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    
    print(f"\nRunning SAM3 with prompt: '{prompt}'")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\nRaw outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}")
    
    # Check what's in outputs
    if hasattr(outputs, 'pred_masks'):
        print(f"pred_masks shape: {outputs.pred_masks.shape}")
        print(f"pred_masks min/max: {outputs.pred_masks.min():.4f} / {outputs.pred_masks.max():.4f}")
    
    if hasattr(outputs, 'pred_boxes'):
        print(f"pred_boxes shape: {outputs.pred_boxes.shape}")
    
    if hasattr(outputs, 'pred_scores'):
        print(f"pred_scores shape: {outputs.pred_scores.shape}")
        print(f"pred_scores values: {outputs.pred_scores}")
    
    # Try post-processing at different thresholds
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        try:
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            masks = results.get('masks', torch.tensor([]))
            boxes = results.get('boxes', torch.tensor([]))
            scores = results.get('scores', torch.tensor([]))
            
            n_masks = masks.shape[0] if torch.is_tensor(masks) and masks.ndim > 0 else 0
            print(f"\nThreshold {threshold:.1f}: {n_masks} instances")
            
            if n_masks > 0 and torch.is_tensor(scores):
                print(f"  Scores: {scores.tolist()}")
                
        except Exception as e:
            print(f"\nThreshold {threshold:.1f}: ERROR - {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--prompt", default="plant", help="Text prompt")
    args = parser.parse_args()
    
    test_sam3_raw(args.image, args.prompt)
