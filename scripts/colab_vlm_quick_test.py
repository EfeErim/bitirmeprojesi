#!/usr/bin/env python3
"""
Quick VLM Pipeline Test for Colab - No config file needed!

Usage in Colab:
    %run scripts/colab_vlm_quick_test.py
"""

import sys
import subprocess
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add project to path

def ensure_ultralytics():
    """Ensure ultralytics is installed for SAM2 backend."""
    try:
        import ultralytics  # noqa: F401
        return
    except Exception:
        print("📦 Installing ultralytics for SAM2 backend...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.vlm_pipeline import VLMPipeline


def ensure_dependencies():
    """Install runtime dependencies required for SAM2 + BioCLIP2 in Colab."""
    packages = [
        'ultralytics',
        'open-clip-torch',
    ]
    ensure_ultralytics()
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])


def main():
    """Run interactive VLM test with image upload."""
    ensure_dependencies()
    
    # Configuration
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': True,
        'router': {
            'crop_mapping': {
                'tomato': {'parts': ['leaf', 'fruit', 'stem', 'whole']},
                'potato': {'parts': ['leaf', 'tuber', 'stem', 'whole']},
                'wheat': {'parts': ['leaf', 'ear', 'stem', 'whole']},
                'corn': {'parts': ['leaf', 'ear', 'stem', 'whole']},
                'grape': {'parts': ['leaf', 'fruit', 'stem', 'whole']},
                'apple': {'parts': ['leaf', 'fruit', 'stem', 'whole']},
                'pepper': {'parts': ['leaf', 'fruit', 'stem', 'whole']},
                'cucumber': {'parts': ['leaf', 'fruit', 'stem', 'whole']},
                'strawberry': {'parts': ['leaf', 'fruit', 'stem', 'whole']},
                'soybean': {'parts': ['leaf', 'pod', 'stem', 'whole']},
            },
            'vlm': {
                'enabled': True,
                'strict_model_loading': True,
                'model_source': 'huggingface',
                'model_ids': {
                    'grounding_dino': 'IDEA-Research/grounding-dino-base',
                    'sam': 'sam2_b.pt',
                    'bioclip': 'imageomics/bioclip-2'
                },
                'confidence_threshold': 0.3,
                'max_detections': 5
            }
        }
    }
    
    # Initialize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🚀 VLM Pipeline Test")
    print(f"{'='*60}")
    print(f"Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    print("🔧 Initializing VLM Pipeline...")
    pipeline = VLMPipeline(config=config, device=device)
    
    # Load models
    print("\n⏳ Loading models (GroundingDINO + SAM2 + BioCLIP2)...")
    print("   First run may take 2-3 minutes to download...\n")
    
    import time
    start = time.time()
    pipeline.load_models()
    elapsed = time.time() - start
    
    print(f"\n✅ Models loaded in {elapsed:.1f}s")
    print(f"   - SAM backend: {pipeline.sam_backend}")
    print(f"   - BioCLIP backend: {pipeline.bioclip_backend}")
    
    # Image upload
    print(f"\n{'='*60}")
    print("📤 Upload Image")
    print(f"{'='*60}\n")
    
    try:
        from google.colab import files
        import io
        
        uploaded = files.upload()
        
        if not uploaded:
            print("❌ No image uploaded. Exiting.")
            return
        
        filename = list(uploaded.keys())[0]
        test_image = Image.open(io.BytesIO(uploaded[filename])).convert('RGB')
        
        # Display
        plt.figure(figsize=(10, 8))
        plt.imshow(test_image)
        plt.axis('off')
        plt.title(f"Uploaded: {filename}")
        plt.tight_layout()
        plt.show()
        
        print(f"\n✅ Image loaded: {test_image.size[0]}x{test_image.size[1]} pixels")
        
    except ImportError:
        print("⚠️ Not running in Colab - using test image path")
        test_image_path = input("Enter image path: ")
        test_image = Image.open(test_image_path).convert('RGB')
        filename = Path(test_image_path).name
    
    # Run VLM analysis
    print(f"\n{'='*60}")
    print("🔍 Running VLM Analysis")
    print(f"{'='*60}\n")
    print("Pipeline stages:")
    print("  1️⃣ GroundingDINO: Detect plant regions")
    print("  2️⃣ SAM2: Segment detected regions")
    print("  3️⃣ BioCLIP2: Classify crop type & plant part\n")
    
    # Prepare tensor
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    image_tensor = transform(test_image)
    
    # Analyze
    start = time.time()
    result = pipeline.analyze_image(
        image_tensor,
        confidence_threshold=0.3,
        max_detections=5
    )
    elapsed = time.time() - start
    
    # Results
    print(f"✅ Analysis complete in {elapsed:.2f}s ({elapsed*1000:.0f}ms)")
    print(f"\n{'='*60}")
    print(f"📊 Results")
    print(f"{'='*60}\n")
    
    detections = result.get('detections', [])
    print(f"Found {len(detections)} detection(s)\n")
    
    if detections:
        for i, det in enumerate(detections, 1):
            print(f"Detection #{i}:")
            print(f"  🌱 Crop: {det.get('crop', 'unknown')}")
            print(f"     Confidence: {det.get('crop_confidence', 0):.1%}")
            print(f"  🍃 Part: {det.get('part', 'unknown')}")
            print(f"     Confidence: {det.get('part_confidence', 0):.1%}")
            if det.get('grounding_label'):
                print(f"  📍 GroundingDINO detected: {det['grounding_label']}")
            bbox = det.get('bbox', [])
            if bbox:
                print(f"  📦 BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            print()
    else:
        print("⚠️ No detections found. Try:")
        print("  - Lower confidence threshold (currently 0.3)")
        print("  - Different image with clearer plant features")
    
    # Raw detections
    if result.get('raw_detections'):
        print(f"\n🔍 Raw GroundingDINO Detections:")
        for i, raw in enumerate(result['raw_detections'], 1):
            print(f"  {i}. {raw.get('label')} (score: {raw.get('score', 0):.3f})")
    
    print(f"\n{'='*60}")
    print("✅ Test Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
