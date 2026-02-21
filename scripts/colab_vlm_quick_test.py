#!/usr/bin/env python3
"""
Quick VLM Pipeline Test for Colab - No config file needed!

Usage in Colab:
    %run scripts/colab_vlm_quick_test.py
"""

import sys
import subprocess
import argparse
import traceback
import platform
import json
from datetime import datetime
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


def _is_colab() -> bool:
    return 'google.colab' in sys.modules


def _safe_import_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, '__version__', 'unknown')
    except Exception:
        return 'not-installed'


def _run_preflight() -> None:
    print("\n🩺 Preflight")
    print(f"  Python: {platform.python_version()}")
    print(f"  Torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  Colab mode: {_is_colab()}")
    
    # Check critical dependencies
    print("\n📦 Dependency Check")
    critical_packages = {
        'transformers': 'transformers',
        'open_clip': 'open-clip-torch',
        'ultralytics': 'ultralytics',
        'groundingdino': 'groundingdino-hf',
    }
    
    missing_deps = []
    for pkg_name, pip_name in critical_packages.items():
        try:
            __import__(pkg_name)
            print(f"  ✅ {pkg_name}: {_safe_import_version(pkg_name)}")
        except ImportError:
            print(f"  ❌ {pkg_name}: NOT INSTALLED (run: !pip install {pip_name})")
            missing_deps.append(pip_name)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies detected!")
        print(f"Install them with:")
        print(f"  !pip install {' '.join(missing_deps)}")
        print(f"\nOr run the setup script:")
        print(f"  %run scripts/colab_setup_dependencies.py")
        raise RuntimeError(f"Missing required packages: {', '.join(missing_deps)}")


def _write_error_report(exc: Exception) -> Path:
    report_dir = PROJECT_ROOT / 'logs'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'colab_vlm_quick_test_error.json'

    payload = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'python': platform.python_version(),
        'platform': platform.platform(),
        'colab_mode': _is_colab(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'ultralytics_version': _safe_import_version('ultralytics'),
        'open_clip_version': _safe_import_version('open_clip'),
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'traceback': traceback.format_exc(),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return report_path


def _print_actionable_hint(exc: Exception) -> None:
    message = str(exc).lower()
    print("\n❌ Quick Test Failed")
    if 'no module named' in message and 'ultralytics' in message:
        print("Hint: ultralytics install failed. Re-run and ensure internet is available in Colab runtime.")
    elif 'requires notebook kernel' in message or 'could not load image via colab uploader' in message:
        print("Hint: use `%run scripts/colab_vlm_quick_test.py` in a notebook cell, or run `!python ... --image /content/file.jpg`.")
    elif 'strict vlm model loading failed' in message or 'sam-2 requires ultralytics' in message:
        print("Hint: SAM2 backend failed. Verify `ultralytics` is installed and checkpoint download is allowed.")
    else:
        print("Hint: check generated report file for full traceback and environment snapshot.")


def _run_backend_health_check(pipeline: VLMPipeline) -> bool:
    """Validate that all expected backends and model handles are loaded."""
    checks = {
        'pipeline_ready': pipeline.is_ready(),
        'grounding_dino_loaded': pipeline.grounding_dino is not None,
        'sam_loaded': pipeline.sam2 is not None,
        'bioclip_loaded': pipeline.bioclip is not None,
        'sam_backend_ultralytics': pipeline.sam_backend == 'ultralytics',
        'bioclip_backend_open_clip': pipeline.bioclip_backend == 'open_clip',
    }

    print("\n🧪 Backend Health Check")
    for name, passed in checks.items():
        status = '✅' if passed else '❌'
        print(f"  {status} {name}")

    return all(checks.values())


def ensure_dependencies():
    """Install runtime dependencies required for SAM2 + BioCLIP2 in Colab."""
    packages = [('ultralytics', 'ultralytics'), ('open-clip-torch', 'open_clip')]
    ensure_ultralytics()
    for package_name, import_name in packages:
        try:
            __import__(import_name)
        except Exception:
            print(f"📦 Installing {package_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package_name])


def _load_test_image(cli_image_path: str = '') -> tuple[Image.Image, str]:
    """Load image from Colab uploader (notebook mode) or CLI path."""
    if cli_image_path:
        image_path = Path(cli_image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")
        return Image.open(image_path).convert('RGB'), image_path.name

    try:
        from google.colab import files
        from IPython import get_ipython
        import io

        if get_ipython() is None:
            raise RuntimeError(
                "Colab uploader requires notebook kernel. "
                "Run with %run in a cell or pass --image /path/to/file.jpg when using !python."
            )

        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No image uploaded")

        filename = list(uploaded.keys())[0]
        test_image = Image.open(io.BytesIO(uploaded[filename])).convert('RGB')
        return test_image, filename
    except Exception as exc:
        raise RuntimeError(
            "Could not load image via Colab uploader. "
            "Use notebook mode (%run scripts/colab_vlm_quick_test.py) or provide --image <path>."
        ) from exc


def main(cli_image_path: str = '', health_only: bool = False):
    """Run interactive VLM test with image upload."""
    _run_preflight()
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

    if not _run_backend_health_check(pipeline):
        raise RuntimeError('Backend health-check failed. See checklist above.')

    if health_only:
        print("\n✅ Health-only mode complete.")
        return
    
    # Image upload
    print(f"\n{'='*60}")
    print("📤 Upload Image")
    print(f"{'='*60}\n")
    
    try:
        test_image, filename = _load_test_image(cli_image_path=cli_image_path)
    except Exception as e:
        print(f"❌ {e}")
        return

    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(test_image)
    plt.axis('off')
    plt.title(f"Uploaded: {filename}")
    plt.tight_layout()
    plt.show()

    print(f"\n✅ Image loaded: {test_image.size[0]}x{test_image.size[1]} pixels")
    
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
    parser = argparse.ArgumentParser(description='Quick VLM Pipeline Test for Colab')
    parser.add_argument('--image', type=str, default='', help='Optional image path for !python mode')
    parser.add_argument('--health-only', action='store_true', help='Only run backend health checks and exit')
    args = parser.parse_args()
    try:
        main(cli_image_path=args.image, health_only=args.health_only)
    except Exception as exc:
        _print_actionable_hint(exc)
        report_path = _write_error_report(exc)
        print(f"Report saved: {report_path}")
