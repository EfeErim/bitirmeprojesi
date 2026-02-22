#!/usr/bin/env python3
"""
Standalone VLM Pipeline Test Script
====================================
Tests the SAM3 + BioCLIP-2.5 pipeline end-to-end without any adapter
dependencies.  Run from the project root:

    python scripts/test_vlm_pipeline_standalone.py

Optional flags:
    --token-file <path>   Path to a text file containing the HF token.
    --image <path>        Path to a test image (PNG/JPG).  If omitted, a
                          synthetic test image is generated automatically.
    --device <cpu|cuda>   Force device.  Default: auto-detect.
    --skip-model-load     Import-only smoke test (no model download).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm_test")

# ─── Helpers ────────────────────────────────────────────────────────────────

def read_token(token_file: str | None) -> str | None:
    """Read HF token from file, env, or well-known location."""
    # 1. Explicit file
    if token_file and Path(token_file).is_file():
        token = Path(token_file).read_text().strip()
        if token:
            logger.info(f"HF token loaded from {token_file}")
            return token

    # 2. Environment variable
    token = os.getenv("HF_TOKEN")
    if token:
        logger.info("HF token loaded from $HF_TOKEN")
        return token

    # 3. Well-known Windows location
    well_known = Path.home() / "Desktop" / "huggingfacetoken.txt"
    if well_known.is_file():
        token = well_known.read_text().strip()
        if token:
            logger.info(f"HF token loaded from {well_known}")
            return token

    return None


def generate_synthetic_image():
    """Create a simple synthetic 640x480 RGB image for testing."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise RuntimeError("Pillow is required: pip install Pillow")

    img = Image.new("RGB", (640, 480), color=(34, 139, 34))  # forest-green base
    draw = ImageDraw.Draw(img)
    # Simulate a leaf-like ellipse
    draw.ellipse([160, 80, 480, 400], fill=(50, 205, 50), outline=(0, 100, 0), width=3)
    # Add some spots (simulate disease)
    for cx, cy, r in [(280, 200, 18), (350, 260, 14), (310, 320, 10)]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(139, 69, 19))
    logger.info("Generated synthetic 640x480 test image")
    return img


# ─── Test stages ────────────────────────────────────────────────────────────

def test_imports():
    """Stage 1: Verify all VLM imports resolve."""
    logger.info("=" * 60)
    logger.info("Stage 1: Checking imports")
    logger.info("=" * 60)

    errors = []

    # Core router module
    try:
        from router.vlm_pipeline import VLMPipeline  # noqa: F401
        logger.info("  [OK] router.vlm_pipeline.VLMPipeline")
    except Exception as e:
        errors.append(f"VLMPipeline import: {e}")
        logger.error(f"  [FAIL] VLMPipeline: {e}")

    # SAM3 from transformers
    try:
        from transformers import Sam3Processor, Sam3Model  # noqa: F401
        logger.info("  [OK] transformers.Sam3Processor / Sam3Model")
    except Exception as e:
        errors.append(f"SAM3 import: {e}")
        logger.error(f"  [FAIL] SAM3: {e}")

    # open_clip for BioCLIP-2.5
    try:
        import open_clip  # noqa: F401
        logger.info(f"  [OK] open_clip {open_clip.__version__}")
    except Exception as e:
        errors.append(f"open_clip import: {e}")
        logger.error(f"  [FAIL] open_clip: {e}")

    # torch
    try:
        import torch  # noqa: F401
        logger.info(f"  [OK] torch {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
    except Exception as e:
        errors.append(f"torch: {e}")

    # PIL
    try:
        from PIL import Image  # noqa: F401
        logger.info("  [OK] Pillow")
    except Exception as e:
        errors.append(f"Pillow: {e}")

    if errors:
        logger.error(f"\n{len(errors)} import error(s):")
        for err in errors:
            logger.error(f"  - {err}")
        return False
    logger.info("All imports OK\n")
    return True


def test_pipeline_construction(device: str):
    """Stage 2: Construct VLMPipeline with project config."""
    logger.info("=" * 60)
    logger.info("Stage 2: Constructing VLMPipeline")
    logger.info("=" * 60)

    config_path = PROJECT_ROOT / "config" / "base.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"  Loaded config from {config_path}")
    else:
        config = {}
        logger.warning("  No base.json found; using defaults")

    from router.vlm_pipeline import VLMPipeline

    pipeline = VLMPipeline(config=config, device=device)
    logger.info(f"  Pipeline created  (enabled={pipeline.enabled},  device={pipeline.device})")
    logger.info(f"  model_ids = {pipeline.model_ids}")
    return pipeline


def test_model_loading(pipeline):
    """Stage 3: Load SAM3 + BioCLIP-2.5 weights."""
    logger.info("=" * 60)
    logger.info("Stage 3: Loading models (may download ~2 GB on first run)")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    pipeline.load_models()
    elapsed = time.perf_counter() - t0

    ready = pipeline.is_ready()
    logger.info(f"  is_ready() = {ready}  ({elapsed:.1f}s)")

    if not ready:
        logger.error("  Pipeline is NOT ready after load_models()")
        return False

    # Quick sanity: check internal model handles
    logger.info(f"  sam_model   : {type(pipeline.sam_model).__name__}")
    logger.info(f"  sam_processor: {type(pipeline.sam_processor).__name__}")
    logger.info(f"  bioclip      : {type(pipeline.bioclip).__name__}")
    logger.info(f"  bioclip_backend: {pipeline.bioclip_backend}")
    logger.info("Model loading OK\n")
    return True


def test_analyze_image(pipeline, pil_image):
    """Stage 4: Run analyze_image on a test image."""
    logger.info("=" * 60)
    logger.info("Stage 4: Running analyze_image()")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    result = pipeline.analyze_image(
        pil_image,
        confidence_threshold=0.5,  # lower bar for test image
        max_detections=5,
    )
    elapsed = time.perf_counter() - t0

    detections = result.get("detections", [])
    logger.info(f"  Detections: {len(detections)}  ({elapsed:.1f}s)")
    logger.info(f"  Image size: {result.get('image_size')}")
    logger.info(f"  Processing time: {result.get('processing_time_ms', 0):.1f}ms")

    for i, det in enumerate(detections):
        label = det.get("label", "?")
        conf = det.get("confidence", 0)
        bbox = det.get("bbox")
        logger.info(f"    [{i}] {label}  conf={conf:.3f}  bbox={bbox}")

    logger.info("analyze_image() completed OK\n")
    return result


def test_route_batch(pipeline, pil_image):
    """Stage 5: Test route_batch with a single-image batch."""
    logger.info("=" * 60)
    logger.info("Stage 5: Testing route_batch()")
    logger.info("=" * 60)

    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(pil_image).unsqueeze(0)  # [1, 3, 224, 224]

    t0 = time.perf_counter()
    results = pipeline.route_batch(tensor)
    elapsed = time.perf_counter() - t0

    logger.info(f"  Batch results: {len(results)} item(s)  ({elapsed:.1f}s)")
    for i, r in enumerate(results):
        crop = r.get("crop", "?")
        conf = r.get("confidence", 0)
        logger.info(f"    [{i}] crop={crop}  confidence={conf:.3f}")

    logger.info("route_batch() completed OK\n")
    return results


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Standalone VLM Pipeline Test")
    parser.add_argument("--token-file", type=str, default=None, help="Path to HF token file")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--skip-model-load", action="store_true", help="Import-only smoke test")
    args = parser.parse_args()

    # Determine device
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # HF authentication
    token = read_token(args.token_file)
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("Authenticated with HuggingFace")
        except Exception as e:
            logger.warning(f"HF login failed: {e}")
    else:
        logger.warning("No HF token found — gated models (SAM3) will fail")

    # Prepare test image
    if args.image:
        from PIL import Image
        pil_image = Image.open(args.image).convert("RGB")
        logger.info(f"Test image: {args.image}  ({pil_image.size})")
    else:
        pil_image = generate_synthetic_image()

    # ── Run stages ──
    passed = 0
    failed = 0

    # Stage 1: Imports
    if test_imports():
        passed += 1
    else:
        failed += 1
        logger.error("ABORT: fix import errors before continuing")
        sys.exit(1)

    # Stage 2: Construction
    try:
        pipeline = test_pipeline_construction(device)
        passed += 1
    except Exception as e:
        logger.error(f"Stage 2 FAILED: {e}")
        failed += 1
        sys.exit(1)

    if args.skip_model_load:
        logger.info("\n--skip-model-load: skipping stages 3-5")
        print(f"\n{'='*60}")
        print(f"  RESULT: {passed} passed, {failed} failed  (model load skipped)")
        print(f"{'='*60}")
        sys.exit(0 if failed == 0 else 1)

    # Stage 3: Model loading
    try:
        if test_model_loading(pipeline):
            passed += 1
        else:
            failed += 1
    except Exception as e:
        logger.error(f"Stage 3 FAILED: {e}")
        failed += 1

    # Stage 4: analyze_image
    if pipeline.is_ready():
        try:
            test_analyze_image(pipeline, pil_image)
            passed += 1
        except Exception as e:
            logger.error(f"Stage 4 FAILED: {e}")
            failed += 1
    else:
        logger.warning("Skipping Stage 4 (pipeline not ready)")

    # Stage 5: route_batch
    if pipeline.is_ready():
        try:
            test_route_batch(pipeline, pil_image)
            passed += 1
        except Exception as e:
            logger.error(f"Stage 5 FAILED: {e}")
            failed += 1
    else:
        logger.warning("Skipping Stage 5 (pipeline not ready)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  VLM PIPELINE TEST RESULT: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
