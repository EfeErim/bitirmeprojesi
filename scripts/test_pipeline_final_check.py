#!/usr/bin/env python3
"""
Final VLM pipeline sanity check without model downloads.
Tests instantiation, disabled mode, and basic API surfaces.
Canonical script location: scripts/test_pipeline_final_check.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image


def test_disabled_mode():
    """Test pipeline in disabled mode (no model loads)."""
    print("\n" + "=" * 70)
    print("FINAL PIPELINE SANITY CHECK")
    print("=" * 70)

    from src.router.vlm_pipeline import VLMPipeline

    config = {
        'vlm_enabled': False,  # Disabled mode
        'vlm_confidence_threshold': 0.7,
        'router': {
            'crop_mapping': {
                'tomato': {'parts': ['leaf', 'fruit']},
                'potato': {'parts': ['leaf', 'tuber']},
            },
            'vlm': {
                'enabled': False,
                'confidence_threshold': 0.7,
            }
        }
    }

    print("\n1. Creating pipeline (disabled mode)...")
    pipeline = VLMPipeline(config=config, device='cpu')
    print("   [OK] Pipeline instantiated")

    print("\n2. Checking attributes...")
    attrs = ['enabled', 'confidence_threshold', 'crop_labels', 'part_labels', 'open_set_enabled']
    for attr in attrs:
        val = getattr(pipeline, attr, '<missing>')
        print(f"   {attr}: {val}")
    print("   [OK] Attributes accessible")

    print("\n3. Testing is_ready() in disabled mode...")
    ready = pipeline.is_ready()
    assert not ready, "is_ready() should be False in disabled mode"
    print(f"   [OK] is_ready() = {ready} (correct)")

    print("\n4. Testing analyze_image() in disabled mode...")
    dummy_image = torch.randn(3, 224, 224)
    result = pipeline.analyze_image(dummy_image, confidence_threshold=0.5, max_detections=3)
    assert 'detections' in result
    assert 'image_size' in result
    assert 'processing_time_ms' in result
    assert len(result['detections']) == 0, "Disabled mode should return empty detections"
    print(f"   [OK] Returns valid structure with {len(result['detections'])} detections")

    print("\n5. Testing process_image() in disabled mode...")
    proc_result = pipeline.process_image(dummy_image)
    assert 'status' in proc_result
    assert 'scenario' in proc_result
    assert proc_result['status'] == 'ok'
    print(f"   [OK] process_image() returns status={proc_result['status']}")

    print("\n6. Testing route_batch() in disabled mode...")
    batch = torch.randn(2, 3, 224, 224)
    crops_out, confs = pipeline.route_batch(batch)
    assert len(crops_out) == 2
    assert len(confs) == 2
    assert all(c == 0.0 for c in confs), "Disabled mode should return 0.0 confidences"
    print(f"   [OK] route_batch() returns {len(crops_out)} crops")

    print("\n" + "=" * 70)
    print("[OK] ALL CHECKS PASSED - Pipeline is stable in disabled mode")
    print("=" * 70 + "\n")


def test_api_imports():
    """Test that imports work and don't crash."""
    print("\n" + "=" * 70)
    print("IMPORT CHECK")
    print("=" * 70)

    print("\n1. Importing VLMPipeline...")
    from src.router.vlm_pipeline import VLMPipeline
    print("   [OK] VLMPipeline")

    print("\n2. Importing DiagnosticScoutingAnalyzer...")
    from src.router.vlm_pipeline import DiagnosticScoutingAnalyzer
    print("   [OK] DiagnosticScoutingAnalyzer")

    print("\n3. Testing analyzer instantiation...")
    config = {'vlm_enabled': False}
    analyzer = DiagnosticScoutingAnalyzer(config, device='cpu')
    print(f"   [OK] Analyzer created, device={analyzer.device}")

    print("\n4. Testing analyzer.quick_assessment()...")
    dummy = torch.randn(3, 224, 224)
    assessment = analyzer.quick_assessment(dummy)
    assert 'status' in assessment
    assert 'explanation' in assessment
    print(f"   [OK] quick_assessment() returns status={assessment['status']}")

    print("\n" + "=" * 70)
    print("[OK] ALL IMPORTS & ANALYZER API PASSED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    try:
        test_api_imports()
        test_disabled_mode()
        print("\n[DONE] FINAL CHECK: ALL SYSTEMS NOMINAL\n")
    except Exception as e:
        print(f"\n[FAIL] FINAL CHECK FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


