#!/usr/bin/env python3
"""Test if package imports work correctly."""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing imports...")

# Test imports
try:
    from aads_ulora_v55.utils.data_loader import CropDataset
    print("[OK] aads_ulora_v55.utils.data_loader imported successfully")
except Exception as e:
    print(f"[ERROR] aads_ulora_v5.5.utils.data_loader import failed: {e}")

# Test imports
try:
    from aads_ulora_v55.router.vlm_pipeline import VLMPipeline
    print("[OK] aads_ulora_v55.router.vlm_pipeline imported successfully")
except Exception as e:
    print(f"[ERROR] aads_ulora_v5.5.router.vlm_pipeline import failed: {e}")

# Test imports
try:
    from aads_ulora_v55.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
    print("[OK] aads_ulora_v55.pipeline.independent_multi_crop_pipeline imported successfully")
except Exception as e:
    print(f"[ERROR] aads_ulora_v5.5.pipeline.independent_multi_crop_pipeline import failed: {e}")

# Test imports
try:
    from aads_ulora_v55.ood.prototypes import PrototypeComputer
    print("[OK] aads_ulora_v55.ood.prototypes imported successfully")
except Exception as e:
    print(f"[ERROR] aads_ulora_v5.5.ood.prototypes import failed: {e}")

print("Import test completed")
