#!/usr/bin/env python3
"""
Simple test script to verify imports work correctly."""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing imports...")

# Test imports
try:
    from src.utils.data_loader import CropDataset
    print("[OK] src.utils.data_loader imported successfully")
except Exception as e:
    print(f"[ERROR] src.utils.data_loader import failed: {e}")

# Test imports
try:
    from src.router.vlm_pipeline import VLMPipeline
    print("[OK] src.router.vlm_pipeline imported successfully")
except Exception as e:
    print(f"[ERROR] src.router.vlm_pipeline import failed: {e}")

# Test imports
try:
    from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
    print("[OK] src.pipeline.independent_multi_crop_pipeline imported successfully")
except Exception as e:
    print(f"[ERROR] src.pipeline.independent_multi_crop_pipeline import failed: {e}")

# Test imports
try:
    from src.ood.prototypes import PrototypeComputer
    print("[OK] src.ood.prototypes imported successfully")
except Exception as e:
    print(f"[ERROR] src.ood.prototypes import failed: {e}")

print("Import test completed")