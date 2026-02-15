#!/usr/bin/env python3
"""
Simple test script to verify imports work correctly."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

# Test imports
try:
    from utils.data_loader import CropDataset
    print("✓ utils.data_loader imported successfully")
except Exception as e:
    print(f"✗ utils.data_loader import failed: {e}")

# Test imports
try:
    from router.simple_crop_router import SimpleCropRouter
    print("✓ router.simple_crop_router imported successfully")
except Exception as e:
    print(f"✗ router.simple_crop_router import failed: {e}")

# Test imports
try:
    from pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
    print("✓ pipeline.independent_multi_crop_pipeline imported successfully")
except Exception as e:
    print(f"✗ pipeline.independent_multi_crop_pipeline import failed: {e}")

# Test imports
try:
    from ood.prototypes import PrototypeComputer
    print("✓ ood.prototypes imported successfully")
except Exception as e:
    print(f"✗ ood.prototypes import failed: {e}")

print("Import test completed")