#!/usr/bin/env python3
"""Run tests directly without pytest to avoid warnings filter issues."""

import sys
import os
import importlib.util
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the package
try:
    import aads_ulora_v55
    print(f"[OK] Package aads_ulora_v55 imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import aads_ulora_v55: {e}")
    sys.exit(1)

# Test imports from the package
tests_to_run = [
    ('aads_ulora_v55.utils.data_loader', 'CropDataset'),
    ('aads_ulora_v55.router.vlm_pipeline', 'VLMPipeline'),
    ('aads_ulora_v55.pipeline.independent_multi_crop_pipeline', 'IndependentMultiCropPipeline'),
    ('aads_ulora_v55.ood.prototypes', 'PrototypeComputer'),
]

print("\nRunning import tests...")
all_passed = True

for module_name, class_name in tests_to_run:
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        print(f"[PASS] {module_name}.{class_name} can be imported")
    except Exception as e:
        print(f"[FAIL] {module_name}.{class_name} import failed: {e}")
        all_passed = False

if all_passed:
    print("\nAll import tests passed!")
    sys.exit(0)
else:
    print("\nSome tests failed!")
    sys.exit(1)
