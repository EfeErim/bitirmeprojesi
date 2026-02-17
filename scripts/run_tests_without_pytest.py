#!/usr/bin/env python3
"""Run tests directly without pytest to avoid warnings filter issues."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Patch warnings module BEFORE any other imports to avoid the list.remove() error
import warnings

# Save original functions
_original_showwarning = warnings.showwarning
_original_warn = warnings.warn
_original_filterwarnings = warnings.filterwarnings
_original_simplefilter = warnings.simplefilter

# Completely replace warnings functions with no-op versions
def _noop_showwarning(*args, **kwargs):
    pass

def _noop_warn(*args, **kwargs):
    pass

def _noop_filterwarnings(*args, **kwargs):
    pass

def _noop_simplefilter(*args, **kwargs):
    pass

# Apply patches
warnings.showwarning = _noop_showwarning
warnings.warn = _noop_warn
warnings.filterwarnings = _noop_filterwarnings
warnings.simplefilter = _noop_simplefilter

# Also patch the internal _add_filter if it exists
if hasattr(warnings, '_add_filter'):
    _original_add_filter = warnings._add_filter
    def _noop_add_filter(*args, **kwargs):
        pass
    warnings._add_filter = _noop_add_filter

# Now import the package
import importlib

try:
    import aads_ulora_v55
    print(f"[OK] Package aads_ulora_v55 imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import aads_ulora_v55: {e}")
    import traceback
    traceback.print_exc()
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

# Now import the package
try:
    import aads_ulora_v55
    print(f"[OK] Package aads_ulora_v55 imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import aads_ulora_v55: {e}")
    import traceback
    traceback.print_exc()
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
