#!/usr/bin/env python3
"""
Verification script for Dinov3 version management system
Tests version switching, backup/restore, and integrity checks
"""

import sys
import os
from pathlib import Path

# Add version_management to path
sys.path.insert(0, str(Path(__file__).parent / "version_management"))

from backup import VersionManager

def test_version_management():
    """Test all version management functionality."""
    print("=" * 60)
    print("AADS-ULoRA Version Management Verification")
    print("=" * 60)
    
    # Initialize version manager
    vm = VersionManager()
    
    # Test 1: List all versions
    print("\n1. Testing version listing...")
    versions = vm.list_versions()
    print(f"   Found {len(versions)} versions:")
    for version, info in sorted(versions.items()):
        manifest_status = "✓" if info['manifest'] else "✗"
        print(f"   - {version:20} | {manifest_status} | {info['file_count']:5} files")
    
    # Test 2: Get current version
    print("\n2. Testing current version detection...")
    current = vm.get_current_version()
    if current:
        print(f"   Current active version: {current}")
    else:
        print("   WARNING: Could not detect current version")
    
    # Test 3: Verify version integrity
    print("\n3. Testing version integrity verification...")
    for version in versions.keys():
        success, msg = vm.verify_backup(version)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"   {version}: {status} - {msg}")
    
    # Test 4: Test version switching (dry run)
    print("\n4. Testing version switching (dry run)...")
    if "v5.5.4-dinov3" in versions:
        success, msg = vm.switch_active_version("v5.5.4-dinov3")
        if success:
            print(f"   ✓ Successfully switched to v5.5.4-dinov3")
            print(f"   Message: {msg}")
        else:
            print(f"   ✗ Failed to switch to v5.5.4-dinov3: {msg}")
    else:
        print("   WARNING: v5.5.4-dinov3 not found in versions")
    
    # Test 5: Verify current version after switch
    print("\n5. Verifying current version after switch...")
    current_after = vm.get_current_version()
    if current_after == "v5.5.4-dinov3":
        print(f"   ✓ Current version correctly set to: {current_after}")
    else:
        print(f"   ✗ Current version is: {current_after} (expected: v5.5.4-dinov3)")
    
    # Test 6: Check critical configuration files
    print("\n6. Checking critical configuration files...")
    critical_files = [
        "config/adapter_spec_v55.json",
        "src/adapter/independent_crop_adapter.py",
        "src/pipeline/independent_multi_crop_pipeline.py",
        "requirements.txt",
        "setup.py"
    ]
    
    for file_path in critical_files:
        full_path = vm.current_dir / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} - MISSING")
    
    # Test 7: Verify Dinov3 integration in config
    print("\n7. Verifying Dinov3 integration...")
    config_path = vm.current_dir / "config/adapter_spec_v55.json"
    if config_path.exists():
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            crop_router_model = config.get("crop_router", {}).get("model_name", "")
            per_crop_model = config.get("per_crop", {}).get("model_name", "")
            
            if "dinov3" in crop_router_model.lower():
                print(f"   ✓ Crop router model: {crop_router_model}")
            else:
                print(f"   ✗ Crop router model: {crop_router_model} (expected Dinov3)")
            
            if "dinov3" in per_crop_model.lower():
                print(f"   ✓ Per-crop model: {per_crop_model}")
            else:
                print(f"   ✗ Per-crop model: {per_crop_model} (expected Dinov3)")
        except Exception as e:
            print(f"   ✗ Error reading config: {e}")
    else:
        print("   ✗ Config file not found")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)
    
    # Summary
    print("\nSUMMARY:")
    print(f"- Total versions: {len(versions)}")
    print(f"- Current version: {current}")
    print(f"- Active directory: {vm.current_dir}")
    print(f"- Versions directory: {vm.versions_dir}")
    print(f"- Backups directory: {vm.backup_dir}")
    
    return True

if __name__ == "__main__":
    try:
        test_version_management()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)