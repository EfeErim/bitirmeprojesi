#!/usr/bin/env python3
"""
Test script to verify all consolidated configuration files load successfully.
"""

import sys
import json
from pathlib import Path

def test_config_file(filepath, description):
    """Test loading a single configuration file."""
    print()
    print("=" * 60)
    print(f"Testing: {description}")
    print(f"File: {filepath}")
    print("=" * 60)
    
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print("[PASS] Valid JSON syntax")
        
        # Check version
        version = config.get('version', 'N/A')
        print(f"[INFO] Version: {version}")
        
        return True
    except json.JSONDecodeError as e:
        print(f"[FAIL] Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

def main():
    """Run all configuration tests."""
    print("=" * 60)
    print("CONFIGURATION CONSOLIDATION TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test individual config files
    config_files = [
        ("config/base.json", "Base Configuration"),
        ("config/development.json", "Development Environment"),
        ("config/production.json", "Production Environment"),
        ("config/staging.json", "Staging Environment"),
        ("config/test.json", "Test Environment"),
        ("config/router-config.json", "Router Configuration"),
        ("config/ood-config.json", "OOD Configuration"),
        ("config/monitoring-config.json", "Monitoring Configuration")
    ]
    
    for filepath, description in config_files:
        results.append((description, test_config_file(filepath, description)))
    
    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Configuration consolidation successful.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())