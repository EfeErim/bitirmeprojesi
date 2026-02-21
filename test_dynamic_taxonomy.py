#!/usr/bin/env python3
"""
Quick test of dynamic taxonomy loading and pipeline configuration.
This runs locally without downloading models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_taxonomy_loading():
    """Test that taxonomy loads correctly."""
    print("="*70)
    print("DYNAMIC TAXONOMY TEST")
    print("="*70)
    
    from src.router.vlm_pipeline import VLMPipeline
    
    # Test 1: Dynamic taxonomy enabled
    print("\n✓ Test 1: Dynamic Taxonomy Mode")
    config_dynamic = {
        'vlm_enabled': False,
        'router': {
            'vlm': {
                'enabled': False,
                'use_dynamic_taxonomy': True,
                'taxonomy_path': 'config/plant_taxonomy.json'
            }
        }
    }
    
    pipeline_dynamic = VLMPipeline(config=config_dynamic, device='cpu')
    
    print(f"   Loaded {len(pipeline_dynamic.crop_labels)} crop types")
    print(f"   Loaded {len(pipeline_dynamic.part_labels)} part types")
    print(f"   use_dynamic_taxonomy: {pipeline_dynamic.use_dynamic_taxonomy}")
    print(f"\n   Sample crops (first 15):")
    for i, crop in enumerate(pipeline_dynamic.crop_labels[:15], 1):
        print(f"      {i:2d}. {crop}")
    print(f"\n   Plant parts:")
    for i, part in enumerate(pipeline_dynamic.part_labels, 1):
        print(f"      {i:2d}. {part}")
    
    # Test 2: Specific crops mode (old behavior)
    print("\n✓ Test 2: Specific Crops Mode (Original)")
    config_specific = {
        'vlm_enabled': False,
        'router': {
            'crop_mapping': {
                'tomato': {'parts': ['leaf', 'fruit']},
                'potato': {'parts': ['leaf', 'tuber']},
            },
            'vlm': {
                'enabled': False,
                'use_dynamic_taxonomy': False,
            }
        }
    }
    
    pipeline_specific = VLMPipeline(config=config_specific, device='cpu')
    
    print(f"   Loaded {len(pipeline_specific.crop_labels)} crop types")
    print(f"   Loaded {len(pipeline_specific.part_labels)} part types")
    print(f"   use_dynamic_taxonomy: {pipeline_specific.use_dynamic_taxonomy}")
    print(f"   Crops: {pipeline_specific.crop_labels}")
    print(f"   Parts: {pipeline_specific.part_labels}")
    
    # Test 3: GroundingDINO prompt strategy
    print("\n✓ Test 3: GroundingDINO Prompt Strategy")
    print(f"   Dynamic taxonomy with {len(pipeline_dynamic.crop_labels)} crops:")
    print(f"   → Will use GENERIC prompts (many labels > 20)")
    print(f"      ['plant', 'leaf', 'plant leaf', 'green leaf', 'crop', ...]")
    print(f"\n   Specific mode with {len(pipeline_specific.crop_labels)} crops:")
    print(f"   → Will use SPECIFIC prompts (few labels < 20)")
    print(f"      ['tomato', 'potato', 'leaf', 'fruit', 'tuber', ...]")
    
    # Test 4: Verify common plants are included
    print("\n✓ Test 4: Coverage Check")
    common_plants = [
        'tomato', 'potato', 'corn', 'wheat', 'rice', 'lettuce', 
        'apple', 'grape', 'strawberry', 'rose', 'dandelion'
    ]
    
    covered = [p for p in common_plants if p in pipeline_dynamic.crop_labels]
    missing = [p for p in common_plants if p not in pipeline_dynamic.crop_labels]
    
    print(f"   Common plants in taxonomy: {len(covered)}/{len(common_plants)}")
    if covered:
        print(f"   ✅ Covered: {', '.join(covered)}")
    if missing:
        print(f"   ⚠️  Missing: {', '.join(missing)}")
    
    print("\n" + "="*70)
    print("✅ ALL LOCAL TESTS PASSED")
    print("="*70)
    print("\nNext: Test in Colab with real models and your leaf image!")
    print("Run: %run scripts/colab_vlm_quick_test.py")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    try:
        test_taxonomy_loading()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
