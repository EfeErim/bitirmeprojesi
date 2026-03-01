#!/usr/bin/env python3
"""Quick test of dynamic taxonomy loading and pipeline configuration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_taxonomy_loading():
    print('=' * 70)
    print('DYNAMIC TAXONOMY TEST')
    print('=' * 70)

    from src.router.vlm_pipeline import VLMPipeline

    print('\n[OK] Test 1: Dynamic Taxonomy Mode')
    config_dynamic = {
        'vlm_enabled': False,
        'router': {
            'vlm': {
                'enabled': False,
                'use_dynamic_taxonomy': True,
                'taxonomy_path': 'config/plant_taxonomy.json',
            }
        },
    }
    pipeline_dynamic = VLMPipeline(config=config_dynamic, device='cpu')
    print(f"   Loaded {len(pipeline_dynamic.crop_labels)} crop types")
    print(f"   Loaded {len(pipeline_dynamic.part_labels)} part types")

    print('\n[OK] Test 2: Specific Crops Mode')
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
            },
        },
    }
    pipeline_specific = VLMPipeline(config=config_specific, device='cpu')
    print(f"   Loaded {len(pipeline_specific.crop_labels)} crop types")
    print(f"   Loaded {len(pipeline_specific.part_labels)} part types")

    assert len(pipeline_dynamic.crop_labels) > 0
    assert len(pipeline_dynamic.part_labels) > 0

    print('\n[OK] All dynamic taxonomy checks passed')


if __name__ == '__main__':
    try:
        test_taxonomy_loading()
    except Exception as exc:
        print(f"\n[FAIL] TEST FAILED: {exc}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
