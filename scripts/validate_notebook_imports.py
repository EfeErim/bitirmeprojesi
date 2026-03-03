#!/usr/bin/env python3
"""Validate continual notebook/runtime imports for AADS v6."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        converted = [str(a).encode("ascii", errors="replace").decode("ascii") for a in args]
        builtins.print(*converted, **kwargs)


print = _safe_print


def gate_label(step_id: str, name: str) -> str:
    return f"[{step_id}] {name}"


def test_dataset_imports() -> bool:
    step_id = "DATA_IMPORTS"
    print(f"Testing {gate_label(step_id, 'dataset imports')}...")
    try:
        from src.dataset.colab_datasets import ColabCropDataset, ColabDomainShiftDataset
        _ = (ColabCropDataset, ColabDomainShiftDataset)
        print(f"PASS {gate_label(step_id, 'Dataset classes imported successfully')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Dataset import failed: {exc}')}")
        return False


def test_continual_trainer_imports() -> bool:
    step_id = "CONTINUAL_IMPORT"
    print(f"\nTesting {gate_label(step_id, 'continual trainer imports')}...")
    try:
        from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer

        config = ContinualSDLoRAConfig.from_training_config(
            {
                "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                "quantization": {"mode": "int8_hybrid", "strict_backend": False, "allow_cpu_fallback": True},
                "adapter": {
                    "target_modules_strategy": "all_linear_transformer",
                    "lora_r": 4,
                    "lora_alpha": 8,
                },
                "fusion": {"layers": [2, 5, 8, 11]},
                "device": "cpu",
            }
        )
        trainer = ContinualSDLoRATrainer(config)
        assert hasattr(trainer, "initialize_engine")
        assert hasattr(trainer, "add_classes")
        assert hasattr(trainer, "train_increment")
        assert hasattr(trainer, "save_adapter")
        assert hasattr(trainer, "load_adapter")

        print(f"PASS {gate_label(step_id, 'Continual trainer surface imported and validated')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Continual trainer test failed: {exc}')}")
        return False


def test_quantization_guard() -> bool:
    step_id = "INT8_GUARD"
    print(f"\nTesting {gate_label(step_id, '4-bit rejection guard')}...")
    try:
        from src.training.quantization import assert_no_prohibited_4bit_flags

        valid_payload = {
            "training": {
                "continual": {
                    "quantization": {"mode": "int8_hybrid"}
                }
            }
        }
        assert_no_prohibited_4bit_flags(valid_payload)

        rejected = False
        try:
            forbidden_key = "load_in_" + "4bit"
            assert_no_prohibited_4bit_flags({forbidden_key: True})
        except ValueError:
            rejected = True

        assert rejected, "4-bit payload was expected to be rejected"
        print(f"PASS {gate_label(step_id, 'Quantization guard behaves correctly')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Quantization guard failed: {exc}')}")
        return False


def test_adapter_surface() -> bool:
    step_id = "ADAPTER_API"
    print(f"\nTesting {gate_label(step_id, 'adapter lifecycle surface')}...")
    try:
        from src.adapter.independent_crop_adapter import IndependentCropAdapter

        adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
        assert hasattr(adapter, "initialize_engine")
        assert hasattr(adapter, "add_classes")
        assert hasattr(adapter, "train_increment")
        assert hasattr(adapter, "save_adapter")
        assert hasattr(adapter, "load_adapter")

        print(f"PASS {gate_label(step_id, 'Adapter lifecycle surface available')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Adapter API test failed: {exc}')}")
        return False


def main() -> int:
    print("=" * 60)
    print("AADS v6 Notebook Import Validation")
    print("=" * 60)

    results = [
        ("Dataset Imports", test_dataset_imports()),
        ("Continual Trainer", test_continual_trainer_imports()),
        ("Quantization Guard", test_quantization_guard()),
        ("Adapter Lifecycle", test_adapter_surface()),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

