#!/usr/bin/env python3
"""Validate the two supported surfaces: Colab training helpers and router inference runtime."""

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


def test_config_surface() -> bool:
    step_id = "CONFIG"
    print(f"Testing {gate_label(step_id, 'minimal config load')}...")
    try:
        from src.core.config_manager import ConfigurationManager

        cfg = ConfigurationManager(config_dir=str(ROOT / "config"), environment="colab").load_all_configs()
        assert {"training", "router", "ood", "colab", "inference"} <= set(cfg.keys())
        print(f"PASS {gate_label(step_id, 'Configuration loaded successfully')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Configuration load failed: {exc}')}")
        return False


def test_continual_trainer_imports() -> bool:
    step_id = "TRAINING"
    print(f"\nTesting {gate_label(step_id, 'continual trainer imports')}...")
    try:
        from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
        from src.training.session import ContinualTrainingSession
        from src.training.validation import evaluate_model

        config = ContinualSDLoRAConfig.from_training_config(
            {
                "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                "adapter": {
                    "target_modules_strategy": "all_linear_transformer",
                    "lora_r": 4,
                    "lora_alpha": 8,
                },
                "fusion": {"layers": [2, 5, 8, 11]},
                "ood": {"threshold_factor": 2.0},
                "device": "cpu",
            }
        )
        trainer = ContinualSDLoRATrainer(config)
        assert hasattr(trainer, "initialize_engine")
        assert hasattr(trainer, "add_classes")
        assert hasattr(trainer, "train_batch")
        assert hasattr(trainer, "snapshot_training_state")
        assert hasattr(trainer, "restore_training_state")
        assert hasattr(trainer, "save_adapter")
        assert hasattr(trainer, "load_adapter")
        assert ContinualTrainingSession is not None
        assert callable(evaluate_model)

        print(f"PASS {gate_label(step_id, 'Continual trainer surface imported and validated')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Continual trainer test failed: {exc}')}")
        return False


def test_quantization_guard() -> bool:
    step_id = "LOW_BIT_GUARD"
    print(f"\nTesting {gate_label(step_id, '4-bit rejection guard')}...")
    try:
        from src.training.quantization import assert_no_prohibited_4bit_flags

        valid_payload = {
            "training": {
                "continual": {
                    "adapter": {"target_modules_strategy": "all_linear_transformer"}
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
        assert hasattr(adapter, "build_training_session")
        assert hasattr(adapter, "save_adapter")
        assert hasattr(adapter, "load_adapter")

        print(f"PASS {gate_label(step_id, 'Adapter lifecycle surface available')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Adapter API test failed: {exc}')}")
        return False


def test_runtime_surface() -> bool:
    step_id = "INFERENCE"
    print(f"\nTesting {gate_label(step_id, 'router runtime surface')}...")
    try:
        from src.pipeline.router_adapter_runtime import RouterAdapterRuntime

        runtime = RouterAdapterRuntime(
            config={
                "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
                "training": {
                    "continual": {
                        "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                        "adapter": {"target_modules_strategy": "all_linear_transformer"},
                        "fusion": {"layers": [2, 5, 8, 11]},
                        "ood": {"threshold_factor": 2.0},
                    }
                },
                "ood": {"threshold_factor": 2.0},
                "inference": {"adapter_root": "models/adapters", "target_size": 224},
            },
            device="cpu",
        )
        assert hasattr(runtime, "load_router")
        assert hasattr(runtime, "load_adapter")
        assert hasattr(runtime, "predict")
        print(f"PASS {gate_label(step_id, 'Router runtime surface available')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Router runtime test failed: {exc}')}")
        return False


def test_colab_helpers() -> bool:
    step_id = "COLAB"
    print(f"\nTesting {gate_label(step_id, 'colab support helpers')}...")
    try:
        from scripts.colab_checkpointing import TrainingCheckpointManager
        from scripts.colab_dataset_layout import prepare_runtime_dataset_layout
        from scripts.colab_live_telemetry import ColabLiveTelemetry
        from scripts.evaluate_dataset_layout import evaluate_layout

        _ = (TrainingCheckpointManager, prepare_runtime_dataset_layout, ColabLiveTelemetry, evaluate_layout)
        print(f"PASS {gate_label(step_id, 'Colab helper surfaces imported successfully')}")
        return True
    except Exception as exc:
        print(f"FAIL {gate_label(step_id, f'Colab helper import failed: {exc}')}")
        return False


def main() -> int:
    print("=" * 60)
    print("AADS v6 Minimal Surface Validation")
    print("=" * 60)

    results = [
        ("Minimal Config", test_config_surface()),
        ("Continual Trainer", test_continual_trainer_imports()),
        ("Quantization Guard", test_quantization_guard()),
        ("Adapter Lifecycle", test_adapter_surface()),
        ("Router Runtime", test_runtime_surface()),
        ("Colab Helpers", test_colab_helpers()),
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
