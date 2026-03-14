#!/usr/bin/env python3
"""Validate the maintained notebook support surfaces."""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


@dataclass(frozen=True)
class ValidationCheck:
    result_name: str
    step_id: str
    description: str
    success_message: str
    failure_prefix: str
    callback: Callable[[], None]
    requires_runtime_dependencies: bool = True


def _run_check(check: ValidationCheck, *, leading_newline: bool = False) -> bool:
    prefix = "\n" if leading_newline else ""
    print(f"{prefix}Testing {gate_label(check.step_id, check.description)}...")
    try:
        check.callback()
    except Exception as exc:
        detail = str(exc).strip()
        failure_message = check.failure_prefix if not detail else f"{check.failure_prefix}: {detail}"
        print(f"FAIL {gate_label(check.step_id, failure_message)}")
        return False

    print(f"PASS {gate_label(check.step_id, check.success_message)}")
    return True


def _check_runtime_dependencies() -> None:
    required = (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "PIL",
    )
    missing = []
    for module_name in required:
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        missing_display = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing dependencies: {missing_display}. Install requirements.txt before running this validation."
        )


def test_config_surface() -> None:
    from src.core.config_manager import ConfigurationManager

    cfg = ConfigurationManager(config_dir=str(ROOT / "config"), environment="colab").load_all_configs()
    assert {"training", "router", "colab", "inference"} <= set(cfg.keys())


def test_continual_trainer_imports() -> None:
    from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
    from src.training.session import ContinualTrainingSession
    from src.training.validation import evaluate_model
    from src.workflows.training import TrainingWorkflow

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
    assert TrainingWorkflow is not None
    assert callable(evaluate_model)


def test_quantization_guard() -> None:
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


def test_adapter_surface() -> None:
    from src.adapter.independent_crop_adapter import IndependentCropAdapter

    adapter = IndependentCropAdapter(crop_name="tomato", device="cpu")
    assert hasattr(adapter, "initialize_engine")
    assert hasattr(adapter, "add_classes")
    assert hasattr(adapter, "build_training_session")
    assert hasattr(adapter, "save_adapter")
    assert hasattr(adapter, "load_adapter")


def test_runtime_surface() -> None:
    from src.pipeline.router_adapter_runtime import RouterAdapterRuntime
    from src.workflows.inference import InferenceWorkflow

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
            "inference": {"adapter_root": "models/adapters", "target_size": 224},
        },
        device="cpu",
    )
    assert hasattr(runtime, "load_router")
    assert hasattr(runtime, "load_adapter")
    assert hasattr(runtime, "predict")
    assert InferenceWorkflow is not None


def test_adapter_smoke_notebook_surface() -> None:
    from scripts.colab_adapter_smoke_test import (
        discover_adapter_candidates,
        load_adapter_summary,
        predict_image_folder,
        predict_single_image,
    )

    assert callable(discover_adapter_candidates)
    assert callable(load_adapter_summary)
    assert callable(predict_single_image)
    assert callable(predict_image_folder)


def test_colab_helpers() -> None:
    from scripts.colab_checkpointing import TrainingCheckpointManager
    from scripts.colab_dataset_layout import prepare_runtime_dataset_layout
    from scripts.colab_live_telemetry import ColabLiveTelemetry
    from scripts.colab_repo_bootstrap import (
        export_current_colab_notebook,
        mirror_checkpoint_state_to_repo,
        mirror_path_to_repo,
        push_repo_run_to_github,
    )
    from scripts.evaluate_dataset_layout import evaluate_layout

    assert hasattr(ColabLiveTelemetry, "configure_repo_output_export")
    assert callable(export_current_colab_notebook)
    assert callable(mirror_checkpoint_state_to_repo)
    assert callable(mirror_path_to_repo)
    assert callable(push_repo_run_to_github)
    _ = (TrainingCheckpointManager, prepare_runtime_dataset_layout, ColabLiveTelemetry, evaluate_layout)


CHECKS = (
    ValidationCheck(
        result_name="Runtime Dependencies",
        step_id="ENV",
        description="runtime dependencies",
        success_message="Runtime dependencies available",
        failure_prefix="Missing dependencies",
        callback=_check_runtime_dependencies,
        requires_runtime_dependencies=False,
    ),
    ValidationCheck(
        result_name="Minimal Config",
        step_id="CONFIG",
        description="minimal config load",
        success_message="Configuration loaded successfully",
        failure_prefix="Configuration load failed",
        callback=test_config_surface,
    ),
    ValidationCheck(
        result_name="Continual Trainer",
        step_id="TRAINING",
        description="continual trainer imports",
        success_message="Continual trainer surface imported and validated",
        failure_prefix="Continual trainer test failed",
        callback=test_continual_trainer_imports,
    ),
    ValidationCheck(
        result_name="Quantization Guard",
        step_id="LOW_BIT_GUARD",
        description="4-bit rejection guard",
        success_message="Quantization guard behaves correctly",
        failure_prefix="Quantization guard failed",
        callback=test_quantization_guard,
    ),
    ValidationCheck(
        result_name="Adapter Lifecycle",
        step_id="ADAPTER_API",
        description="adapter lifecycle surface",
        success_message="Adapter lifecycle surface available",
        failure_prefix="Adapter API test failed",
        callback=test_adapter_surface,
    ),
    ValidationCheck(
        result_name="Router Runtime",
        step_id="INFERENCE",
        description="router runtime surface",
        success_message="Router runtime surface available",
        failure_prefix="Router runtime test failed",
        callback=test_runtime_surface,
    ),
    ValidationCheck(
        result_name="Adapter Smoke Notebook",
        step_id="ADAPTER_SMOKE",
        description="adapter smoke-test helper surface",
        success_message="Adapter smoke-test helper surface available",
        failure_prefix="Adapter smoke-test surface failed",
        callback=test_adapter_smoke_notebook_surface,
    ),
    ValidationCheck(
        result_name="Colab Helpers",
        step_id="COLAB",
        description="colab support helpers",
        success_message="Colab helper surfaces imported successfully",
        failure_prefix="Colab helper import failed",
        callback=test_colab_helpers,
    ),
)


def main() -> int:
    print("=" * 60)
    print("AADS v6 Minimal Surface Validation")
    print("=" * 60)

    results = []
    runtime_dependencies_ready = True
    for index, check in enumerate(CHECKS):
        if check.requires_runtime_dependencies and not runtime_dependencies_ready:
            continue
        ok = _run_check(check, leading_newline=index > 1)
        results.append((check.result_name, ok))
        if check.step_id == "ENV":
            runtime_dependencies_ready = ok

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
