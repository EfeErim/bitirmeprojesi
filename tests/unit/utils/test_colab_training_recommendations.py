from pathlib import Path

from scripts.colab_training_recommendations import (
    inspect_runtime_dataset,
    inspect_runtime_hardware,
    recommend_notebook_training_params,
    resolve_effective_notebook_params,
)


def _write_images(root: Path, split_name: str, class_name: str, count: int) -> None:
    target = root / split_name / class_name
    target.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (target / f"img_{index:04d}.jpg").write_bytes(b"test")


def _base_params(**overrides):
    params = {
        "EPOCHS": 12,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 2e-4,
        "LORA_R": 24,
        "LORA_ALPHA": 24,
        "LORA_DROPOUT": 0.1,
        "OOD_FACTOR": 3.0,
        "SURE_SEMANTIC_PERCENTILE": 90.0,
        "SURE_CONFIDENCE_PERCENTILE": 97.0,
        "CONFORMAL_ALPHA": 0.05,
        "CONFORMAL_METHOD": "raps",
        "CONFORMAL_RAPS_LAMBDA": 0.2,
        "CONFORMAL_RAPS_K_REG": 1,
        "BER_ENABLED": False,
        "BER_LAMBDA_OLD": 0.1,
        "BER_LAMBDA_NEW": 0.1,
        "BER_WARMUP_STEPS": 50,
        "WEIGHT_DECAY": 0.01,
        "MIXED_PRECISION": "bf16",
        "GRAD_ACCUM_STEPS": 1,
        "MAX_GRAD_NORM": 1.0,
        "LABEL_SMOOTHING": 0.0,
        "LOSS_NAME": "logitnorm",
        "LOGITNORM_TAU": 1.0,
        "SCHEDULER_NAME": "cosine",
        "SCHEDULER_WARMUP_RATIO": 0.1,
        "SCHEDULER_MIN_LR": 1e-6,
        "EARLY_STOPPING_PATIENCE": 5,
        "EARLY_STOPPING_MIN_DELTA": 0.0,
        "DETERMINISTIC": False,
        "NUM_WORKERS": 8,
        "PREFETCH": 4,
        "PIN_MEMORY": True,
        "USE_CACHE": True,
        "CACHE_TRAIN_SPLIT": True,
        "VALIDATION_EVERY_N_EPOCHS": 2,
        "CHECKPOINT_EVERY_N_STEPS": 250,
        "CHECKPOINT_ON_EXCEPTION": True,
        "STDOUT_BATCH_INTERVAL": 12,
        "FEW_SHOT_RESEARCH_MODE": False,
        "FEW_SHOT_MIN_CLASS_SAMPLES": 1,
        "RESUME_MODE": "fresh",
    }
    params.update(overrides)
    return params


def test_inspect_runtime_dataset_reads_manifest_backed_risk_hints(tmp_path: Path):
    root = tmp_path / "grape__fruit"
    _write_images(root, "continual", "healthy", 3)
    _write_images(root, "continual", "disease_a", 2)
    _write_images(root, "val", "healthy", 1)
    _write_images(root, "val", "disease_a", 1)
    _write_images(root, "test", "healthy", 1)
    _write_images(root, "test", "disease_a", 1)
    _write_images(root, "ood", "unknown", 2)
    (root / "split_manifest.json").write_text(
        """
        {
          "classes": [
            {"class_name": "healthy", "image_count": 240},
            {"class_name": "disease_a", "image_count": 120}
          ],
          "rows": [
            {"synthetic_hint": true, "eval_quality_risk": false, "family_eval_eligible": true, "family_assignment": "continual"},
            {"synthetic_hint": false, "eval_quality_risk": true, "family_eval_eligible": false, "family_assignment": "test"}
          ]
        }
        """,
        encoding="utf-8",
    )

    report = inspect_runtime_dataset(root)

    assert report["manifest_present"] is True
    assert report["split_totals"]["ood"] == 2
    assert report["manifest_class_counts"] == {"healthy": 240, "disease_a": 120}
    assert report["reference_class_counts"] == {"healthy": 240, "disease_a": 120}
    assert report["real_ood_present"] is True
    assert report["noisy_or_high_risk"] is True
    assert report["manifest_row_summary"]["synthetic_hint_count"] == 1
    assert report["manifest_row_summary"]["eval_quality_risk_count"] == 1


def test_inspect_runtime_dataset_accepts_explicit_ood_root(tmp_path: Path):
    root = tmp_path / "grape__fruit"
    external_ood_root = tmp_path / "external_ood"
    _write_images(root, "continual", "healthy", 120)
    _write_images(root, "val", "healthy", 10)
    _write_images(root, "test", "healthy", 10)
    _write_images(external_ood_root, "unknown", "hard_negative", 3)

    report = inspect_runtime_dataset(root, ood_root=external_ood_root)

    assert report["split_totals"]["ood"] == 3
    assert report["ood_root"] == str(external_ood_root.resolve())
    assert report["real_ood_present"] is True
    assert report["split_presence"]["ood"] is True


def test_inspect_runtime_dataset_prefers_reference_image_count_for_augmented_manifests(tmp_path: Path):
    root = tmp_path / "grape__fruit_train_aug"
    _write_images(root, "continual", "healthy", 300)
    _write_images(root, "continual", "powdery_mildew", 138)
    _write_images(root, "val", "healthy", 10)
    _write_images(root, "val", "powdery_mildew", 8)
    _write_images(root, "test", "healthy", 10)
    _write_images(root, "test", "powdery_mildew", 8)
    (root / "split_manifest.json").write_text(
        """
        {
          "classes": [
            {"class_name": "healthy", "image_count": 320, "reference_image_count": 100},
            {"class_name": "powdery_mildew", "image_count": 154, "reference_image_count": 46}
          ]
        }
        """,
        encoding="utf-8",
    )

    report = inspect_runtime_dataset(root)

    assert report["manifest_class_counts"] == {"healthy": 100, "powdery_mildew": 46}
    assert report["under_min_classes"] == ["powdery_mildew"]


def test_inspect_runtime_dataset_counts_grouped_manifest_rows_as_reference(tmp_path: Path):
    root = tmp_path / "grape__fruit"
    _write_images(root, "continual", "healthy", 5)
    _write_images(root, "continual", "powdery_mildew", 5)
    _write_images(root, "val", "healthy", 1)
    _write_images(root, "test", "healthy", 1)
    (root / "split_manifest.json").write_text(
        """
        {
          "rows": [
            {"normalized_class_name": "healthy", "split": "continual"},
            {"normalized_class_name": "healthy", "split": "val"},
            {"normalized_class_name": "healthy", "split": "test"},
            {"normalized_class_name": "healthy", "split": "continual", "generated_offline_augmentation": true},
            {"normalized_class_name": "healthy", "split": "continual", "synthetic_hint": true},
            {"normalized_class_name": "powdery_mildew", "split": "continual"}
          ]
        }
        """,
        encoding="utf-8",
    )

    report = inspect_runtime_dataset(root)

    assert report["manifest_class_counts"] == {"healthy": 3, "powdery_mildew": 1}
    assert report["reference_class_counts"] == {"healthy": 3, "powdery_mildew": 1}


def test_inspect_runtime_dataset_falls_back_without_manifest(tmp_path: Path):
    root = tmp_path / "tomato"
    _write_images(root, "continual", "healthy", 40)
    _write_images(root, "continual", "disease_a", 20)
    _write_images(root, "val", "healthy", 5)
    _write_images(root, "val", "disease_a", 5)
    _write_images(root, "test", "healthy", 5)
    _write_images(root, "test", "disease_a", 5)

    report = inspect_runtime_dataset(root)

    assert report["manifest_present"] is False
    assert report["reference_class_counts"] == {"healthy": 40, "disease_a": 20}
    assert report["real_ood_present"] is False
    assert report["under_min_classes"] == ["disease_a", "healthy"]
    assert any("split_manifest.json was not found" in warning for warning in report["warnings"])


def test_recommend_notebook_training_params_for_small_noisy_dataset():
    dataset_report = {
        "dataset_scale_bucket": "small",
        "class_count": 3,
        "split_totals": {"continual": 900, "val": 180, "test": 180, "ood": 0},
        "real_ood_present": False,
        "noisy_or_high_risk": True,
        "blockers": [],
        "warnings": ["synthetic hints present"],
    }
    hardware_report = {
        "effective_device": "cuda",
        "total_vram_gb": 24.0,
        "cpu_count": 12,
        "warnings": [],
        "strong_gpu": False,
        "memory_constrained": False,
    }

    report = recommend_notebook_training_params(_base_params(BATCH_SIZE=96, LORA_R=24), dataset_report, hardware_report)

    assert report["recommended_params"]["BATCH_SIZE"] == 16
    assert report["recommended_params"]["GRAD_ACCUM_STEPS"] == 2
    assert report["recommended_params"]["EPOCHS"] == 16
    assert report["recommended_params"]["VALIDATION_EVERY_N_EPOCHS"] == 1
    assert report["recommended_params"]["LEARNING_RATE"] == 1e-4
    assert report["recommended_params"]["LORA_R"] == 16
    assert report["recommended_params"]["LORA_ALPHA"] == 16
    assert report["recommended_params"]["LORA_DROPOUT"] == 0.2
    assert report["recommended_params"]["NUM_WORKERS"] == 6
    assert report["recommended_params"]["PREFETCH"] == 4
    assert report["has_changes"] is True


def test_recommend_notebook_training_params_for_large_clean_dataset():
    dataset_report = {
        "dataset_scale_bucket": "large",
        "class_count": 9,
        "split_totals": {"continual": 16000, "val": 2400, "test": 2400, "ood": 200},
        "real_ood_present": True,
        "noisy_or_high_risk": False,
        "blockers": [],
        "warnings": [],
    }
    hardware_report = {
        "effective_device": "cuda",
        "total_vram_gb": 80.0,
        "cpu_count": 16,
        "warnings": [],
        "strong_gpu": True,
        "memory_constrained": False,
    }

    report = recommend_notebook_training_params(_base_params(), dataset_report, hardware_report)

    assert report["recommended_params"]["BATCH_SIZE"] == 96
    assert report["recommended_params"]["GRAD_ACCUM_STEPS"] == 2
    assert report["recommended_params"]["EPOCHS"] == 8
    assert report["recommended_params"]["VALIDATION_EVERY_N_EPOCHS"] == 3
    assert report["recommended_params"]["LORA_R"] == 32
    assert report["recommended_params"]["LORA_ALPHA"] == 32
    assert report["recommended_params"]["LORA_DROPOUT"] == 0.05
    assert report["recommended_params"]["PREFETCH"] == 8
    assert report["recommended_params"]["CACHE_TRAIN_SPLIT"] is False


def test_recommend_notebook_training_params_for_cpu_only_runtime():
    dataset_report = {
        "dataset_scale_bucket": "tiny",
        "class_count": 2,
        "split_totals": {"continual": 120, "val": 30, "test": 30, "ood": 0},
        "real_ood_present": False,
        "noisy_or_high_risk": False,
        "blockers": [],
        "warnings": [],
    }
    hardware_report = {
        "effective_device": "cpu",
        "total_vram_gb": None,
        "cpu_count": 4,
        "warnings": ["CUDA unavailable"],
        "strong_gpu": False,
        "memory_constrained": True,
    }

    report = recommend_notebook_training_params(_base_params(), dataset_report, hardware_report)

    assert report["recommended_params"]["BATCH_SIZE"] == 8
    assert report["recommended_params"]["NUM_WORKERS"] == 0
    assert report["recommended_params"]["PREFETCH"] == 0
    assert report["recommended_params"]["PIN_MEMORY"] is False
    assert report["recommended_params"]["USE_CACHE"] is False
    assert report["recommended_params"]["CACHE_TRAIN_SPLIT"] is False


def test_recommend_notebook_training_params_reports_under_minimum_support():
    dataset_report = {
        "dataset_scale_bucket": "tiny",
        "class_count": 2,
        "split_totals": {"continual": 40, "val": 10, "test": 10, "ood": 0},
        "real_ood_present": False,
        "noisy_or_high_risk": True,
        "blockers": [
            "Supported classes remain below the production minimum of 100 images/class: healthy=40."
        ],
        "warnings": [],
    }
    hardware_report = {
        "effective_device": "cuda",
        "total_vram_gb": 24.0,
        "cpu_count": 8,
        "warnings": [],
        "strong_gpu": False,
        "memory_constrained": False,
    }

    report = recommend_notebook_training_params(_base_params(), dataset_report, hardware_report)

    assert report["blockers"]
    assert report["has_changes"] is True


def test_recommend_notebook_training_params_detects_invalid_ber_loss_combo():
    dataset_report = {
        "dataset_scale_bucket": "medium",
        "class_count": 4,
        "split_totals": {"continual": 4000, "val": 600, "test": 600, "ood": 0},
        "real_ood_present": False,
        "noisy_or_high_risk": False,
        "blockers": [],
        "warnings": [],
    }
    hardware_report = {
        "effective_device": "cuda",
        "total_vram_gb": 24.0,
        "cpu_count": 8,
        "warnings": [],
        "strong_gpu": False,
        "memory_constrained": False,
    }

    report = recommend_notebook_training_params(_base_params(BER_ENABLED=True, LOSS_NAME="logitnorm"), dataset_report, hardware_report)

    assert any("BER_ENABLED requires LOSS_NAME='cross_entropy'" in blocker for blocker in report["blockers"])


def test_resolve_effective_notebook_params_honors_manual_overrides():
    recommendation_report = {
        "recommended_params": _base_params(BATCH_SIZE=16, EPOCHS=16),
    }

    resolved = resolve_effective_notebook_params(
        _base_params(BATCH_SIZE=96, EPOCHS=12, PIN_MEMORY=True),
        recommendation_report,
        {"BATCH_SIZE": "20", "PIN_MEMORY": "false"},
        accepted=True,
    )

    assert resolved["EPOCHS"] == 16
    assert resolved["BATCH_SIZE"] == 20
    assert resolved["PIN_MEMORY"] is False


def test_recommend_notebook_training_params_returns_noop_when_already_aligned():
    dataset_report = {
        "dataset_scale_bucket": "medium",
        "class_count": 4,
        "split_totals": {"continual": 5000, "val": 800, "test": 800, "ood": 100},
        "real_ood_present": True,
        "noisy_or_high_risk": False,
        "blockers": [],
        "warnings": [],
    }
    hardware_report = {
        "effective_device": "cuda",
        "total_vram_gb": 39.0,
        "cpu_count": 16,
        "warnings": [],
        "strong_gpu": False,
        "memory_constrained": False,
    }
    base = _base_params(
        BATCH_SIZE=32,
        GRAD_ACCUM_STEPS=2,
        EPOCHS=12,
        VALIDATION_EVERY_N_EPOCHS=2,
        LEARNING_RATE=2e-4,
        LORA_R=24,
        LORA_ALPHA=24,
        LORA_DROPOUT=0.1,
        NUM_WORKERS=8,
        PREFETCH=4,
        PIN_MEMORY=True,
        USE_CACHE=True,
        CACHE_TRAIN_SPLIT=True,
    )

    report = recommend_notebook_training_params(base, dataset_report, hardware_report)

    assert report["changes"] == {}
    assert report["has_changes"] is False


def test_inspect_runtime_hardware_reports_cpu_runtime():
    report = inspect_runtime_hardware("cpu")

    assert report["requested_device"] == "cpu"
    assert report["effective_device"] == "cpu"
    assert report["cpu_count"] >= 1
