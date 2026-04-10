import json
from pathlib import Path

import pytest

from src.training.services.class_balance import (
    build_class_balance_runtime,
    compute_effective_number_weights,
    resolve_reference_class_counts,
)


def _write_manifest(crop_root: Path, rows: list[dict], *, filename: str = "split_manifest.json") -> None:
    crop_root.mkdir(parents=True, exist_ok=True)
    (crop_root / filename).write_text(
        json.dumps({"classes": rows}, indent=2),
        encoding="utf-8",
    )


def test_resolve_reference_class_counts_prefers_manifest_and_normalizes_names(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "Healthy", "image_count": 240},
            {"class_name": "Disease A", "image_count": 120},
        ],
    )

    resolved = resolve_reference_class_counts(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 10, "disease_a": 10}},
    )

    assert resolved["count_source"] == "manifest:split_manifest.json"
    assert resolved["resolved_class_counts"] == {"healthy": 240, "disease_a": 120}
    assert resolved["missing_classes"] == []


def test_resolve_reference_class_counts_falls_back_to_train_split_without_manifest(tmp_path: Path):
    resolved = resolve_reference_class_counts(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 160, "disease_a": 110}},
    )

    assert resolved["count_source"] == "train_split"
    assert resolved["resolved_class_counts"] == {"healthy": 160, "disease_a": 110}
    assert resolved["missing_classes"] == []


def test_build_class_balance_runtime_activates_and_normalizes_weights(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 260},
            {"class_name": "disease_a", "image_count": 120},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 200, "disease_a": 90}},
    )

    assert runtime["active"] is True
    assert runtime["eligible_classes"] == ["disease_a"]
    assert set(runtime["weights_by_class"].keys()) == {"healthy", "disease_a"}
    assert pytest.approx(sum(runtime["weights_by_class"].values()) / 2.0) == 1.0
    assert runtime["weights_by_class"]["disease_a"] > runtime["weights_by_class"]["healthy"]


def test_build_class_balance_runtime_reports_under_min_classes(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 240},
            {"class_name": "disease_a", "image_count": 54},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 180, "disease_a": 40}},
    )

    assert runtime["active"] is False
    assert runtime["under_min_classes"] == ["disease_a"]
    assert runtime["production_under_min_classes"] == ["disease_a"]
    assert runtime["production_guardrail_bypassed"] is False
    assert runtime["weights_by_class"] == {}


def test_build_class_balance_runtime_records_few_shot_research_bypass(tmp_path: Path):
    crop_root = tmp_path / "tomato"
    _write_manifest(
        crop_root,
        [
            {"class_name": "healthy", "image_count": 12},
            {"class_name": "disease_a", "image_count": 8},
        ],
    )

    runtime = build_class_balance_runtime(
        crop_name="tomato",
        data_dir=tmp_path,
        detected_classes=["healthy", "disease_a"],
        split_class_counts={"train": {"healthy": 9, "disease_a": 6}},
        min_supported_samples=1,
        few_shot_research_mode=True,
    )

    assert runtime["few_shot_research_mode"] is True
    assert runtime["under_min_classes"] == []
    assert runtime["production_under_min_classes"] == ["healthy", "disease_a"]
    assert runtime["production_guardrail_bypassed"] is True
    assert runtime["active"] is True


def test_compute_effective_number_weights_normalizes_to_mean_one():
    weights = compute_effective_number_weights([260, 120, 180])

    assert len(weights) == 3
    assert pytest.approx(sum(weights) / 3.0) == 1.0
