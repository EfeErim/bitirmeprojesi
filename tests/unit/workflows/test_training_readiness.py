import json

import pytest

from src.workflows.training_readiness import (
    build_production_readiness_context,
    build_provenance_slice_breakdown,
    record_adapter_export_metadata,
    select_authoritative_artifacts,
    select_authoritative_evaluation,
    summarize_provenance_slice_breakdown,
)


class _FakeEvaluation:
    def __init__(self, y_true, *, prediction_rows=None):
        self.y_true = list(y_true)
        self.prediction_rows = list(prediction_rows or [])


class _RecordingAdapter:
    def __init__(self):
        self.calls = []

    def set_export_metadata(self, *, ood_calibration=None, adapter_runtime=None):
        self.calls.append(
            {
                "ood_calibration": dict(ood_calibration or {}),
                "adapter_runtime": dict(adapter_runtime or {}),
            }
        )


def test_select_authoritative_artifacts_prefers_test_split():
    split_name, artifacts = select_authoritative_artifacts(
        {"metric_gate": {"passed": True}},
        {"metric_gate": {"passed": False}},
        calibration_split_name="val",
    )

    assert split_name == "test"
    assert artifacts["metric_gate"]["passed"] is False


def test_select_authoritative_evaluation_skips_val_when_used_for_calibration():
    split_name, evaluation = select_authoritative_evaluation(
        _FakeEvaluation([0, 1]),
        None,
        calibration_split_name="val",
    )

    assert split_name == ""
    assert evaluation is None


def test_record_adapter_export_metadata_passes_structured_provenance():
    adapter = _RecordingAdapter()

    record_adapter_export_metadata(
        adapter,
        ood_calibration={"ood_calibration": {"version": 3}},
        calibration_split_name="val",
        calibration_loader_size=12,
        authoritative_split="test",
        ood_evidence_source="real_ood_split",
        requested_primary_score_method="auto",
        selected_primary_score_method="energy",
        selection_source="real_ood_split",
        best_state_restored=True,
    )

    assert adapter.calls == [
        {
            "ood_calibration": {
                "version": 3,
                "source_split": "val",
                "source_loader_size": 12,
                "authoritative_classification_split": "test",
                "ood_evidence_source": "real_ood_split",
                "requested_primary_score_method": "auto",
                "primary_score_method": "energy",
                "selection_source": "real_ood_split",
            },
            "adapter_runtime": {"best_state_restored": True},
        }
    ]


def test_build_production_readiness_context_tracks_selection_fields():
    payload = build_production_readiness_context(
        run_id="run_1",
        crop_name="tomato",
        loader_sizes={"train": 10},
        loader_batch_counts={"train": 2},
        split_class_counts={"train": {"healthy": 10}},
        calibration_split_name="val",
        best_state_restored=True,
        classification_split="test",
        requested_primary_score_method="auto",
        selected_primary_score_method="knn",
        selection_source="held_out_benchmark",
        ood_benchmark={"status": "completed", "passed": True},
        provenance_summary={"available": True, "reported_dimension_count": 1},
        provenance_warnings=["1 unmatched row"],
        ood_method_comparison={"selected_primary_score_method": "knn"},
    )

    assert payload["classification_split"] == "test"
    assert payload["ood_requested_primary_score_method"] == "auto"
    assert payload["ood_primary_score_method"] == "knn"
    assert payload["ood_primary_score_selection_source"] == "held_out_benchmark"
    assert payload["provenance_summary"]["available"] is True
    assert payload["provenance_warnings"] == ["1 unmatched row"]
    assert payload["ood_method_comparison"]["selected_primary_score_method"] == "knn"


def test_build_provenance_slice_breakdown_reports_populated_dimensions(tmp_path):
    crop_root = tmp_path / "runtime_data" / "tomato"
    crop_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    prediction_rows = []
    set_b_predictions = [0, 0, 0, 1, 1]
    for index in range(10):
        class_name = "healthy" if index % 2 == 0 else "disease_a"
        runtime_relative_path = f"test/{class_name}/img_{index}.jpg"
        manifest_rows.append(
            {
                "split": "test",
                "raw_class_name": class_name,
                "normalized_class_name": class_name,
                "relative_path": f"{class_name}/img_{index}.jpg",
                "runtime_relative_path": runtime_relative_path,
                "source_dataset": "set_a" if index < 5 else "set_b",
                "source_subset": "subset_small" if index < 4 else "subset_large",
                "capture_group_id": f"group_{index // 5}",
                "domain_tag": "field" if index < 5 else "lab",
                "source_hint": "original",
            }
        )
        true_index = index % 2
        pred_index = true_index if index < 5 else set_b_predictions[index - 5]
        prediction_rows.append(
            {
                "sample_origin": "in_distribution",
                "split_name": "test",
                "image_path": str(crop_root / runtime_relative_path),
                "true_index": true_index,
                "pred_index": pred_index,
            }
        )

    (crop_root / "split_manifest.json").write_text(
        json.dumps({"rows": manifest_rows, "provenance_manifest": {"warnings": ["1 unmatched row"]}}, indent=2),
        encoding="utf-8",
    )

    breakdown = build_provenance_slice_breakdown(
        crop_root=crop_root,
        classification_split="test",
        authoritative_evaluation=_FakeEvaluation(
            [row["true_index"] for row in prediction_rows],
            prediction_rows=prediction_rows,
        ),
    )

    assert breakdown["available"] is True
    assert breakdown["matched_sample_count"] == 10
    assert breakdown["unmatched_sample_count"] == 0
    assert "1 unmatched row" in breakdown["warnings"]
    assert breakdown["dimensions"]["source_dataset"]["reported_slice_count"] == 2
    assert breakdown["dimensions"]["source_dataset"]["worst_slices"]["accuracy"]["name"] == "set_b"
    assert breakdown["dimensions"]["source_subset"]["reported_slice_count"] == 1
    assert breakdown["dimensions"]["source_subset"]["skipped_slice_count"] == 1

    summary = summarize_provenance_slice_breakdown(breakdown)

    assert summary["available"] is True
    assert summary["matched_sample_count"] == 10
    assert "source_dataset" in summary["reported_dimensions"]


def test_build_provenance_slice_breakdown_uses_authoritative_split_for_slice_deltas(tmp_path):
    crop_root = tmp_path / "runtime_data" / "tomato"
    crop_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    prediction_rows = []
    for index in range(10):
        class_name = "healthy" if index % 2 == 0 else "disease_a"
        runtime_relative_path = f"test/{class_name}/img_{index}.jpg"
        manifest_rows.append(
            {
                "split": "test",
                "raw_class_name": class_name,
                "normalized_class_name": class_name,
                "relative_path": f"{class_name}/img_{index}.jpg",
                "runtime_relative_path": runtime_relative_path,
                "source_dataset": "set_a" if index < 5 else "",
                "source_hint": "original",
            }
        )
        true_index = index % 2
        pred_index = true_index if index < 8 else 0
        prediction_rows.append(
            {
                "sample_origin": "in_distribution",
                "split_name": "test",
                "image_path": str(crop_root / runtime_relative_path),
                "true_index": true_index,
                "pred_index": pred_index,
            }
        )

    (crop_root / "split_manifest.json").write_text(
        json.dumps({"rows": manifest_rows}, indent=2),
        encoding="utf-8",
    )

    breakdown = build_provenance_slice_breakdown(
        crop_root=crop_root,
        classification_split="test",
        authoritative_evaluation=_FakeEvaluation(
            [row["true_index"] for row in prediction_rows],
            prediction_rows=prediction_rows,
        ),
    )

    source_dataset = breakdown["dimensions"]["source_dataset"]
    assert source_dataset["reported_slice_count"] == 1
    assert source_dataset["slices"][0]["name"] == "set_a"
    assert source_dataset["slices"][0]["deltas"]["accuracy"] == pytest.approx(0.1)
