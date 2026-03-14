"""Readiness and authoritative-split helpers for the training workflow."""

from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")


def has_metric_gate(candidate: Dict[str, Any]) -> bool:
    return isinstance(candidate, dict) and isinstance(candidate.get("metric_gate"), dict)


def has_evaluation_labels(candidate: Any) -> bool:
    return candidate is not None and bool(list(getattr(candidate, "y_true", []) or []))


def select_authoritative_value(
    validation_value: T,
    test_value: T,
    *,
    calibration_split_name: str,
    is_present: Callable[[T], bool],
    empty_value: T,
) -> tuple[str, T]:
    if is_present(test_value):
        return "test", test_value
    if calibration_split_name != "val" and is_present(validation_value):
        return "val", validation_value
    return "", empty_value


def select_authoritative_artifacts(
    validation_artifacts: Dict[str, Any],
    test_artifacts: Dict[str, Any],
    *,
    calibration_split_name: str,
) -> tuple[str, Dict[str, Any]]:
    return select_authoritative_value(
        validation_artifacts,
        test_artifacts,
        calibration_split_name=calibration_split_name,
        is_present=has_metric_gate,
        empty_value={},
    )


def select_authoritative_evaluation(
    validation_evaluation: Any,
    test_evaluation: Any,
    *,
    calibration_split_name: str,
) -> tuple[str, Any]:
    return select_authoritative_value(
        validation_evaluation,
        test_evaluation,
        calibration_split_name=calibration_split_name,
        is_present=has_evaluation_labels,
        empty_value=None,
    )


def record_primary_score_selection(
    ood_calibration: Dict[str, Any],
    *,
    requested_primary_score_method: str,
    selected_primary_score_method: str,
    selection_source: str,
) -> None:
    calibration = ood_calibration.get("ood_calibration")
    if not isinstance(calibration, dict):
        return
    calibration["requested_primary_score_method"] = requested_primary_score_method
    calibration["primary_score_method"] = selected_primary_score_method
    calibration["selection_source"] = selection_source


def record_adapter_export_metadata(
    adapter: Any,
    *,
    ood_calibration: Dict[str, Any],
    calibration_split_name: str,
    calibration_loader_size: int,
    authoritative_split: str,
    ood_evidence_source: str,
    requested_primary_score_method: str,
    selected_primary_score_method: str,
    selection_source: str,
    best_state_restored: bool,
) -> None:
    setter = getattr(adapter, "set_export_metadata", None)
    if not callable(setter):
        return

    calibration_metadata = (
        dict(ood_calibration.get("ood_calibration", {}))
        if isinstance(ood_calibration.get("ood_calibration"), dict)
        else {}
    )
    calibration_metadata.update(
        {
            "source_split": str(calibration_split_name),
            "source_loader_size": int(calibration_loader_size),
            "authoritative_classification_split": str(authoritative_split or ""),
            "ood_evidence_source": str(ood_evidence_source or ""),
            "requested_primary_score_method": str(requested_primary_score_method),
            "primary_score_method": str(selected_primary_score_method),
            "selection_source": str(selection_source),
        }
    )
    setter(
        ood_calibration=calibration_metadata,
        adapter_runtime={"best_state_restored": bool(best_state_restored)},
    )


def build_production_readiness_context(
    *,
    run_id: str,
    crop_name: str,
    loader_sizes: Dict[str, int],
    loader_batch_counts: Dict[str, int],
    split_class_counts: Dict[str, Dict[str, int]],
    calibration_split_name: str,
    best_state_restored: bool,
    classification_split: str,
    requested_primary_score_method: str,
    selected_primary_score_method: str,
    selection_source: str,
    ood_benchmark: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "crop_name": crop_name,
        "loader_sizes": loader_sizes,
        "loader_batch_counts": loader_batch_counts,
        "split_class_counts": split_class_counts,
        "calibration_split_name": calibration_split_name,
        "best_state_restored": bool(best_state_restored),
        "classification_split": classification_split,
        "ood_requested_primary_score_method": requested_primary_score_method,
        "ood_primary_score_method": selected_primary_score_method,
        "ood_primary_score_selection_source": selection_source,
        "ood_benchmark_status": ood_benchmark.get("status"),
        "ood_benchmark_passed": ood_benchmark.get("passed"),
    }


def build_training_summary_payload(
    *,
    run_id: str,
    crop_name: str,
    detected_classes: list[str],
    loader_sizes: Dict[str, int],
    loader_batch_counts: Dict[str, int],
    split_class_counts: Dict[str, Dict[str, int]],
    adapter_dir: str,
    artifact_dir: str,
    checkpoint_records: list[Dict[str, Any]],
    ood_calibration: Dict[str, Any],
    history_payload: Dict[str, Any],
    calibration_split_name: str,
    ood_evidence_source: str,
    ood_benchmark: Dict[str, Any],
    production_readiness: Dict[str, Any],
    best_state_restored: bool,
    requested_primary_score_method: str,
    selected_primary_score_method: str,
    primary_score_selection_source: str,
    final_metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "crop_name": crop_name,
        "class_names": list(detected_classes),
        "class_count": len(detected_classes),
        "loader_sizes": loader_sizes,
        "loader_batch_counts": loader_batch_counts,
        "split_class_counts": split_class_counts,
        "adapter_dir": str(adapter_dir),
        "artifact_dir": str(artifact_dir),
        "checkpoint_count": len(checkpoint_records),
        "ood_calibration": dict(ood_calibration),
        "calibration_split_name": str(calibration_split_name),
        "stopped_early": bool(history_payload.get("stopped_early", False)),
        "global_step": int(history_payload.get("global_step", 0)),
        "optimizer_steps": int(history_payload.get("optimizer_steps", 0)),
        "best_metric_name": str(history_payload.get("best_metric_name", "")),
        "best_metric_value": history_payload.get("best_metric_value"),
        "best_epoch": int(history_payload.get("best_epoch", 0)),
        "best_state_restored": bool(best_state_restored),
        "ood_requested_primary_score_method": str(requested_primary_score_method),
        "ood_primary_score_method": str(selected_primary_score_method),
        "ood_primary_score_selection_source": str(primary_score_selection_source),
        "ood_evidence_source": str(ood_evidence_source or ""),
        "ood_benchmark": dict(ood_benchmark),
        "production_readiness": dict(production_readiness),
        "final_metrics": dict(final_metrics),
    }
