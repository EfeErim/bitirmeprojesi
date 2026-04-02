"""Readiness and authoritative-split helpers for the training workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, TypeVar

from src.shared.json_utils import read_json
from src.training.services.metrics import compute_plan_metrics

T = TypeVar("T")

_PROVENANCE_DIMENSIONS = (
    "source_dataset",
    "source_subset",
    "capture_group_id",
    "domain_tag",
    "source_hint",
)
_MIN_PROVENANCE_SLICE_SAMPLES = 5
_SLICE_METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1")


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


def _normalize_runtime_relative_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    return "/".join(part for part in text.split("/") if part and part != ".")


def _infer_runtime_relative_path_from_manifest_row(row: Dict[str, Any]) -> str:
    runtime_relative_path = _normalize_runtime_relative_path(row.get("runtime_relative_path"))
    if runtime_relative_path:
        return runtime_relative_path
    split_name = str(row.get("split", "") or "").strip()
    class_name = str(row.get("normalized_class_name", "") or "").strip()
    relative_path = Path(str(row.get("relative_path", "") or ""))
    raw_class_name = str(row.get("raw_class_name", "") or "").strip()
    if not split_name or not class_name or not relative_path.parts:
        return ""
    try:
        destination_relative = relative_path.relative_to(raw_class_name)
    except Exception:
        destination_relative = Path(relative_path.name)
    destination_text = _normalize_runtime_relative_path(destination_relative.as_posix())
    if not destination_text:
        return ""
    return f"{split_name}/{class_name}/{destination_text}"


def _infer_runtime_relative_path_from_prediction(image_path: Any, *, classification_split: str) -> str:
    normalized_path = _normalize_runtime_relative_path(image_path)
    if not normalized_path:
        return ""
    parts = normalized_path.split("/")
    for index, part in enumerate(parts):
        if part.lower() == str(classification_split or "").lower():
            return "/".join(parts[index:])
    return ""


def _to_metric_triplet(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {metric_name: metrics.get(metric_name) for metric_name in _SLICE_METRIC_NAMES}


def _slice_metric_payload(name: str, rows: list[Dict[str, Any]], *, pooled_metrics: Dict[str, Any]) -> Dict[str, Any]:
    y_true = [int(row["true_index"]) for row in rows]
    y_pred = [int(row["pred_index"]) for row in rows]
    metrics = _to_metric_triplet(compute_plan_metrics(y_true=y_true, y_pred=y_pred))
    deltas = {
        metric_name: (
            None
            if metrics.get(metric_name) is None or pooled_metrics.get(metric_name) is None
            else float(metrics[metric_name]) - float(pooled_metrics[metric_name])
        )
        for metric_name in _SLICE_METRIC_NAMES
    }
    return {
        "name": str(name),
        "sample_count": int(len(rows)),
        "metrics": metrics,
        "deltas": deltas,
    }


def build_provenance_slice_breakdown(
    *,
    crop_root: Path,
    classification_split: str,
    authoritative_evaluation: Any,
    min_reported_slice_samples: int = _MIN_PROVENANCE_SLICE_SAMPLES,
) -> Dict[str, Any]:
    manifest_path = Path(crop_root) / "split_manifest.json"
    payload: Dict[str, Any] = {
        "schema_version": "v1_provenance_slice_breakdown",
        "report_only": True,
        "available": False,
        "classification_split": str(classification_split or ""),
        "manifest_path": str(manifest_path),
        "authoritative_sample_count": 0,
        "matched_sample_count": 0,
        "unmatched_sample_count": 0,
        "warnings": [],
        "dimensions": {},
        "min_reported_slice_samples": int(max(1, min_reported_slice_samples)),
    }
    warnings: list[str] = []

    if not classification_split:
        warnings.append("Authoritative classification split was not resolved; provenance analysis skipped.")
        payload["warnings"] = warnings
        return payload
    if authoritative_evaluation is None:
        warnings.append("Authoritative evaluation payload was missing; provenance analysis skipped.")
        payload["warnings"] = warnings
        return payload
    if not manifest_path.exists():
        warnings.append("Runtime split_manifest.json was not found; provenance analysis skipped.")
        payload["warnings"] = warnings
        return payload

    manifest = read_json(manifest_path, default={}, expect_type=dict)
    manifest_rows = list(manifest.get("rows", [])) if isinstance(manifest, dict) else []
    provenance_manifest = dict(manifest.get("provenance_manifest", {})) if isinstance(manifest, dict) else {}
    warnings.extend(str(item) for item in list(provenance_manifest.get("warnings", [])) if str(item).strip())

    split_rows = [
        dict(row)
        for row in manifest_rows
        if isinstance(row, dict) and str(row.get("split", "") or "").strip() == str(classification_split)
    ]
    if not split_rows:
        warnings.append(f"Runtime manifest did not contain rows for authoritative split '{classification_split}'.")
        payload["warnings"] = warnings
        return payload

    metadata_available = any(
        any(str(row.get(field_name, "") or "").strip() for field_name in _PROVENANCE_DIMENSIONS)
        for row in split_rows
    )
    if not metadata_available:
        warnings.append("Runtime manifest rows did not include populated provenance fields for the authoritative split.")
        payload["warnings"] = warnings
        return payload

    manifest_index: Dict[str, Dict[str, Any]] = {}
    for row in split_rows:
        runtime_relative_path = _infer_runtime_relative_path_from_manifest_row(row)
        if runtime_relative_path:
            manifest_index[runtime_relative_path] = row

    prediction_rows = [
        dict(row)
        for row in list(getattr(authoritative_evaluation, "prediction_rows", []) or [])
        if isinstance(row, dict)
        and str(row.get("sample_origin", "") or "").strip() == "in_distribution"
        and str(row.get("split_name", classification_split) or classification_split).strip() == str(classification_split)
    ]
    authoritative_metric_rows = [
        dict(row)
        for row in prediction_rows
        if row.get("true_index") is not None and row.get("pred_index") is not None
    ]
    payload["authoritative_sample_count"] = int(len(prediction_rows))
    if not prediction_rows:
        warnings.append("Authoritative evaluation did not expose in-distribution prediction rows for provenance analysis.")
        payload["warnings"] = warnings
        return payload
    authoritative_pooled_metrics = _to_metric_triplet(
        compute_plan_metrics(
            y_true=[int(row["true_index"]) for row in authoritative_metric_rows],
            y_pred=[int(row["pred_index"]) for row in authoritative_metric_rows],
        )
    )

    matched_rows: list[Dict[str, Any]] = []
    unmatched_count = 0
    for row in prediction_rows:
        runtime_relative_path = _infer_runtime_relative_path_from_prediction(
            row.get("image_path", ""),
            classification_split=classification_split,
        )
        manifest_row = manifest_index.get(runtime_relative_path)
        if manifest_row is None:
            unmatched_count += 1
            continue
        true_index = row.get("true_index")
        pred_index = row.get("pred_index")
        if true_index is None or pred_index is None:
            continue
        enriched_row = {
            "true_index": int(true_index),
            "pred_index": int(pred_index),
            "runtime_relative_path": runtime_relative_path,
        }
        for field_name in _PROVENANCE_DIMENSIONS:
            enriched_row[field_name] = str(manifest_row.get(field_name, "") or "").strip()
        matched_rows.append(enriched_row)

    payload["matched_sample_count"] = int(len(matched_rows))
    payload["unmatched_sample_count"] = int(unmatched_count)
    if unmatched_count > 0:
        warnings.append(
            f"{unmatched_count} authoritative prediction row(s) could not be matched to runtime manifest rows."
        )
    if not matched_rows:
        warnings.append("No authoritative prediction rows matched populated provenance metadata.")
        payload["warnings"] = warnings
        return payload

    dimensions: Dict[str, Any] = {}
    for field_name in _PROVENANCE_DIMENSIONS:
        dimension_rows = [row for row in matched_rows if str(row.get(field_name, "") or "").strip()]
        if not dimension_rows:
            continue
        pooled_metrics = _to_metric_triplet(
            compute_plan_metrics(
                y_true=[int(row["true_index"]) for row in dimension_rows],
                y_pred=[int(row["pred_index"]) for row in dimension_rows],
            )
        )
        slice_to_rows: Dict[str, list[Dict[str, Any]]] = {}
        for row in dimension_rows:
            slice_to_rows.setdefault(str(row.get(field_name, "")), []).append(row)
        reported_slices = []
        skipped_slice_count = 0
        for slice_name in sorted(slice_to_rows.keys()):
            slice_rows = slice_to_rows[slice_name]
            if len(slice_rows) < int(max(1, min_reported_slice_samples)):
                skipped_slice_count += 1
                continue
            reported_slices.append(
                _slice_metric_payload(slice_name, slice_rows, pooled_metrics=authoritative_pooled_metrics)
            )
        worst_slices: Dict[str, Any] = {}
        for metric_name in _SLICE_METRIC_NAMES:
            metric_candidates = [
                slice_payload
                for slice_payload in reported_slices
                if slice_payload.get("metrics", {}).get(metric_name) is not None
            ]
            if metric_candidates:
                worst_slices[metric_name] = min(
                    metric_candidates,
                    key=lambda item: (
                        float(item["metrics"][metric_name]),
                        -int(item.get("sample_count", 0)),
                        str(item.get("name", "")),
                    ),
                )
        dimensions[field_name] = {
            "pooled_sample_count": int(len(dimension_rows)),
            "pooled_metrics": pooled_metrics,
            "reported_slice_count": int(len(reported_slices)),
            "skipped_slice_count": int(skipped_slice_count),
            "min_reported_slice_samples": int(max(1, min_reported_slice_samples)),
            "worst_slices": worst_slices,
            "slices": reported_slices,
        }

    payload["available"] = True
    payload["dimensions"] = dimensions
    payload["warnings"] = warnings
    return payload


def summarize_provenance_slice_breakdown(provenance_breakdown: Dict[str, Any] | None) -> Dict[str, Any]:
    breakdown = dict(provenance_breakdown or {})
    dimensions = dict(breakdown.get("dimensions", {})) if isinstance(breakdown.get("dimensions"), dict) else {}
    reported_dimensions = [
        field_name
        for field_name, payload in dimensions.items()
        if isinstance(payload, dict) and int(payload.get("reported_slice_count", 0)) > 0
    ]
    return {
        "available": bool(breakdown.get("available", False)),
        "classification_split": str(breakdown.get("classification_split", "") or ""),
        "matched_sample_count": int(breakdown.get("matched_sample_count", 0) or 0),
        "unmatched_sample_count": int(breakdown.get("unmatched_sample_count", 0) or 0),
        "reported_dimensions": reported_dimensions,
        "reported_dimension_count": int(len(reported_dimensions)),
    }


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
    provenance_summary: Dict[str, Any] | None = None,
    provenance_warnings: list[str] | None = None,
    ood_method_comparison: Dict[str, Any] | None = None,
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
        "provenance_summary": dict(provenance_summary or {}),
        "provenance_warnings": [str(item) for item in list(provenance_warnings or []) if str(item).strip()],
        "ood_method_comparison": dict(ood_method_comparison or {}),
    }


def build_training_summary_payload(
    *,
    run_id: str,
    crop_name: str,
    detected_classes: list[str],
    loader_sizes: Dict[str, int],
    loader_batch_counts: Dict[str, int],
    split_class_counts: Dict[str, Dict[str, int]],
    class_balance: Dict[str, Any],
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
    loss_name: str,
    logitnorm_tau: float,
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
        "class_balance": dict(class_balance),
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
        "optimization": {
            "loss_name": str(loss_name),
            "logitnorm_tau": float(logitnorm_tau),
        },
        "final_metrics": dict(final_metrics),
    }

