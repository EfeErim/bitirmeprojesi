"""Helpers for requested-vs-selected OOD score methods."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from src.training.services.metrics import compute_plan_metrics
from src.training.types import EvaluationArtifactsPayload

AUTO_PRIMARY_SCORE_METHOD = "auto"
SUPPORTED_CONCRETE_OOD_SCORE_METHODS = ("ensemble", "energy", "knn")
SUPPORTED_REQUESTED_OOD_SCORE_METHODS = (AUTO_PRIMARY_SCORE_METHOD, *SUPPORTED_CONCRETE_OOD_SCORE_METHODS)
_METHOD_PREFERENCE = {"ensemble": 0, "energy": 1, "knn": 2}


def normalize_requested_primary_score_method(value: Any) -> str:
    resolved = str(value or AUTO_PRIMARY_SCORE_METHOD).strip().lower()
    if resolved not in SUPPORTED_REQUESTED_OOD_SCORE_METHODS:
        raise ValueError(
            "ood.primary_score_method must be one of: "
            + ", ".join(SUPPORTED_REQUESTED_OOD_SCORE_METHODS)
            + "."
        )
    return resolved


def is_auto_primary_score_method(value: Any) -> bool:
    return normalize_requested_primary_score_method(value) == AUTO_PRIMARY_SCORE_METHOD


def resolve_runtime_primary_score_method(value: Any) -> str:
    requested = normalize_requested_primary_score_method(value)
    if requested == AUTO_PRIMARY_SCORE_METHOD:
        return "ensemble"
    return requested


def compute_method_metrics_from_evaluation(evaluation: Optional[EvaluationArtifactsPayload]) -> Dict[str, Dict[str, Optional[float]]]:
    if evaluation is None or not evaluation.ood_labels:
        return {}
    method_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for method_name, scores in dict(evaluation.ood_scores_by_method or {}).items():
        score_values = list(scores)
        if not score_values:
            continue
        method_metrics[str(method_name)] = compute_plan_metrics(
            y_true=evaluation.y_true,
            y_pred=evaluation.y_pred,
            ood_labels=evaluation.ood_labels,
            ood_scores=score_values,
            sure_ds_f1=evaluation.sure_ds_f1,
            conformal_empirical_coverage=evaluation.conformal_empirical_coverage,
            conformal_avg_set_size=evaluation.conformal_avg_set_size,
        )
    return method_metrics


def select_best_ood_score_method(
    method_metrics: Mapping[str, Mapping[str, Any]],
    *,
    fallback: str = "ensemble",
) -> str:
    resolved_fallback = resolve_runtime_primary_score_method(fallback)
    candidates = []
    for method_name in SUPPORTED_CONCRETE_OOD_SCORE_METHODS:
        metrics = dict(method_metrics.get(method_name, {}))
        auroc = metrics.get("ood_auroc")
        fpr = metrics.get("ood_false_positive_rate")
        if auroc is None and fpr is None:
            continue
        candidates.append(
            (
                float("-inf") if auroc is None else float(auroc),
                float("inf") if fpr is None else float(fpr),
                _METHOD_PREFERENCE[method_name],
                method_name,
            )
        )
    if not candidates:
        return resolved_fallback
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return str(candidates[0][3])


def apply_primary_score_method_to_evaluation(
    evaluation: Optional[EvaluationArtifactsPayload],
    primary_score_method: str,
    *,
    requested_primary_score_method: Optional[str] = None,
    selection_source: str = "",
) -> Optional[EvaluationArtifactsPayload]:
    if evaluation is None:
        return None
    resolved_method = resolve_runtime_primary_score_method(primary_score_method)
    selected_scores = list(dict(evaluation.ood_scores_by_method or {}).get(resolved_method, evaluation.ood_scores or []))
    updated_context = dict(evaluation.context)
    updated_context["ood_primary_score_method"] = resolved_method
    if requested_primary_score_method is not None:
        updated_context["ood_requested_primary_score_method"] = normalize_requested_primary_score_method(
            requested_primary_score_method
        )
    if selection_source:
        updated_context["ood_primary_score_selection_source"] = str(selection_source)
    return replace(
        evaluation,
        ood_scores=selected_scores if selected_scores else evaluation.ood_scores,
        ood_primary_score_method=resolved_method,
        context=updated_context,
    )
