"""Helpers for assembling stable inference payloads."""

from __future__ import annotations

from typing import Any, Dict

from src.shared.contracts import InferenceResult, OODAnalysis


def best_detection_from_analysis(analysis: Any) -> Dict[str, Any]:
    """Return the runtime-selected primary detection from an analysis payload."""
    detections = analysis.get("detections", []) if isinstance(analysis, dict) else []
    if not detections:
        return {}
    return detections[0]


def build_default_ood(*, is_ood: bool) -> OODAnalysis:
    return OODAnalysis(
        score_method="ensemble",
        primary_score=1.0 if is_ood else 0.0,
        decision_threshold=1.0 if is_ood else 0.0,
        is_ood=bool(is_ood),
        calibration_version=0,
    )


def normalize_ood_analysis(payload: Dict[str, Any] | None) -> OODAnalysis:
    return OODAnalysis.from_dict(payload)


def build_unknown_crop_result(*, part_name: str | None, router_confidence: float, include_ood: bool) -> InferenceResult:
    return InferenceResult(
        status="unknown_crop",
        crop=None,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=None,
        confidence=0.0,
        message="Router could not resolve a supported crop.",
        ood_analysis=build_default_ood(is_ood=True) if include_ood else None,
    )


def build_router_unavailable_result(*, message: str, include_ood: bool) -> InferenceResult:
    return InferenceResult(
        status="router_unavailable",
        crop=None,
        part=None,
        router_confidence=0.0,
        diagnosis=None,
        confidence=0.0,
        message=str(message),
        ood_analysis=build_default_ood(is_ood=False) if include_ood else None,
    )


def build_adapter_unavailable_result(
    *,
    crop_name: str,
    part_name: str | None,
    router_confidence: float,
    message: str,
    include_ood: bool,
) -> InferenceResult:
    return InferenceResult(
        status="adapter_unavailable",
        crop=crop_name,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=None,
        confidence=0.0,
        message=str(message),
        ood_analysis=build_default_ood(is_ood=False) if include_ood else None,
    )


def build_success_result(
    *,
    crop_name: str,
    part_name: str | None,
    router_confidence: float,
    result: Dict[str, Any],
    include_ood: bool,
) -> InferenceResult:
    disease = result.get("disease", {}) if isinstance(result, dict) else {}
    raw_ood = result.get("ood_analysis", {}) if isinstance(result, dict) else {}
    return InferenceResult(
        status=str(result.get("status", "success")),
        crop=crop_name,
        part=part_name,
        router_confidence=float(router_confidence),
        diagnosis=disease.get("name"),
        diagnosis_index=(
            None if disease.get("class_index") is None else int(disease.get("class_index"))
        ),
        confidence=float(disease.get("confidence", 0.0)),
        ood_analysis=normalize_ood_analysis(raw_ood) if include_ood else None,
    )
