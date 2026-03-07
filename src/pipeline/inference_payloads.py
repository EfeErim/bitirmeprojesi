"""Helpers for assembling stable inference payloads."""

from __future__ import annotations

from typing import Any, Dict

from src.shared.contracts import InferenceResult, OODAnalysis


def best_detection_from_analysis(analysis: Any) -> Dict[str, Any]:
    detections = analysis.get("detections", []) if isinstance(analysis, dict) else []
    if not detections:
        return {}
    return max(detections, key=lambda item: float(item.get("crop_confidence", 0.0)))


def build_default_ood(*, is_ood: bool) -> OODAnalysis:
    return OODAnalysis(
        ensemble_score=1.0 if is_ood else 0.0,
        class_threshold=1.0 if is_ood else 0.0,
        is_ood=bool(is_ood),
        calibration_version=0,
    )


def normalize_ood_analysis(payload: Dict[str, Any] | None) -> OODAnalysis:
    data = dict(payload or {})
    return OODAnalysis(
        ensemble_score=float(data.get("ensemble_score", 0.0)),
        class_threshold=float(data.get("class_threshold", 0.0)),
        is_ood=bool(data.get("is_ood", False)),
        calibration_version=int(data.get("calibration_version", 0)),
        mahalanobis_z=(
            None if data.get("mahalanobis_z") is None else float(data.get("mahalanobis_z", 0.0))
        ),
        energy_z=None if data.get("energy_z") is None else float(data.get("energy_z", 0.0)),
    )


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
