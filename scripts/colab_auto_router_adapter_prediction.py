#!/usr/bin/env python3
"""Thin Notebook 8 helper for Notebook 1 router result -> adapter prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.workflows.inference import InferenceWorkflow

StatusPrinter = Callable[[str], None]
WorkflowFactory = Callable[..., Any]

_ADAPTER_ALLOWED_ROUTER_STATUSES = {"ok", "trusted_hint_skipped", "skipped"}


def _emit_status(status_printer: Optional[StatusPrinter], message: str) -> None:
    if status_printer is not None:
        status_printer(str(message))


def _normalize_optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or text == "none":
        return None
    return text


def _router_payload(router_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = router_result.get("router")
    return dict(payload) if isinstance(payload, dict) else {}


def _resolve_router_handoff(router_result: Dict[str, Any]) -> Dict[str, Any]:
    router_payload = _router_payload(router_result)
    primary_detection = router_payload.get("primary_detection")
    primary = dict(primary_detection) if isinstance(primary_detection, dict) else {}
    crop = _normalize_optional_text(router_result.get("crop") or primary.get("crop"))
    part = _normalize_optional_text(router_result.get("part") or primary.get("part"))
    status = str(router_result.get("status") or router_payload.get("status") or "").strip().lower()
    message = str(router_result.get("message") or router_payload.get("message") or "")
    return {
        "status": status,
        "crop": crop,
        "part": part,
        "message": message,
        "router": router_payload,
    }


def run_auto_router_adapter_prediction(
    image_path: str | Path,
    *,
    router_result: Dict[str, Any],
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    adapter_root: Optional[str | Path] = None,
    return_ood: bool = True,
    status_printer: Optional[StatusPrinter] = None,
    workflow_factory: WorkflowFactory = InferenceWorkflow,
) -> Dict[str, Any]:
    """Run adapter prediction from an already-computed Notebook 1 router result.

    Notebook 8 deliberately gets crop/part routing from Notebook 1's maintained
    cell scripts, then calls the canonical inference workflow with a trusted
    handoff so router behavior is not duplicated in this wrapper.
    """
    if not isinstance(router_result, dict):
        raise TypeError("router_result must be a dictionary produced by Notebook 1 analysis.")

    handoff = _resolve_router_handoff(router_result)
    status = str(handoff["status"] or "").strip().lower()
    crop = handoff["crop"]
    part = handoff["part"]
    adapter_allowed = status in _ADAPTER_ALLOWED_ROUTER_STATUSES and bool(crop) and crop != "unknown"

    if not adapter_allowed:
        if not crop or crop == "unknown":
            result_status = "unknown_crop"
            default_message = "Router could not resolve a supported crop."
        else:
            result_status = status or "router_unavailable"
            default_message = "Router result is not eligible for adapter prediction."
        message = handoff["message"] or default_message
        _emit_status(status_printer, f"[AUTO] Adapter skipped: status={status or 'unknown'} crop={crop or 'unknown'}")
        return {
            "status": result_status,
            "crop": crop,
            "part": part,
            "router_confidence": float(router_result.get("router_confidence", 0.0) or 0.0),
            "diagnosis": None,
            "confidence": 0.0,
            "message": message,
            "router": handoff["router"],
            "router_handoff": {
                "adapter_ran": False,
                "source_status": status,
                "reason": message,
            },
        }

    _emit_status(
        status_printer,
        f"[AUTO] Router handoff accepted crop={crop} part={part or 'unknown'}; loading adapter.",
    )
    workflow = workflow_factory(
        environment=config_env,
        device=device,
        adapter_root=adapter_root,
        status_callback=status_printer,
    )
    payload = workflow.predict(
        image_path,
        crop_hint=crop,
        part_hint=part,
        return_ood=return_ood,
        trust_crop_hint=True,
    )
    combined = dict(payload)
    combined["router_source"] = handoff["router"]
    combined["router_handoff"] = {
        "adapter_ran": True,
        "source_status": status,
        "crop": crop,
        "part": part,
    }
    return combined


__all__ = ["run_auto_router_adapter_prediction"]
