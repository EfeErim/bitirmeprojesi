#!/usr/bin/env python3
"""Thin entrypoint for Colab router-only crop and part identification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image

from src.core.config_manager import get_config
from src.pipeline.inference_payloads import build_router_skipped_analysis
from src.router.vlm_pipeline import VLMPipeline
from src.shared.contracts import RouterAnalysisResult

StatusPrinter = Callable[[str], None]


def _emit_status(status_printer: Optional[StatusPrinter], message: str) -> None:
    if status_printer is not None:
        status_printer(str(message))


def _build_router_payload(analysis: RouterAnalysisResult) -> Dict[str, Any]:
    detection = analysis.primary_detection
    return {
        "status": str(analysis.status or "ok"),
        "crop": None if detection is None else str(detection.crop or "unknown"),
        "part": None if detection is None else str(detection.part or "unknown"),
        "router_confidence": 0.0 if detection is None else float(detection.crop_confidence),
        "message": str(analysis.message or ""),
        "router": analysis.to_summary_dict(),
    }


def run_inference(
    image_path: str | Path,
    *,
    config_env: Optional[str] = "colab",
    crop_hint: Optional[str] = None,
    part_hint: Optional[str] = None,
    adapter_root: Optional[str | Path] = None,
    device: str = "cuda",
    status_printer: Optional[StatusPrinter] = None,
) -> Dict[str, Any]:
    del adapter_root
    image_ref = Path(image_path)
    _emit_status(status_printer, f"[INFER] image={image_ref.name} device={device}")
    config = get_config(environment=config_env)
    image = Image.open(image_path).convert("RGB")

    if crop_hint:
        crop_name = str(crop_hint).strip().lower()
        part_name = str(part_hint).strip().lower() if part_hint else None
        _emit_status(
            status_printer,
            f"[ROUTER] Skipped; using crop hint crop={crop_name} part={part_name or 'unknown'}",
        )
        analysis = build_router_skipped_analysis(
            crop_name=crop_name,
            part_name=part_name,
            router_confidence=1.0,
        )
        payload = _build_router_payload(analysis)
        _emit_status(
            status_printer,
            f"[RESULT] status={payload['status']} crop={payload['crop'] or 'unknown'} "
            f"part={payload['part'] or 'unknown'} router_confidence={float(payload['router_confidence']):.3f}",
        )
        return payload

    _emit_status(status_printer, f"[ROUTER] Loading models on {device}...")
    router = VLMPipeline(config=config, device=device)
    router.load_models()
    if not router.is_ready():
        raise RuntimeError(
            "Router models failed to become ready for inference. "
            "Check router.vlm.enabled, model availability, and VLM dependency installation."
        )
    _emit_status(status_printer, "[ROUTER] Ready.")

    analysis = router.analyze_image_result(image)
    payload = _build_router_payload(analysis)
    _emit_status(
        status_printer,
        f"[ROUTER] crop={payload['crop'] or 'unknown'} "
        f"part={payload['part'] or 'unknown'} "
        f"confidence={float(payload['router_confidence']):.3f}"
        + (f" message={payload['message']}" if payload["message"] else ""),
    )
    _emit_status(
        status_printer,
        f"[RESULT] status={payload['status']} crop={payload['crop'] or 'unknown'} "
        f"part={payload['part'] or 'unknown'} router_confidence={float(payload['router_confidence']):.3f}",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run router-only crop and part identification for a single image.")
    parser.add_argument("image", type=Path, help="Image path")
    parser.add_argument("--config-env", default="colab", help="Config environment override (default: colab)")
    parser.add_argument("--crop", dest="crop_hint", help="Optional crop hint to bypass the router")
    parser.add_argument("--part", dest="part_hint", help="Optional part hint")
    parser.add_argument("--adapter-root", type=Path, help="Unused legacy adapter-root argument kept for notebook compatibility")
    parser.add_argument("--device", default="cuda", help="Torch device preference")
    args = parser.parse_args()

    result = run_inference(
        args.image,
        config_env=args.config_env,
        crop_hint=args.crop_hint,
        part_hint=args.part_hint,
        adapter_root=args.adapter_root,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
