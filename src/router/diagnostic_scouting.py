"""Diagnostic scouting helper extracted from the main VLM pipeline module."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class DiagnosticScoutingAnalyzer:
    """Simplified analyzer for crop classification using VLM models."""

    def __init__(self, config: Dict, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = config
        self.vlm_pipeline: Optional[Any] = None
        # Resolve lazily to avoid module import cycles.
        try:
            from src.router.vlm_pipeline import VLMPipeline

            self.vlm_pipeline = VLMPipeline(config, device=device)
        except Exception:
            pass

        vlm_conf = config.get("router", {}).get("vlm", {}) if isinstance(config.get("router"), dict) else {}
        self.confidence_threshold = config.get("vlm_confidence_threshold", vlm_conf.get("confidence_threshold", 0.8))
        configured_max = config.get("vlm_max_detections", vlm_conf.get("max_detections", 0))
        try:
            configured_max_int = int(configured_max)
        except Exception:
            configured_max_int = 0
        self.max_detections = None if configured_max_int <= 0 else configured_max_int

        logger.info("DiagnosticScoutingAnalyzer initialized on %s", self.device)

    def quick_assessment(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Quick wrapper that returns compact assessment status."""
        explanation: Dict[str, Any]
        if self.vlm_pipeline is not None and getattr(self.vlm_pipeline, "enabled", False):
            try:
                analysis = self.vlm_pipeline.process_image(image_tensor)
                explanation = {"analysis": analysis}
                status = "ok"
            except Exception as e:
                explanation = {"error": str(e)}
                status = "error"
        else:
            explanation = {
                "reason": "vlm_pipeline_disabled" if self.vlm_pipeline is not None else "no_vlm_pipeline"
            }
            status = "skipped"

        return {"status": status, "explanation": explanation}

    def analyze_image(self, image_tensor: torch.Tensor, crop_hint: Optional[str] = None) -> Dict[str, Any]:
        """Analyze image for crop identification."""
        del crop_hint

        if self.vlm_pipeline is None:
            return {
                "status": "error",
                "crop": "unknown",
                "part": "unknown",
                "confidence": 0.0,
                "detections": [],
                "message": "vlm_pipeline_unavailable",
            }

        try:
            analysis = self.vlm_pipeline.analyze_image(
                image_tensor,
                confidence_threshold=self.confidence_threshold,
                max_detections=self.max_detections,
            )

            detections = analysis.get("detections", []) or []
            normalized_detections = []
            best = None
            best_confidence = -1.0

            for det in detections:
                crop_conf = float(det.get("crop_confidence", det.get("confidence", 0.0)))
                normalized = {
                    "crop": det.get("crop", "unknown"),
                    "part": det.get("part", "unknown"),
                    "confidence": crop_conf,
                    "bbox": det.get("bbox"),
                    "mask": det.get("mask"),
                }
                normalized_detections.append(normalized)
                if crop_conf > best_confidence:
                    best_confidence = crop_conf
                    best = normalized

            if best is None:
                best = {"crop": "unknown", "part": "unknown", "confidence": 0.0}

            return {
                "status": "ok",
                "crop": best.get("crop", "unknown"),
                "part": best.get("part", "unknown"),
                "confidence": float(best.get("confidence", 0.0)),
                "detections": normalized_detections,
                "processing_time_ms": analysis.get("processing_time_ms", 0.0),
            }
        except Exception as e:
            return {
                "status": "error",
                "crop": "unknown",
                "part": "unknown",
                "confidence": 0.0,
                "detections": [],
                "message": str(e),
            }
