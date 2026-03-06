#!/usr/bin/env python3
"""Thin runtime that routes an image, loads one crop adapter, and returns a small payload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.utils.data_loader import preprocess_image


class RouterAdapterRuntime:
    """Minimal inference surface for router -> adapter -> OOD."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        environment: Optional[str] = None,
        device: str = "cuda",
        adapter_root: Optional[str | Path] = None,
    ) -> None:
        self.config = dict(config or get_config(environment=environment))
        self.device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        inference_cfg = self.config.get("inference", {})
        self.adapter_root = Path(adapter_root or inference_cfg.get("adapter_root", "models/adapters"))
        self.target_size = int(inference_cfg.get("target_size", 224))
        self.router: Optional[Any] = None
        self.adapters: Dict[str, IndependentCropAdapter] = {}

    def _build_router(self) -> Any:
        from src.router.vlm_pipeline import VLMPipeline

        return VLMPipeline(config=self.config, device=str(self.device))

    def _build_adapter(self, crop_name: str) -> IndependentCropAdapter:
        return IndependentCropAdapter(crop_name=crop_name, device=str(self.device))

    def load_router(self) -> Any:
        if self.router is None:
            self.router = self._build_router()
            self.router.load_models()
        return self.router

    def _resolve_adapter_dir(self, crop_name: str, adapter_dir: Optional[str | Path] = None) -> Path:
        if adapter_dir is not None:
            root = Path(adapter_dir)
        else:
            root = self.adapter_root / crop_name / "continual_sd_lora_adapter"
        if root.is_dir() and (root / "adapter_meta.json").exists():
            return root
        raise FileNotFoundError(f"Adapter not found for crop '{crop_name}' at {root}")

    def load_adapter(self, crop_name: str, adapter_dir: Optional[str | Path] = None) -> IndependentCropAdapter:
        crop_key = str(crop_name).strip().lower()
        if crop_key in self.adapters:
            return self.adapters[crop_key]

        resolved_dir = self._resolve_adapter_dir(crop_key, adapter_dir=adapter_dir)
        adapter = self._build_adapter(crop_key)
        adapter.load_adapter(str(resolved_dir))
        self.adapters[crop_key] = adapter
        return adapter

    def _coerce_image(self, image: Any) -> Any:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image

    def _route(self, image: Any) -> Dict[str, Any]:
        router = self.load_router()
        analysis = router.analyze_image(image)
        detections = analysis.get("detections", []) if isinstance(analysis, dict) else []
        if not detections:
            return {}
        return max(detections, key=lambda item: float(item.get("crop_confidence", 0.0)))

    @staticmethod
    def _default_ood(*, is_ood: bool) -> Dict[str, Any]:
        return {
            "ensemble_score": 1.0 if is_ood else 0.0,
            "class_threshold": 1.0 if is_ood else 0.0,
            "is_ood": bool(is_ood),
            "calibration_version": 0,
        }

    def predict(
        self,
        image: Any,
        crop_hint: Optional[str] = None,
        part_hint: Optional[str] = None,
        return_ood: bool = True,
    ) -> Dict[str, Any]:
        prepared_image = self._coerce_image(image)
        crop_name = str(crop_hint).strip().lower() if crop_hint else None
        part_name = str(part_hint).strip().lower() if part_hint else None
        router_confidence = 1.0 if crop_name else 0.0

        if not crop_name:
            detection = self._route(prepared_image)
            crop_name = str(detection.get("crop", "")).strip().lower() or None
            part_name = part_name or str(detection.get("part", "")).strip().lower() or None
            router_confidence = float(detection.get("crop_confidence", 0.0))

        if not crop_name or crop_name == "unknown":
            payload = {
                "status": "unknown_crop",
                "crop": None,
                "part": part_name,
                "router_confidence": float(router_confidence),
                "diagnosis": None,
                "confidence": 0.0,
                "message": "Router could not resolve a supported crop.",
            }
            if return_ood:
                payload["ood_analysis"] = self._default_ood(is_ood=True)
            return payload

        try:
            adapter = self.load_adapter(crop_name)
        except FileNotFoundError as exc:
            payload = {
                "status": "adapter_unavailable",
                "crop": crop_name,
                "part": part_name,
                "router_confidence": float(router_confidence),
                "diagnosis": None,
                "confidence": 0.0,
                "message": str(exc),
            }
            if return_ood:
                payload["ood_analysis"] = self._default_ood(is_ood=False)
            return payload

        image_tensor = preprocess_image(prepared_image, target_size=self.target_size)
        result = adapter.predict_with_ood(image_tensor)
        disease = result.get("disease", {}) if isinstance(result, dict) else {}
        raw_ood = result.get("ood_analysis", {}) if isinstance(result, dict) else {}
        ood_payload = {
            "ensemble_score": float(raw_ood.get("ensemble_score", 0.0)),
            "class_threshold": float(raw_ood.get("class_threshold", 0.0)),
            "is_ood": bool(raw_ood.get("is_ood", False)),
            "calibration_version": int(raw_ood.get("calibration_version", 0)),
        }

        payload = {
            "status": str(result.get("status", "success")),
            "crop": crop_name,
            "part": part_name,
            "router_confidence": float(router_confidence),
            "diagnosis": disease.get("name"),
            "diagnosis_index": disease.get("class_index"),
            "confidence": float(disease.get("confidence", 0.0)),
        }
        if return_ood:
            payload["ood_analysis"] = ood_payload
        return payload
