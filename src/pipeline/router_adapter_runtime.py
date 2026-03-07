#!/usr/bin/env python3
"""Thin runtime that routes an image, loads one crop adapter, and returns a small payload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.pipeline.inference_payloads import (
    best_detection_from_analysis,
    build_adapter_unavailable_result,
    build_success_result,
    build_unknown_crop_result,
)
from src.shared.contracts import InferenceResult
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
        return best_detection_from_analysis(analysis)

    def predict_result(
        self,
        image: Any,
        crop_hint: Optional[str] = None,
        part_hint: Optional[str] = None,
        return_ood: bool = True,
    ) -> InferenceResult:
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
            return build_unknown_crop_result(
                part_name=part_name,
                router_confidence=router_confidence,
                include_ood=return_ood,
            )

        try:
            adapter = self.load_adapter(crop_name)
        except FileNotFoundError as exc:
            return build_adapter_unavailable_result(
                crop_name=crop_name,
                part_name=part_name,
                router_confidence=router_confidence,
                message=str(exc),
                include_ood=return_ood,
            )

        image_tensor = preprocess_image(prepared_image, target_size=self.target_size)
        result = adapter.predict_with_ood(image_tensor)
        return build_success_result(
            crop_name=crop_name,
            part_name=part_name,
            router_confidence=router_confidence,
            result=result,
            include_ood=return_ood,
        )

    def predict(
        self,
        image: Any,
        crop_hint: Optional[str] = None,
        part_hint: Optional[str] = None,
        return_ood: bool = True,
    ) -> Dict[str, Any]:
        return self.predict_result(
            image,
            crop_hint=crop_hint,
            part_hint=part_hint,
            return_ood=return_ood,
        ).to_dict(include_ood=return_ood)
