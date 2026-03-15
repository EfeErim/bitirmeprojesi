#!/usr/bin/env python3
"""Thin runtime that routes an image, loads one crop adapter, and returns a small payload."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from PIL import Image

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.transforms import preprocess_image
from src.pipeline.inference_payloads import (
    build_adapter_unavailable_result,
    build_router_skipped_analysis,
    build_router_unavailable_result,
    build_success_result,
    build_unknown_crop_result,
    normalize_router_analysis,
)
from src.shared.contracts import InferenceResult, RouterAnalysisResult
from src.training.services.runtime import resolve_runtime_device

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class _CachedAdapter:
    adapter: IndependentCropAdapter
    adapter_dir: Path
    adapter_bundle_token: str


class RouterAdapterRuntime:
    """Minimal inference surface for router -> adapter -> OOD."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        environment: Optional[str] = None,
        device: str = "cuda",
        adapter_root: Optional[str | Path] = None,
        status_callback: Optional[StatusCallback] = None,
    ) -> None:
        self.config = dict(config or get_config(environment=environment))
        self.device = resolve_runtime_device(device)
        inference_cfg = self.config.get("inference", {})
        self.adapter_root = Path(adapter_root or inference_cfg.get("adapter_root", "models/adapters"))
        self.target_size = int(inference_cfg.get("target_size", 224))
        self.router: Optional[Any] = None
        self.adapters: Dict[str, _CachedAdapter] = {}
        self.status_callback = status_callback

    def _emit_status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(str(message))

    def _build_router(self) -> Any:
        from src.router.vlm_pipeline import VLMPipeline

        return VLMPipeline(config=self.config, device=str(self.device))

    def _build_adapter(self, crop_name: str) -> IndependentCropAdapter:
        return IndependentCropAdapter(crop_name=crop_name, device=str(self.device))

    def load_router(self) -> Any:
        if self.router is not None:
            readiness_probe = getattr(self.router, "is_ready", None)
            if callable(readiness_probe) and not bool(readiness_probe()):
                self.router = None
        if self.router is None:
            self._emit_status(f"[ROUTER] Loading models on {self.device}...")
            router = self._build_router()
            try:
                router.load_models()
                readiness_probe = getattr(router, "is_ready", None)
                if callable(readiness_probe) and not bool(readiness_probe()):
                    self._emit_status("[ROUTER] Unavailable.")
                    raise RuntimeError(
                        "Router models failed to become ready for inference. "
                        "Check router.vlm.enabled, model availability, and VLM dependency installation."
                    )
            except Exception:
                self.router = None
                raise
            self.router = router
            self._emit_status("[ROUTER] Ready.")
        return self.router

    def _resolve_adapter_dir(self, crop_name: str, adapter_dir: Optional[str | Path] = None) -> Path:
        if adapter_dir is not None:
            root = Path(adapter_dir)
        else:
            root = self.adapter_root / crop_name / "continual_sd_lora_adapter"
        if root.is_dir() and (root / "adapter_meta.json").exists():
            return root
        raise FileNotFoundError(f"Adapter not found for crop '{crop_name}' at {root}")

    @staticmethod
    def _bundle_sentinel_paths(adapter_dir: Path) -> Iterable[Path]:
        try:
            entries = sorted(adapter_dir.iterdir())
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No adapter assets found under {adapter_dir}") from exc

        yielded: set[Path] = set()
        for entry in entries:
            yielded.add(entry)
            yield entry
            if not entry.is_dir():
                continue
            try:
                children = sorted(entry.iterdir())
            except OSError:
                continue
            for child in children:
                if not child.is_file():
                    continue
                yielded.add(child)
                yield child
        meta_path = adapter_dir / "adapter_meta.json"
        if meta_path.exists() and meta_path not in yielded:
            yield meta_path

    @staticmethod
    def _adapter_cache_token(adapter_dir: Path) -> tuple[Path, str]:
        resolved_dir = adapter_dir.resolve()
        digest = hashlib.sha256()
        file_count = 0
        for path in RouterAdapterRuntime._bundle_sentinel_paths(resolved_dir):
            stat = path.stat()
            digest.update(path.relative_to(resolved_dir).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(b"f" if path.is_file() else b"d")
            digest.update(b":")
            digest.update(str(int(stat.st_size)).encode("utf-8"))
            digest.update(b":")
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
            digest.update(b"\0")
            file_count += 1
        if file_count <= 0:
            raise FileNotFoundError(f"No adapter assets found under {resolved_dir}")
        return resolved_dir, digest.hexdigest()

    def load_adapter(self, crop_name: str, adapter_dir: Optional[str | Path] = None) -> IndependentCropAdapter:
        crop_key = str(crop_name).strip().lower()
        resolved_dir = self._resolve_adapter_dir(crop_key, adapter_dir=adapter_dir)
        resolved_cache_dir, bundle_token = self._adapter_cache_token(resolved_dir)
        cached = self.adapters.get(crop_key)
        if cached is not None:
            if cached.adapter_dir == resolved_cache_dir and cached.adapter_bundle_token == bundle_token:
                return cached.adapter
            self._emit_status(f"[ADAPTER] Reloading adapter for crop={crop_key}...")
        else:
            self._emit_status(f"[ADAPTER] Loading adapter for crop={crop_key}...")

        adapter = self._build_adapter(crop_key)
        adapter.load_adapter(str(resolved_dir))
        self.adapters[crop_key] = _CachedAdapter(
            adapter=adapter,
            adapter_dir=resolved_cache_dir,
            adapter_bundle_token=bundle_token,
        )
        self._emit_status(f"[ADAPTER] Ready crop={crop_key}")
        return adapter

    def _coerce_image(self, image: Any) -> Any:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image

    def _route(self, image: Any) -> RouterAnalysisResult:
        router = self.load_router()
        if hasattr(router, "analyze_image_result"):
            analysis = router.analyze_image_result(image)
        else:
            analysis = router.analyze_image(image)
        if analysis is None:
            raise RuntimeError("Router returned no analysis payload.")
        if not isinstance(analysis, (RouterAnalysisResult, dict)):
            raise TypeError(f"Router returned unsupported analysis payload type: {type(analysis).__name__}")
        normalized = normalize_router_analysis(analysis)
        router_status = str(normalized.status or "").strip().lower()
        if router_status in {"unavailable", "error", "failed"}:
            raise RuntimeError(str(normalized.message or f"Router reported status={router_status}"))
        return normalized

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
        router_analysis: RouterAnalysisResult

        if crop_name:
            self._emit_status(
                f"[ROUTER] Skipped; using crop hint crop={crop_name} part={part_name or 'unknown'}"
            )
            router_analysis = build_router_skipped_analysis(
                crop_name=crop_name,
                part_name=part_name,
                router_confidence=router_confidence,
            )
        else:
            try:
                router_analysis = self._route(prepared_image)
            except Exception as exc:
                result = build_router_unavailable_result(
                    message=f"Router runtime unavailable: {exc}",
                    include_ood=return_ood,
                )
                self._emit_status(f"[RESULT] status={result.status} message={result.message}")
                return result
            detection = router_analysis.primary_detection.to_dict() if router_analysis.primary_detection else {}
            crop_name = str(detection.get("crop", "")).strip().lower() or None
            part_name = part_name or str(detection.get("part", "")).strip().lower() or None
            router_confidence = float(detection.get("crop_confidence", 0.0))
            status_message = (
                f"[ROUTER] crop={crop_name or 'unknown'} "
                f"part={part_name or 'unknown'} "
                f"confidence={router_confidence:.3f}"
            )
            if router_analysis.message:
                status_message = f"{status_message} message={router_analysis.message}"
            self._emit_status(
                status_message
            )

        if not crop_name or crop_name == "unknown":
            result = build_unknown_crop_result(
                part_name=part_name,
                router_confidence=router_confidence,
                include_ood=return_ood,
                router_analysis=router_analysis,
            )
            self._emit_status(
                f"[RESULT] status={result.status} router_confidence={result.router_confidence:.3f}"
            )
            return result

        try:
            adapter = self.load_adapter(crop_name)
        except FileNotFoundError as exc:
            result = build_adapter_unavailable_result(
                crop_name=crop_name,
                part_name=part_name,
                router_confidence=router_confidence,
                message=str(exc),
                include_ood=return_ood,
                router_analysis=router_analysis,
            )
            self._emit_status(f"[RESULT] status={result.status} crop={crop_name} message={result.message}")
            return result

        image_tensor = preprocess_image(prepared_image, target_size=self.target_size)
        adapter_result = adapter.predict_with_ood(image_tensor)
        payload = build_success_result(
            crop_name=crop_name,
            part_name=part_name,
            router_confidence=router_confidence,
            result=adapter_result,
            include_ood=return_ood,
            router_analysis=router_analysis,
        )
        status_bits = [
            f"[RESULT] status={payload.status}",
            f"crop={payload.crop or 'unknown'}",
            f"confidence={payload.confidence:.3f}",
        ]
        if payload.diagnosis:
            status_bits.insert(2, f"diagnosis={payload.diagnosis}")
        if return_ood and payload.ood_analysis is not None:
            status_bits.append(f"ood={payload.ood_analysis.is_ood}")
        self._emit_status(" ".join(status_bits))
        return payload

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
