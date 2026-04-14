#!/usr/bin/env python3
"""Thin runtime that routes an image, loads one crop adapter, and returns a small payload."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, TypeAlias

from PIL import Image

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.transforms import preprocess_image
from src.shared.adapter_paths import normalize_adapter_name, resolve_adapter_bundle_dir
from src.shared.json_utils import read_json_dict
from src.pipeline.inference_payloads import (
    build_adapter_unavailable_result,
    build_router_skipped_analysis,
    build_router_unavailable_result,
    build_router_uncertain_result,
    build_success_result,
    build_unknown_crop_result,
    normalize_router_analysis,
)
from src.shared.contracts import InferenceResult, RouterAnalysisResult
from src.training.services.runtime import resolve_runtime_device

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]
ImageInput: TypeAlias = Image.Image | str | Path


class RouterLike(Protocol):
    def load_models(self) -> None: ...

    def analyze_image(self, image: ImageInput) -> RouterAnalysisResult | Dict[str, Any]: ...

    def analyze_image_result(self, image: ImageInput) -> RouterAnalysisResult | Dict[str, Any]: ...

    def is_ready(self) -> bool: ...


@dataclass(frozen=True)
class _CachedAdapter:
    adapter: IndependentCropAdapter
    adapter_dir: Path
    part_name: str
    adapter_meta_mtime_ns: int
    adapter_meta_size: int


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
        self.router_min_confidence = float(inference_cfg.get("router_min_confidence", 0.65))
        self.router_min_margin = float(inference_cfg.get("router_min_margin", 0.10))
        self.router: Optional[RouterLike] = None
        self.adapters: Dict[str, _CachedAdapter] = {}
        self.status_callback = status_callback

    def _emit_status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(str(message))

    def _build_router(self) -> RouterLike:
        from src.router.router_pipeline import RouterPipeline

        return RouterPipeline(config=self.config, device=str(self.device))

    def _build_adapter(self, crop_name: str) -> IndependentCropAdapter:
        return IndependentCropAdapter(crop_name=crop_name, device=str(self.device))

    @staticmethod
    def _adapter_cache_key(crop_name: str, part_name: Optional[str]) -> str:
        return f"{normalize_adapter_name(crop_name)}::{normalize_adapter_name(part_name)}"

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
                        "Check router.vlm.enabled, model availability, and router dependency installation."
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Router initialization failed: %s", exc)
                self.router = None
                raise
            self.router = router
            self._emit_status("[ROUTER] Ready.")
        return self.router

    def _resolve_adapter_dir(
        self,
        crop_name: str,
        part_name: Optional[str] = None,
        adapter_dir: Optional[str | Path] = None,
    ) -> Path:
        def _discover_fallback_adapter_dir(search_root: Path) -> Optional[Path]:
            if not search_root.exists() or not search_root.is_dir():
                return None

            crop_key = normalize_adapter_name(crop_name)
            requested_part = normalize_adapter_name(part_name) if part_name else ""
            ranked_candidates: list[tuple[int, Path]] = []
            for meta_path in sorted(search_root.rglob("adapter_meta.json")):
                adapter_bundle_dir = meta_path.parent
                if adapter_bundle_dir.name != "continual_sd_lora_adapter":
                    continue
                try:
                    meta = read_json_dict(meta_path)
                except Exception:
                    continue
                meta_crop = normalize_adapter_name(meta.get("crop_name")) if meta.get("crop_name") else ""
                if meta_crop and meta_crop != crop_key:
                    continue
                meta_part = normalize_adapter_name(meta.get("part_name")) if meta.get("part_name") else ""
                if requested_part and meta_part == requested_part:
                    ranked_candidates.append((0, adapter_bundle_dir))
                elif not requested_part and (not meta_part or meta_part == "unspecified"):
                    ranked_candidates.append((1, adapter_bundle_dir))
                elif not requested_part:
                    ranked_candidates.append((2, adapter_bundle_dir))
                elif meta_part in {"", "unspecified"}:
                    ranked_candidates.append((1, adapter_bundle_dir))
                else:
                    ranked_candidates.append((2, adapter_bundle_dir))
            if not ranked_candidates:
                return None
            ranked_candidates.sort(key=lambda item: (item[0], str(item[1])))
            return ranked_candidates[0][1]

        if adapter_dir is not None:
            root = Path(adapter_dir)
            try:
                return resolve_adapter_bundle_dir(root, crop_name=crop_name, part_name=part_name)
            except FileNotFoundError as exc:
                fallback = _discover_fallback_adapter_dir(root)
                if fallback is not None:
                    return fallback
                raise FileNotFoundError(f"Could not resolve adapter bundle from {root}") from exc
        try:
            return resolve_adapter_bundle_dir(self.adapter_root, crop_name=crop_name, part_name=part_name)
        except FileNotFoundError as exc:
            fallback = _discover_fallback_adapter_dir(self.adapter_root)
            if fallback is not None:
                return fallback
            expected_part = normalize_adapter_name(part_name) if part_name else "unspecified"
            raise FileNotFoundError(
                f"Adapter not found for crop '{crop_name}' part '{expected_part}' under {self.adapter_root}"
            ) from exc

    @staticmethod
    def _adapter_meta_state(adapter_dir: Path) -> tuple[Path, str, int, int]:
        resolved_dir = adapter_dir.resolve()
        meta_path = resolved_dir / "adapter_meta.json"
        try:
            meta_stat = meta_path.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No adapter assets found under {resolved_dir}") from exc

        # Adapter export rewrites adapter_meta.json after bundle assets are persisted,
        # so metadata freshness is the maintained reload marker for runtime caching.
        meta = read_json_dict(meta_path)
        return (
            resolved_dir,
            normalize_adapter_name(meta.get("part_name")) if meta.get("part_name") else "unspecified",
            int(meta_stat.st_mtime_ns),
            int(meta_stat.st_size),
        )

    def load_adapter(
        self,
        crop_name: str,
        part_name: Optional[str] = None,
        adapter_dir: Optional[str | Path] = None,
    ) -> IndependentCropAdapter:
        crop_key = str(crop_name).strip().lower()
        part_key = normalize_adapter_name(part_name) if part_name else None
        resolved_dir = self._resolve_adapter_dir(crop_key, part_name=part_key, adapter_dir=adapter_dir)
        (
            resolved_cache_dir,
            resolved_part_name,
            adapter_meta_mtime_ns,
            adapter_meta_size,
        ) = self._adapter_meta_state(resolved_dir)
        cache_key = self._adapter_cache_key(crop_key, part_key or resolved_part_name)
        cached = self.adapters.get(cache_key)
        if cached is not None:
            if (
                cached.adapter_dir == resolved_cache_dir
                and cached.part_name == (part_key or resolved_part_name)
                and cached.adapter_meta_mtime_ns == adapter_meta_mtime_ns
                and cached.adapter_meta_size == adapter_meta_size
            ):
                return cached.adapter
            self._emit_status(f"[ADAPTER] Reloading adapter for crop={crop_key} part={part_key or resolved_part_name}...")
        else:
            self._emit_status(f"[ADAPTER] Loading adapter for crop={crop_key} part={part_key or resolved_part_name}...")

        adapter = self._build_adapter(crop_key)
        adapter.part_name = part_key or resolved_part_name
        adapter.load_adapter(str(resolved_dir))
        self.adapters[cache_key] = _CachedAdapter(
            adapter=adapter,
            adapter_dir=resolved_cache_dir,
            part_name=part_key or resolved_part_name,
            adapter_meta_mtime_ns=adapter_meta_mtime_ns,
            adapter_meta_size=adapter_meta_size,
        )
        self._emit_status(f"[ADAPTER] Ready crop={crop_key} part={part_key or resolved_part_name}")
        return adapter

    def _coerce_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened_image:
                return opened_image.convert("RGB")
        return image

    def _route(self, image: Image.Image) -> RouterAnalysisResult:
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

    @staticmethod
    def _routing_score(detection: Any) -> float:
        quality_score = getattr(detection, "quality_score", None)
        if quality_score is not None:
            return float(quality_score)
        return float(getattr(detection, "crop_confidence", 0.0))

    def _routing_runner_up(self, router_analysis: RouterAnalysisResult) -> tuple[Optional[str], Optional[float]]:
        primary = router_analysis.primary_detection
        if primary is None:
            return None, None
        primary_crop = str(primary.crop or "").strip().lower()
        best_crop: Optional[str] = None
        best_score: Optional[float] = None
        for detection in list(router_analysis.detections or []):
            candidate_crop = str(getattr(detection, "crop", "") or "").strip().lower()
            if not candidate_crop or candidate_crop == primary_crop:
                continue
            candidate_score = self._routing_score(detection)
            if best_score is None or candidate_score > best_score:
                best_crop = candidate_crop
                best_score = candidate_score
        return best_crop, best_score

    def _router_uncertainty_message(self, router_analysis: RouterAnalysisResult) -> str:
        primary = router_analysis.primary_detection
        if primary is None:
            return ""

        reasons: list[str] = []
        primary_confidence = float(primary.crop_confidence)
        if self.router_min_confidence > 0.0 and primary_confidence < self.router_min_confidence:
            reasons.append(
                f"crop_confidence={primary_confidence:.3f} < min_confidence={self.router_min_confidence:.3f}"
            )

        runner_up_crop, runner_up_score = self._routing_runner_up(router_analysis)
        if runner_up_score is not None and self.router_min_margin > 0.0:
            primary_score = self._routing_score(primary)
            margin = primary_score - runner_up_score
            if margin < self.router_min_margin:
                reasons.append(
                    "routing_margin="
                    f"{margin:.3f} < min_margin={self.router_min_margin:.3f} "
                    f"vs alternate_crop={runner_up_crop}"
                )

        if not reasons:
            return ""
        return f"Router uncertainty gate rejected crop='{primary.crop}': " + "; ".join(reasons)

    def predict_result(
        self,
        image: ImageInput,
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
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Router analysis failed: %s", exc)
                result = build_router_unavailable_result(
                    message=f"Router runtime unavailable: {exc}",
                    include_ood=return_ood,
                )
                self._emit_status(f"[RESULT] status={result.status} message={result.message}")
                return result
            detection = router_analysis.primary_detection.to_dict() if router_analysis.primary_detection else {}
            crop_name = str(detection.get("crop", "")).strip().lower() or None
            part_name = str(detection.get("part", "")).strip().lower() or None
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

        if not crop_hint:
            uncertainty_message = self._router_uncertainty_message(router_analysis)
            if uncertainty_message:
                result = build_router_uncertain_result(
                    crop_name=crop_name,
                    part_name=part_name,
                    router_confidence=router_confidence,
                    message=uncertainty_message,
                    include_ood=return_ood,
                    router_analysis=router_analysis,
                )
                self._emit_status(
                    f"[RESULT] status={result.status} "
                    f"router_confidence={result.router_confidence:.3f} "
                    f"message={result.message}"
                )
                return result

        try:
            adapter = self.load_adapter(crop_name, part_name=part_name)
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

