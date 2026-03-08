#!/usr/bin/env python3
"""Direct adapter smoke-test helpers for Colab notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, UnidentifiedImageError

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.transforms import preprocess_image
from src.shared.json_utils import read_json_dict

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_config(config_env: Optional[str]) -> Dict[str, Any]:
    return dict(get_config(environment=config_env))


def _default_adapter_root(config_env: Optional[str]) -> Path:
    config = _load_config(config_env)
    inference_cfg = config.get("inference", {})
    return Path(str(inference_cfg.get("adapter_root", "models/adapters")))


def _target_size(config_env: Optional[str]) -> int:
    config = _load_config(config_env)
    inference_cfg = config.get("inference", {})
    return int(inference_cfg.get("target_size", 224))


def _normalize_crop_name(crop_name: str) -> str:
    normalized = str(crop_name).strip().lower()
    if not normalized:
        raise ValueError("crop_name must be a non-empty string")
    return normalized


def _resolve_adapter_dir(
    crop_name: str,
    *,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
) -> Path:
    crop_key = _normalize_crop_name(crop_name)
    if adapter_dir is not None:
        root = Path(adapter_dir)
        if root.is_dir() and (root / "adapter_meta.json").exists():
            return root
        nested = root / "continual_sd_lora_adapter"
        if nested.is_dir() and (nested / "adapter_meta.json").exists():
            return nested
        raise FileNotFoundError(f"adapter_meta.json not found under explicit adapter_dir={root}")

    base_root = Path(adapter_root) if adapter_root is not None else _default_adapter_root(config_env)
    candidate = base_root / crop_key / "continual_sd_lora_adapter"
    if candidate.is_dir() and (candidate / "adapter_meta.json").exists():
        return candidate
    raise FileNotFoundError(f"Adapter not found for crop '{crop_key}' at {candidate}")


def _build_adapter(crop_name: str, *, device: str) -> IndependentCropAdapter:
    return IndependentCropAdapter(crop_name=_normalize_crop_name(crop_name), device=device)


def _read_adapter_meta(adapter_dir: Path) -> Dict[str, Any]:
    return read_json_dict(adapter_dir / "adapter_meta.json")


def _summary_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    class_to_idx = meta.get("class_to_idx", {})
    ordered_classes = [
        name
        for name, _ in sorted(
            ((str(name), int(index)) for name, index in dict(class_to_idx).items()),
            key=lambda item: item[1],
        )
    ]
    return {
        "backbone_model_name": str(dict(meta.get("backbone", {})).get("model_name", "")),
        "class_names": ordered_classes,
        "class_count": len(ordered_classes),
        "fusion": dict(meta.get("fusion", {})),
        "target_modules_resolved": [str(item) for item in list(meta.get("target_modules_resolved", []))],
        "ood_calibration_version": int(dict(meta.get("ood_calibration", {})).get("version", 0)),
        "adapter_runtime": dict(meta.get("adapter_runtime", {})),
    }


def _flatten_prediction(
    image_name: str,
    payload: Dict[str, Any],
    *,
    resolved_adapter_dir: Path,
) -> Dict[str, Any]:
    disease = dict(payload.get("disease", {}))
    ood = dict(payload.get("ood_analysis", {}))
    return {
        "image_name": str(image_name),
        "adapter_dir": str(resolved_adapter_dir),
        "status": str(payload.get("status", "")),
        "predicted_class": disease.get("name"),
        "predicted_index": disease.get("class_index"),
        "confidence": float(disease.get("confidence", 0.0)),
        "is_ood": bool(ood.get("is_ood", False)),
        "ensemble_score": float(ood.get("ensemble_score", 0.0)),
        "class_threshold": float(ood.get("class_threshold", 0.0)),
        "calibration_version": int(ood.get("calibration_version", 0)),
        "ood_analysis": ood,
        "raw_payload": payload,
    }


def load_adapter_summary(
    crop_name: str,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load an adapter and return a compact runtime + metadata summary."""
    crop_key = _normalize_crop_name(crop_name)
    resolved_dir = _resolve_adapter_dir(
        crop_key,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
    )
    meta = _read_adapter_meta(resolved_dir)
    adapter = _build_adapter(crop_key, device=device)
    adapter.load_adapter(str(resolved_dir))
    summary = dict(adapter.get_summary())
    summary.update(_summary_from_meta(meta))
    summary["resolved_adapter_dir"] = str(resolved_dir)
    return summary


def predict_single_image(
    image_path: str | Path,
    crop_name: str,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run a single direct adapter prediction for a smoke test."""
    crop_key = _normalize_crop_name(crop_name)
    resolved_dir = _resolve_adapter_dir(
        crop_key,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
    )
    adapter = _build_adapter(crop_key, device=device)
    adapter.load_adapter(str(resolved_dir))

    image_ref = Path(image_path)
    with Image.open(image_ref) as image:
        image_tensor = preprocess_image(image.convert("RGB"), target_size=_target_size(config_env))
    payload = adapter.predict_with_ood(image_tensor)
    result = _flatten_prediction(image_ref.name, payload, resolved_adapter_dir=resolved_dir)
    result["image_path"] = str(image_ref)
    return result


def _iter_image_candidates(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def predict_image_folder(
    image_dir: str | Path,
    crop_name: str,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Run direct adapter predictions for all supported images in a folder."""
    crop_key = _normalize_crop_name(crop_name)
    folder = Path(image_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"image_dir is not a directory: {folder}")

    resolved_dir = _resolve_adapter_dir(
        crop_key,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
    )
    adapter = _build_adapter(crop_key, device=device)
    adapter.load_adapter(str(resolved_dir))
    target_size = _target_size(config_env)

    rows: List[Dict[str, Any]] = []
    for image_path in _iter_image_candidates(folder):
        try:
            with Image.open(image_path) as image:
                image_tensor = preprocess_image(image.convert("RGB"), target_size=target_size)
            payload = adapter.predict_with_ood(image_tensor)
            row = _flatten_prediction(image_path.name, payload, resolved_adapter_dir=resolved_dir)
            row["image_path"] = str(image_path)
            row["error"] = ""
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            row = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "adapter_dir": str(resolved_dir),
                "status": "error",
                "predicted_class": None,
                "predicted_index": None,
                "confidence": 0.0,
                "is_ood": None,
                "ensemble_score": None,
                "class_threshold": None,
                "calibration_version": None,
                "ood_analysis": None,
                "raw_payload": None,
                "error": str(exc),
            }
        rows.append(row)
    return rows


__all__ = [
    "load_adapter_summary",
    "predict_single_image",
    "predict_image_folder",
]
