#!/usr/bin/env python3
"""Direct adapter smoke-test helpers for Colab notebooks."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image, UnidentifiedImageError

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.transforms import preprocess_image
from src.shared.json_utils import read_json_dict

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DISCOVERY_ROOTS = (
    Path("/content/drive/MyDrive/aads_ulora"),
    Path("/content/drive/MyDrive"),
)
SKIP_DISCOVERY_DIR_NAMES = {
    ".git",
    ".hg",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime_tmp",
    ".venv",
    "__pycache__",
    "node_modules",
}


@lru_cache(maxsize=None)
def _load_inference_settings(config_env: Optional[str]) -> tuple[Path, int]:
    inference_cfg = dict(dict(get_config(environment=config_env)).get("inference", {}))
    return (
        Path(str(inference_cfg.get("adapter_root", "models/adapters"))),
        int(inference_cfg.get("target_size", 224)),
    )


def _default_adapter_root(config_env: Optional[str]) -> Path:
    return _load_inference_settings(config_env)[0]


def _target_size(config_env: Optional[str]) -> int:
    return _load_inference_settings(config_env)[1]


def _normalize_crop_name(crop_name: str) -> str:
    normalized = str(crop_name).strip().lower()
    if not normalized:
        raise ValueError("crop_name must be a non-empty string")
    return normalized


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json_dict(path)
    except Exception:
        return {}


def _extract_crop_name_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    context = payload.get("context", {})
    if not isinstance(context, dict):
        context = {}

    for candidate in (
        payload.get("crop"),
        payload.get("crop_name"),
        context.get("crop"),
        context.get("crop_name"),
    ):
        if not candidate:
            continue
        try:
            return _normalize_crop_name(str(candidate))
        except ValueError:
            continue
    return None


def _infer_crop_name_from_meta(adapter_dir: Path) -> Optional[str]:
    meta = _read_json_if_exists(adapter_dir / "adapter_meta.json")
    class_to_idx = meta.get("class_to_idx", {})
    if not isinstance(class_to_idx, dict):
        return None

    candidate_counts: Dict[str, int] = {}
    for raw_name in class_to_idx.keys():
        class_name = str(raw_name).strip().lower()
        if not class_name or class_name == "healthy":
            continue
        prefix = class_name.split("_", 1)[0].strip()
        if not prefix or prefix in {"unknown", "ood"}:
            continue
        candidate_counts[prefix] = candidate_counts.get(prefix, 0) + 1

    if not candidate_counts:
        return None
    inferred = max(candidate_counts.items(), key=lambda item: (item[1], item[0]))[0]
    return _normalize_crop_name(inferred)


def _iter_crop_metadata_candidates(adapter_dir: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    bases = [adapter_dir, *list(adapter_dir.parents)[:5]]
    relative_candidates = (
        Path("crop_info.json"),
        Path("production_readiness.json"),
        Path("training") / "summary.json",
        Path("artifacts") / "crop_info.json",
        Path("artifacts") / "production_readiness.json",
        Path("artifacts") / "training" / "summary.json",
    )
    for base in bases:
        for relative in relative_candidates:
            candidate = base / relative
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _infer_crop_name_from_adapter_dir(adapter_dir: Path) -> Optional[str]:
    if adapter_dir.name == "continual_sd_lora_adapter" and adapter_dir.parent.name:
        parent_crop = adapter_dir.parent.name.strip().lower()
        if parent_crop not in {
            "adapter",
            "adapter_export",
            "artifacts",
            "colab_notebook_training",
            "models",
            "outputs",
        }:
            return parent_crop

    for candidate in _iter_crop_metadata_candidates(adapter_dir):
        inferred = _extract_crop_name_from_payload(_read_json_if_exists(candidate))
        if inferred is not None:
            return inferred
    inferred_from_meta = _infer_crop_name_from_meta(adapter_dir)
    if inferred_from_meta is not None:
        return inferred_from_meta
    return None


def _resolve_crop_name(crop_name: Optional[str], *, adapter_dir: Path) -> str:
    if crop_name is not None:
        return _normalize_crop_name(crop_name)
    inferred = _infer_crop_name_from_adapter_dir(adapter_dir)
    if inferred is not None:
        return inferred
    raise ValueError(
        "crop_name is required when it cannot be inferred from the adapter path or crop_info.json. "
        "Set CROP_NAME manually, or pass an adapter path that already implies the crop."
    )


def _infer_run_id(adapter_dir: Path) -> str:
    parts = list(adapter_dir.parts)
    if "telemetry" not in parts:
        return ""
    idx = parts.index("telemetry")
    if idx + 1 < len(parts):
        return str(parts[idx + 1])
    return ""


def _is_adapter_dir(path: Path) -> bool:
    return path.is_dir() and path.name == "continual_sd_lora_adapter" and (path / "adapter_meta.json").exists()


def _iter_explicit_adapter_dir_candidates(path: Path) -> Iterable[Path]:
    if path.name == "adapter_meta.json":
        yield path.parent

    for relative in (
        Path("."),
        Path("continual_sd_lora_adapter"),
        Path("adapter_export"),
        Path("adapter_export") / "continual_sd_lora_adapter",
        Path("artifacts") / "adapter_export",
        Path("artifacts") / "adapter_export" / "continual_sd_lora_adapter",
        Path("outputs") / "colab_notebook_training",
        Path("outputs") / "colab_notebook_training" / "continual_sd_lora_adapter",
    ):
        yield path / relative


def _iter_adapter_root_candidates(path: Path, *, crop_key: str) -> Iterable[Path]:
    yield path
    yield path / "continual_sd_lora_adapter"
    yield path / crop_key
    yield path / crop_key / "continual_sd_lora_adapter"


def _first_adapter_dir(candidates: Iterable[Path]) -> Optional[Path]:
    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if _is_adapter_dir(candidate):
            return candidate
    return None


def _resolve_adapter_dir(
    crop_name: Optional[str],
    *,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
) -> Path:
    if adapter_dir is not None:
        root = Path(adapter_dir)
        resolved = _first_adapter_dir(_iter_explicit_adapter_dir_candidates(root))
        if resolved is not None:
            return resolved
        raise FileNotFoundError(
            "Could not resolve an adapter bundle from "
            f"adapter_dir={root}. Pass the adapter asset directory, its parent export directory, "
            "or adapter_meta.json."
        )

    base_root = Path(adapter_root) if adapter_root is not None else _default_adapter_root(config_env)
    if crop_name is None:
        resolved = _first_adapter_dir((base_root, base_root / "continual_sd_lora_adapter"))
        if resolved is not None:
            return resolved
        raise ValueError(
            "crop_name is required when resolving from ADAPTER_ROOT or the configured adapter_root. "
            "Set CROP_NAME manually, or use ADAPTER_DIR to point at a specific adapter export."
        )

    crop_key = _normalize_crop_name(crop_name)
    resolved = _first_adapter_dir(_iter_adapter_root_candidates(base_root, crop_key=crop_key))
    if resolved is not None:
        return resolved
    raise FileNotFoundError(
        f"Adapter not found for crop '{crop_key}' under adapter_root={base_root}. "
        "Expected either <adapter_root>/<crop>/continual_sd_lora_adapter/, "
        "<adapter_root>/<crop>/, or a direct adapter asset directory."
    )


def _build_adapter(crop_name: str, *, device: str) -> IndependentCropAdapter:
    return IndependentCropAdapter(crop_name=_normalize_crop_name(crop_name), device=device)


def _load_adapter_context(
    crop_name: Optional[str],
    *,
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> tuple[Path, str, IndependentCropAdapter]:
    resolved_dir = _resolve_adapter_dir(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
    )
    crop_key = _resolve_crop_name(crop_name, adapter_dir=resolved_dir)
    adapter = _build_adapter(crop_key, device=device)
    adapter.load_adapter(str(resolved_dir))
    return resolved_dir, crop_key, adapter


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


def _candidate_label(
    *,
    adapter_dir: Path,
    crop_name: Optional[str],
    backbone_model_name: str,
    class_count: int,
    run_id: str,
) -> str:
    label_bits = [crop_name or "unknown-crop", f"{class_count} classes"]
    if backbone_model_name:
        label_bits.append(backbone_model_name)
    if run_id:
        label_bits.append(f"run={run_id}")
    label_bits.append(str(adapter_dir))
    return " | ".join(label_bits)


def _iter_adapter_meta_paths(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DISCOVERY_DIR_NAMES]
        if "adapter_meta.json" in filenames:
            yield Path(dirpath) / "adapter_meta.json"


def discover_adapter_candidates(
    search_roots: Optional[Sequence[str | Path]] = None,
    *,
    crop_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search one or more roots for adapter bundles and return selection-ready metadata."""
    requested_crop = _normalize_crop_name(crop_name) if crop_name else None
    roots = [Path(root) for root in (search_roots or DEFAULT_DISCOVERY_ROOTS)]
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for root in roots:
        for meta_path in sorted(_iter_adapter_meta_paths(root)):
            adapter_dir = meta_path.parent
            if not _is_adapter_dir(adapter_dir):
                continue
            adapter_key = str(adapter_dir.resolve())
            if adapter_key in seen:
                continue
            seen.add(adapter_key)

            inferred_crop = _infer_crop_name_from_adapter_dir(adapter_dir)
            if requested_crop is not None and inferred_crop not in {None, requested_crop}:
                continue

            meta = _read_adapter_meta(adapter_dir)
            meta_summary = _summary_from_meta(meta)
            candidate_crop = inferred_crop or requested_crop
            run_id = _infer_run_id(adapter_dir)
            candidates.append(
                {
                    "adapter_dir": str(adapter_dir),
                    "crop_name": candidate_crop,
                    "backbone_model_name": meta_summary["backbone_model_name"],
                    "class_count": meta_summary["class_count"],
                    "class_names": list(meta_summary["class_names"]),
                    "fusion": dict(meta_summary["fusion"]),
                    "target_modules_resolved": list(meta_summary["target_modules_resolved"]),
                    "ood_calibration_version": int(meta_summary["ood_calibration_version"]),
                    "run_id": run_id,
                    "display_name": _candidate_label(
                        adapter_dir=adapter_dir,
                        crop_name=candidate_crop,
                        backbone_model_name=meta_summary["backbone_model_name"],
                        class_count=int(meta_summary["class_count"]),
                        run_id=run_id,
                    ),
                }
            )
    return candidates


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
        "score_method": str(ood.get("score_method", "ensemble")),
        "primary_score": float(
            ood.get(
                "primary_score",
                ood.get("ensemble_score", 0.0),
            )
        ),
        "decision_threshold": float(
            ood.get(
                "decision_threshold",
                ood.get("class_threshold", 0.0),
            )
        ),
        "calibration_version": int(ood.get("calibration_version", 0)),
        "ood_analysis": ood,
        "raw_payload": payload,
    }


def _predict_image_row(
    image_path: Path,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    resolved_adapter_dir: Path,
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        image_tensor = preprocess_image(image.convert("RGB"), target_size=target_size)
    payload = adapter.predict_with_ood(image_tensor)
    row = _flatten_prediction(image_path.name, payload, resolved_adapter_dir=resolved_adapter_dir)
    row["image_path"] = str(image_path)
    return row


def _error_row(image_path: Path, *, resolved_adapter_dir: Path, error: Exception) -> Dict[str, Any]:
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "adapter_dir": str(resolved_adapter_dir),
        "status": "error",
        "predicted_class": None,
        "predicted_index": None,
        "confidence": 0.0,
        "is_ood": None,
        "score_method": None,
        "primary_score": None,
        "decision_threshold": None,
        "calibration_version": None,
        "ood_analysis": None,
        "raw_payload": None,
        "error": str(error),
    }


def load_adapter_summary(
    crop_name: Optional[str],
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load an adapter and return a compact runtime + metadata summary."""
    resolved_dir, crop_key, adapter = _load_adapter_context(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
        device=device,
    )
    meta = _read_adapter_meta(resolved_dir)
    summary = dict(adapter.get_summary())
    summary.update(_summary_from_meta(meta))
    summary["crop_name"] = crop_key
    summary["resolved_adapter_dir"] = str(resolved_dir)
    return summary


def predict_single_image(
    image_path: str | Path,
    crop_name: Optional[str],
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run a single direct adapter prediction for a smoke test."""
    resolved_dir, _, adapter = _load_adapter_context(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
        device=device,
    )
    image_ref = Path(image_path)
    return _predict_image_row(
        image_ref,
        adapter=adapter,
        target_size=_target_size(config_env),
        resolved_adapter_dir=resolved_dir,
    )


def _iter_image_candidates(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def predict_image_folder(
    image_dir: str | Path,
    crop_name: Optional[str],
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Run direct adapter predictions for all supported images in a folder."""
    folder = Path(image_dir)
    if not folder.is_dir():
        raise NotADirectoryError(
            f"image_dir must be a directory containing images, but got: {folder}"
        )

    resolved_dir, _, adapter = _load_adapter_context(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
        device=device,
    )
    target_size = _target_size(config_env)

    rows: List[Dict[str, Any]] = []
    for image_path in _iter_image_candidates(folder):
        try:
            row = _predict_image_row(
                image_path,
                adapter=adapter,
                target_size=target_size,
                resolved_adapter_dir=resolved_dir,
            )
            row["error"] = ""
        except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
            row = _error_row(image_path, resolved_adapter_dir=resolved_dir, error=exc)
        rows.append(row)
    return rows


__all__ = [
    "discover_adapter_candidates",
    "load_adapter_summary",
    "predict_single_image",
    "predict_image_folder",
]
