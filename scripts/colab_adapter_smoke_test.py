#!/usr/bin/env python3
"""Direct adapter smoke-test helpers for Colab notebooks."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.transforms import build_image_transform, preprocess_image
from src.shared.json_utils import read_json_dict

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_ROBUST_VIEWS = ("full_resize", "resize_pad", "center_crop")
IMAGE_MEAN_PAD_RGB = (124, 116, 104)
DEFAULT_DISCOVERY_ROOTS = (
    Path("."),
    Path("outputs"),
    Path("models"),
    Path("models") / "adapters",
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


def _read_json_if_exists(path: Path, *, strict: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json_dict(path)
    except Exception as exc:
        if strict:
            raise ValueError(f"Failed to parse JSON metadata at {path}: {exc}") from exc
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


def _infer_crop_name_from_meta(adapter_dir: Path, *, strict_metadata: bool = False) -> Optional[str]:
    meta = _read_json_if_exists(adapter_dir / "adapter_meta.json", strict=strict_metadata)
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


def _infer_crop_name_from_adapter_dir(adapter_dir: Path, *, strict_metadata: bool = False) -> Optional[str]:
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
        inferred = _extract_crop_name_from_payload(
            _read_json_if_exists(candidate, strict=strict_metadata)
        )
        if inferred is not None:
            return inferred
    inferred_from_meta = _infer_crop_name_from_meta(adapter_dir, strict_metadata=strict_metadata)
    if inferred_from_meta is not None:
        return inferred_from_meta
    return None


def _resolve_crop_name(crop_name: Optional[str], *, adapter_dir: Path) -> str:
    if crop_name is not None:
        return _normalize_crop_name(crop_name)
    inferred = _infer_crop_name_from_adapter_dir(adapter_dir, strict_metadata=True)
    if inferred is not None:
        return inferred
    raise ValueError(
        "crop_name is required when it cannot be inferred from the adapter path or crop_info.json. "
        "Set CROP_NAME manually, or pass an adapter path that already implies the crop."
    )


def _infer_run_id(adapter_dir: Path) -> str:
    parts = list(adapter_dir.parts)
    reserved = {"artifacts", "adapter_export", "continual_sd_lora_adapter", "checkpoint_state", "outputs"}
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            candidate = str(parts[idx + 1]).strip()
            if candidate and candidate not in reserved:
                return candidate
    if "telemetry" in parts:
        idx = parts.index("telemetry")
        if idx + 1 < len(parts):
            candidate = str(parts[idx + 1]).strip()
            if candidate and candidate not in reserved:
                return candidate
    return ""


def _adapter_source_rank(adapter_dir: Path) -> int:
    normalized = adapter_dir.as_posix().lower()
    if "/runs/" in normalized and "/outputs/colab_notebook_training/continual_sd_lora_adapter" in normalized:
        return 0
    if "/runs/" in normalized and "/checkpoint_state/artifacts/adapter_export/continual_sd_lora_adapter" in normalized:
        return 4
    if "/runs/" in normalized and "/telemetry/artifacts/adapter_export/continual_sd_lora_adapter" in normalized:
        return 2
    if "/runs/" in normalized and "/artifacts/adapter_export/continual_sd_lora_adapter" in normalized:
        return 1
    if "/telemetry/" in normalized and "/artifacts/adapter_export/continual_sd_lora_adapter" in normalized:
        return 3
    if "/models/adapters/" in normalized and normalized.endswith("/continual_sd_lora_adapter"):
        return 5
    if normalized.endswith("/outputs/colab_notebook_training/continual_sd_lora_adapter"):
        return 6
    return 7


def _candidate_identity_key(*, adapter_dir: Path, crop_name: Optional[str], run_id: str) -> str:
    if run_id:
        return f"run:{run_id}:{crop_name or 'unknown'}"
    return f"path:{adapter_dir.resolve()}"


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
    metadata_error: str = "",
) -> str:
    label_bits = [crop_name or "unknown-crop", f"{class_count} classes"]
    if backbone_model_name:
        label_bits.append(backbone_model_name)
    if run_id:
        label_bits.append(f"run={run_id}")
    if metadata_error:
        label_bits.append("metadata-warning")
    label_bits.append(str(adapter_dir))
    return " | ".join(label_bits)


def _iter_adapter_meta_paths(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DISCOVERY_DIR_NAMES]
        if "adapter_meta.json" in filenames:
            yield Path(dirpath) / "adapter_meta.json"


def _normalize_search_roots(search_roots: Sequence[str | Path]) -> List[Path]:
    resolved_pairs = [(Path(raw_root), Path(raw_root).resolve(strict=False)) for raw_root in search_roots]
    kept: List[tuple[Path, Path]] = []
    for index, (root, root_key) in enumerate(resolved_pairs):
        if any(
            index != other_index and (root_key == other_key or root_key.is_relative_to(other_key))
            for other_index, (_other_root, other_key) in enumerate(resolved_pairs)
        ):
            continue
        if any(root_key == existing_key for _existing_root, existing_key in kept):
            continue
        kept.append((root, root_key))
    return [root for root, _root_key in kept]


def discover_adapter_candidates(
    search_roots: Optional[Sequence[str | Path]] = None,
    *,
    crop_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search one or more roots for adapter bundles and return selection-ready metadata."""
    requested_crop = _normalize_crop_name(crop_name) if crop_name else None
    roots = _normalize_search_roots(search_roots or DEFAULT_DISCOVERY_ROOTS)
    candidates_by_identity: Dict[str, Dict[str, Any]] = {}
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

            metadata_error = ""
            try:
                inferred_crop = _infer_crop_name_from_adapter_dir(adapter_dir, strict_metadata=True)
            except ValueError as exc:
                inferred_crop = None
                metadata_error = str(exc)
            if requested_crop is not None and inferred_crop not in {None, requested_crop}:
                continue

            meta = _read_adapter_meta(adapter_dir)
            meta_summary = _summary_from_meta(meta)
            candidate_crop = inferred_crop or requested_crop
            run_id = _infer_run_id(adapter_dir)
            identity_key = _candidate_identity_key(
                adapter_dir=adapter_dir,
                crop_name=candidate_crop,
                run_id=run_id,
            )
            candidate = {
                "adapter_dir": str(adapter_dir),
                "crop_name": candidate_crop,
                "backbone_model_name": meta_summary["backbone_model_name"],
                "class_count": meta_summary["class_count"],
                "class_names": list(meta_summary["class_names"]),
                "fusion": dict(meta_summary["fusion"]),
                "target_modules_resolved": list(meta_summary["target_modules_resolved"]),
                "ood_calibration_version": int(meta_summary["ood_calibration_version"]),
                "run_id": run_id,
                "metadata_error": metadata_error,
            }
            existing = candidates_by_identity.get(identity_key)
            if existing is not None:
                existing_rank = _adapter_source_rank(Path(str(existing.get("adapter_dir", ""))))
                candidate_rank = _adapter_source_rank(adapter_dir)
                if (candidate_rank, len(str(adapter_dir))) >= (
                    existing_rank,
                    len(str(existing.get("adapter_dir", ""))),
                ):
                    continue
            candidates_by_identity[identity_key] = candidate

    candidates = sorted(
        candidates_by_identity.values(),
        key=lambda item: (
            str(item.get("crop_name") or ""),
            str(item.get("run_id") or ""),
            _adapter_source_rank(Path(str(item.get("adapter_dir", "")))),
            str(item.get("adapter_dir", "")),
        ),
    )
    for candidate in candidates:
        adapter_dir = Path(str(candidate["adapter_dir"]))
        candidate["display_name"] = _candidate_label(
            adapter_dir=adapter_dir,
            crop_name=candidate.get("crop_name"),
            backbone_model_name=str(candidate.get("backbone_model_name", "")),
            class_count=int(candidate.get("class_count", 0)),
            run_id=str(candidate.get("run_id", "")),
            metadata_error=str(candidate.get("metadata_error", "")),
        )
    return candidates


def _flatten_prediction(
    image_name: str,
    payload: Dict[str, Any],
    *,
    resolved_adapter_dir: Path,
    view_name: Optional[str] = None,
) -> Dict[str, Any]:
    disease = dict(payload.get("disease", {}))
    ood = dict(payload.get("ood_analysis", {}))
    row = {
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
    if view_name is not None:
        row["view_name"] = str(view_name)
    return row


def _normalize_pil_image(image: Image.Image, *, target_size: int) -> Any:
    transform = build_image_transform(int(target_size), training=False)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image)


def _prepare_view_tensor(image: Image.Image, *, target_size: int, view_name: str) -> Any:
    rgb_image = image.convert("RGB")
    if view_name == "full_resize":
        return preprocess_image(rgb_image, target_size=target_size)

    if view_name == "resize_pad":
        width, height = rgb_image.size
        longest = max(width, height, 1)
        scale = float(target_size) / float(longest)
        resized = rgb_image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (int(target_size), int(target_size)), IMAGE_MEAN_PAD_RGB)
        paste_x = (int(target_size) - resized.width) // 2
        paste_y = (int(target_size) - resized.height) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return _normalize_pil_image(canvas, target_size=int(target_size))

    if view_name == "center_crop":
        width, height = rgb_image.size
        shortest = max(1, min(width, height))
        scale = float(target_size) / float(shortest)
        resized = rgb_image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        left = max(0, (resized.width - int(target_size)) // 2)
        top = max(0, (resized.height - int(target_size)) // 2)
        cropped = ImageOps.crop(
            resized,
            border=(
                left,
                top,
                max(0, resized.width - int(target_size) - left),
                max(0, resized.height - int(target_size) - top),
            ),
        )
        if cropped.size != (int(target_size), int(target_size)):
            cropped = ImageOps.fit(
                resized,
                (int(target_size), int(target_size)),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
        return _normalize_pil_image(cropped, target_size=int(target_size))

    raise ValueError(
        f"Unsupported robust smoke view '{view_name}'. Expected one of: {', '.join(DEFAULT_ROBUST_VIEWS)}."
    )


def _view_error_row(
    image_name: str,
    *,
    resolved_adapter_dir: Path,
    view_name: str,
    error: Exception,
) -> Dict[str, Any]:
    row = _error_row(Path(image_name), resolved_adapter_dir=resolved_adapter_dir, error=error)
    row["view_name"] = str(view_name)
    return row


def _predict_image_view(
    image_name: str,
    image: Image.Image,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    resolved_adapter_dir: Path,
    view_name: str,
) -> Dict[str, Any]:
    image_tensor = _prepare_view_tensor(image, target_size=target_size, view_name=view_name)
    payload = adapter.predict_with_ood(image_tensor)
    return _flatten_prediction(
        image_name,
        payload,
        resolved_adapter_dir=resolved_adapter_dir,
        view_name=view_name,
    )


def _normalize_requested_views(robust_views: Optional[Sequence[str]]) -> List[str]:
    raw_views = list(robust_views or DEFAULT_ROBUST_VIEWS)
    if not raw_views:
        raise ValueError("robust_views must contain at least one view name when robust mode is enabled.")
    normalized: List[str] = []
    seen: set[str] = set()
    for view_name in raw_views:
        normalized_name = str(view_name).strip().lower()
        if not normalized_name:
            continue
        if normalized_name in seen:
            continue
        if normalized_name not in DEFAULT_ROBUST_VIEWS:
            raise ValueError(
                f"Unsupported robust smoke view '{normalized_name}'. "
                f"Expected one of: {', '.join(DEFAULT_ROBUST_VIEWS)}."
            )
        seen.add(normalized_name)
        normalized.append(normalized_name)
    if not normalized:
        raise ValueError("robust_views must contain at least one supported view name.")
    return normalized


def _view_consistency_summary(views: Sequence[Dict[str, Any]], *, primary_view: str) -> Dict[str, Any]:
    successful = [row for row in views if row.get("status") != "error"]
    successful_names = [str(row.get("view_name", "")) for row in successful if row.get("view_name")]
    failed_names = [str(row.get("view_name", "")) for row in views if row.get("status") == "error" and row.get("view_name")]
    predicted_classes = {
        str(row.get("view_name", "")): row.get("predicted_class")
        for row in successful
        if row.get("view_name")
    }
    ood_votes = {
        str(row.get("view_name", "")): row.get("is_ood")
        for row in successful
        if row.get("view_name")
    }
    confidences = [
        float(row.get("confidence", 0.0))
        for row in successful
        if row.get("confidence") is not None
    ]
    warning_codes: List[str] = []
    distinct_classes = {value for value in predicted_classes.values() if value is not None}
    if len(distinct_classes) > 1:
        warning_codes.append("view_class_disagreement")
    distinct_ood = {value for value in ood_votes.values() if value is not None}
    if len(distinct_ood) > 1:
        warning_codes.append("view_ood_disagreement")
    if confidences:
        confidence_min = min(confidences)
        confidence_max = max(confidences)
        confidence_spread = confidence_max - confidence_min
        if confidence_spread >= 0.20:
            warning_codes.append("view_confidence_spread_high")
    else:
        confidence_min = None
        confidence_max = None
        confidence_spread = None
    if failed_names:
        warning_codes.append("view_error_present")
    return {
        "primary_view": str(primary_view),
        "successful_views": successful_names,
        "failed_views": failed_names,
        "warning_codes": warning_codes,
        "predicted_classes": predicted_classes,
        "ood_votes": ood_votes,
        "confidence_min": confidence_min,
        "confidence_max": confidence_max,
        "confidence_spread": confidence_spread,
        "stable": len(successful_names) >= 2 and not warning_codes,
    }


def _uncertainty_diagnostics(
    row: Dict[str, Any],
    *,
    view_consistency: Dict[str, Any],
) -> Dict[str, Any]:
    raw_ood_analysis = row.get("ood_analysis")
    ood_analysis = dict(raw_ood_analysis) if isinstance(raw_ood_analysis, dict) else {}
    warning_codes: List[str] = []
    status = str(row.get("status", "")).strip().lower()
    if status == "error":
        warning_codes.append("prediction_error")
    else:
        warning_codes.append("confidence_not_calibrated")
        if bool(row.get("is_ood")):
            warning_codes.append("ood_flagged")
    sure_confidence_reject = ood_analysis.get("sure_confidence_reject")
    if sure_confidence_reject is True:
        warning_codes.append("sure_confidence_reject")
    conformal_set = ood_analysis.get("conformal_set")
    conformal_set_size = ood_analysis.get("conformal_set_size")
    if conformal_set_size is None and isinstance(conformal_set, list):
        conformal_set_size = len(conformal_set)
    if isinstance(conformal_set_size, int) and conformal_set_size > 1:
        warning_codes.append("conformal_set_wide")
    if not bool(view_consistency.get("stable")):
        warning_codes.append("view_instability")

    diagnostics: Dict[str, Any] = {
        "confidence_source": "top1_softmax",
        "confidence_is_calibrated": False,
        "status": status,
        "warning_codes": warning_codes,
    }
    if row.get("error"):
        diagnostics["error"] = str(row.get("error"))
    if "candidate_scores" in ood_analysis:
        diagnostics["candidate_scores"] = dict(ood_analysis.get("candidate_scores", {}))
    if "candidate_thresholds" in ood_analysis:
        diagnostics["candidate_thresholds"] = dict(ood_analysis.get("candidate_thresholds", {}))
    if "sure_confidence_reject" in ood_analysis:
        diagnostics["sure_confidence_reject"] = bool(ood_analysis.get("sure_confidence_reject"))
    if "sure_semantic_ood" in ood_analysis:
        diagnostics["sure_semantic_ood"] = bool(ood_analysis.get("sure_semantic_ood"))
    if conformal_set is not None:
        diagnostics["conformal_set"] = list(conformal_set)
    if conformal_set_size is not None:
        diagnostics["conformal_set_size"] = int(conformal_set_size)
    return diagnostics


def _predict_single_image_robust(
    image_path: Path,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    resolved_adapter_dir: Path,
    robust_views: Sequence[str],
) -> Dict[str, Any]:
    image_name = image_path.name
    rows: List[Dict[str, Any]] = []
    with Image.open(image_path) as opened_image:
        base_image = opened_image.convert("RGB")
        for view_name in robust_views:
            try:
                rows.append(
                    _predict_image_view(
                        image_name,
                        base_image,
                        adapter=adapter,
                        target_size=target_size,
                        resolved_adapter_dir=resolved_adapter_dir,
                        view_name=view_name,
                    )
                )
            except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as exc:
                rows.append(
                    _view_error_row(
                        image_name,
                        resolved_adapter_dir=resolved_adapter_dir,
                        view_name=view_name,
                        error=exc,
                    )
                )

    primary_view = str(robust_views[0])
    primary_row = next(
        (row for row in rows if row.get("view_name") == primary_view),
        _view_error_row(
            image_name,
            resolved_adapter_dir=resolved_adapter_dir,
            view_name=primary_view,
            error=RuntimeError(f"Primary view '{primary_view}' did not produce a result."),
        ),
    )
    result = dict(primary_row)
    result["image_path"] = str(image_path)
    result["views"] = rows
    result["view_consistency"] = _view_consistency_summary(rows, primary_view=primary_view)
    result["uncertainty_diagnostics"] = _uncertainty_diagnostics(
        result,
        view_consistency=result["view_consistency"],
    )
    return result


def _predict_image_row(
    image_path: Path,
    *,
    adapter: IndependentCropAdapter,
    target_size: int,
    resolved_adapter_dir: Path,
) -> Dict[str, Any]:
    with Image.open(image_path) as image:
        row = _predict_image_view(
            image_path.name,
            image,
            adapter=adapter,
            target_size=target_size,
            resolved_adapter_dir=resolved_adapter_dir,
            view_name="full_resize",
        )
    row.pop("view_name", None)
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
    """Resolve an adapter bundle and return a compact metadata summary."""
    resolved_dir = _resolve_adapter_dir(
        crop_name,
        adapter_dir=adapter_dir,
        adapter_root=adapter_root,
        config_env=config_env,
    )
    crop_key = _resolve_crop_name(crop_name, adapter_dir=resolved_dir)
    meta = _read_adapter_meta(resolved_dir)
    meta_summary = _summary_from_meta(meta)
    class_to_idx = dict(meta.get("class_to_idx", {}))
    summary = {
        "crop_name": crop_key,
        "model_name": str(dict(meta.get("backbone", {})).get("model_name", "")),
        "engine": str(meta.get("engine", "continual_sd_lora")),
        "schema_version": str(meta.get("schema_version", "v6")),
        "is_trained": True,
        "num_classes": len(class_to_idx),
        "class_to_idx": {str(k): int(v) for k, v in class_to_idx.items()},
        "ood_calibration_version": int(dict(meta.get("ood_calibration", {})).get("version", 0)),
    }
    summary.update(meta_summary)
    summary["resolved_adapter_dir"] = str(resolved_dir)
    return summary


def predict_single_image(
    image_path: str | Path,
    crop_name: Optional[str],
    adapter_dir: Optional[str | Path] = None,
    adapter_root: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    enable_robust_smoke: bool = False,
    robust_views: Optional[Sequence[str]] = None,
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
    if enable_robust_smoke:
        return _predict_single_image_robust(
            image_ref,
            adapter=adapter,
            target_size=_target_size(config_env),
            resolved_adapter_dir=resolved_dir,
            robust_views=_normalize_requested_views(robust_views),
        )
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

