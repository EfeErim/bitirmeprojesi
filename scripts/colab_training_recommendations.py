#!/usr/bin/env python3
"""Notebook 2 dataset-driven training recommendations."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict

from src.shared.json_utils import read_json
from src.training.services.class_balance import MIN_SUPPORTED_CLASS_SAMPLES, normalize_class_name

try:  # pragma: no cover - exercised indirectly in environments with torch.
    import torch
except Exception:  # pragma: no cover - keep notebook helper importable without torch.
    torch = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RUNTIME_SPLITS = ("continual", "val", "test")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _count_split(root: Path, split_name: str) -> Dict[str, int]:
    split_root = root / split_name
    if not split_root.is_dir():
        return {}
    counts: Dict[str, int] = {}
    for class_dir in sorted((item for item in split_root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        class_name = normalize_class_name(class_dir.name)
        counts[class_name] = sum(1 for item in class_dir.rglob("*") if _is_image(item))
    return counts


def _summarize_manifest_rows(rows: list[Any]) -> Dict[str, Any]:
    synthetic_hint_count = 0
    eval_quality_risk_count = 0
    eval_ineligible_count = 0
    family_assignment_counts: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("synthetic_hint")):
            synthetic_hint_count += 1
        if bool(row.get("eval_quality_risk")):
            eval_quality_risk_count += 1
        if row.get("family_eval_eligible") is False:
            eval_ineligible_count += 1
        assignment = str(row.get("family_assignment", row.get("split", "")) or "").strip().lower()
        if assignment:
            family_assignment_counts[assignment] = int(family_assignment_counts.get(assignment, 0)) + 1
    return {
        "row_count": len(rows),
        "synthetic_hint_count": synthetic_hint_count,
        "eval_quality_risk_count": eval_quality_risk_count,
        "eval_ineligible_count": eval_ineligible_count,
        "family_assignment_counts": family_assignment_counts,
    }


def _manifest_class_counts(payload: Dict[str, Any]) -> Dict[str, int]:
    classes = payload.get("classes", [])
    counts: Dict[str, int] = {}
    if isinstance(classes, list):
        for entry in classes:
            if not isinstance(entry, dict):
                continue
            class_name = normalize_class_name(entry.get("class_name", ""))
            if not class_name:
                continue
            try:
                counts[class_name] = int(entry.get("reference_image_count", entry.get("image_count")))
            except (TypeError, ValueError):
                continue
    if counts:
        return counts

    rows = payload.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            split_name = str(row.get("split", "")).strip().lower()
            if split_name not in {"continual", "val", "test"}:
                continue
            if bool(row.get("runtime_skipped")) or bool(row.get("generated_offline_augmentation")):
                continue
            if bool(row.get("synthetic_hint")):
                continue
            class_name = normalize_class_name(row.get("normalized_class_name", row.get("class_name", "")))
            if not class_name:
                continue
            counts[class_name] = int(counts.get(class_name, 0)) + 1
    return counts


def _dataset_scale_bucket(continual_images: int) -> str:
    total = int(max(0, continual_images))
    if total <= 400:
        return "tiny"
    if total <= 2000:
        return "small"
    if total <= 10000:
        return "medium"
    return "large"


def _gpu_batch_size(total_vram_gb: float | None, *, effective_device: str) -> int:
    if effective_device != "cuda" or total_vram_gb is None:
        return 8
    if total_vram_gb <= 16.0:
        return 8
    if total_vram_gb <= 24.0:
        return 16
    if total_vram_gb <= 39.0:
        return 32
    if total_vram_gb <= 59.0:
        return 64
    return 96


def _effective_batch_target(scale_bucket: str) -> int:
    return {
        "tiny": 16,
        "small": 32,
        "medium": 64,
        "large": 128,
    }.get(str(scale_bucket), 32)


def _epoch_schedule(scale_bucket: str) -> tuple[int, int]:
    return {
        "tiny": (24, 1),
        "small": (16, 1),
        "medium": (12, 2),
        "large": (8, 3),
    }.get(str(scale_bucket), (12, 2))


def _coerce_like(reference: Any, value: Any) -> Any:
    if isinstance(reference, bool):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, str):
        return str(value)
    return value


def inspect_runtime_dataset(dataset_root: Path, ood_root: Path | str | None = None) -> Dict[str, Any]:
    """Inspect the selected runtime dataset root for Notebook 2 recommendations."""

    resolved_root = Path(dataset_root).expanduser().resolve()
    split_class_counts = {split_name: _count_split(resolved_root, split_name) for split_name in RUNTIME_SPLITS}
    split_totals = {
        split_name: int(sum(class_counts.values()))
        for split_name, class_counts in split_class_counts.items()
    }
    resolved_ood_root = (
        Path(ood_root).expanduser().resolve()
        if ood_root is not None and str(ood_root).strip()
        else resolved_root / "ood"
    )
    ood_count = 0
    if resolved_ood_root.is_dir():
        ood_count = sum(1 for item in resolved_ood_root.rglob("*") if _is_image(item))
    manifest_path = resolved_root / "split_manifest.json"
    manifest_payload = (
        read_json(manifest_path, default={}, expect_type=dict)
        if manifest_path.exists()
        else {}
    )
    manifest_payload = dict(manifest_payload) if isinstance(manifest_payload, dict) else {}
    manifest_rows = list(manifest_payload.get("rows", [])) if isinstance(manifest_payload.get("rows", []), list) else []
    manifest_class_counts = _manifest_class_counts(manifest_payload)
    reference_class_counts = manifest_class_counts or dict(split_class_counts.get("continual", {}))
    class_names = sorted(
        {
            *split_class_counts.get("continual", {}).keys(),
            *split_class_counts.get("val", {}).keys(),
            *split_class_counts.get("test", {}).keys(),
        }
    )
    missing_required_splits = [split_name for split_name in RUNTIME_SPLITS if not (resolved_root / split_name).is_dir()]
    manifest_row_summary = _summarize_manifest_rows(manifest_rows)
    scale_bucket = _dataset_scale_bucket(split_totals.get("continual", 0))
    noisy_or_high_risk = bool(
        manifest_row_summary["synthetic_hint_count"] > 0 or manifest_row_summary["eval_quality_risk_count"] > 0
    )
    under_min_classes = sorted(
        class_name
        for class_name, count in reference_class_counts.items()
        if int(count) < int(MIN_SUPPORTED_CLASS_SAMPLES)
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if missing_required_splits:
        blockers.append(f"Missing required split folders: {missing_required_splits}")
    if not class_names:
        blockers.append("No class directories were found in the runtime dataset.")
    if split_totals.get("val", 0) <= 0:
        blockers.append("Validation split has zero images.")
    if split_totals.get("test", 0) <= 0:
        blockers.append("Test split has zero images.")
    if under_min_classes:
        details = ", ".join(f"{name}={reference_class_counts[name]}" for name in under_min_classes)
        blockers.append(
            "Supported classes remain below the production minimum of "
            f"{MIN_SUPPORTED_CLASS_SAMPLES} images/class: {details}. "
            "Enable ALLOW_UNDER_MIN_TRAINING manually only for a research-only run."
        )
    if not manifest_payload:
        warnings.append("split_manifest.json was not found; recommendations fall back to runtime split counts.")
    if ood_count <= 0:
        warnings.append("Real runtime OOD evidence is missing; Notebook 2 will rely on fallback held-out benchmarking.")
    if manifest_row_summary["synthetic_hint_count"] > 0:
        warnings.append(
            f"Manifest rows include {manifest_row_summary['synthetic_hint_count']} synthetic-hint item(s); keep main benchmark claims conservative."
        )
    if manifest_row_summary["eval_quality_risk_count"] > 0:
        warnings.append(
            f"Manifest rows include {manifest_row_summary['eval_quality_risk_count']} eval-quality-risk item(s); prefer cautious regularization."
        )
    return {
        "dataset_root": str(resolved_root),
        "manifest_path": str(manifest_path),
        "manifest_present": bool(manifest_payload),
        "split_presence": {
            **{split_name: (resolved_root / split_name).is_dir() for split_name in RUNTIME_SPLITS},
            "ood": resolved_ood_root.is_dir(),
        },
        "split_totals": {**split_totals, "ood": int(ood_count)},
        "ood_root": str(resolved_ood_root),
        "split_class_counts": split_class_counts,
        "manifest_class_counts": manifest_class_counts,
        "reference_class_counts": reference_class_counts,
        "class_names": class_names,
        "class_count": len(class_names),
        "dataset_scale_bucket": scale_bucket,
        "manifest_row_summary": manifest_row_summary,
        "real_ood_present": bool(ood_count > 0),
        "noisy_or_high_risk": noisy_or_high_risk,
        "under_min_classes": under_min_classes,
        "blockers": blockers,
        "warnings": warnings,
    }


def inspect_runtime_hardware(device: str) -> Dict[str, Any]:
    """Inspect the current runtime hardware for Notebook 2 recommendations."""

    requested_device = str(device or "cpu").strip().lower()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    effective_device = "cuda" if requested_device.startswith("cuda") and cuda_available else "cpu"
    cpu_count = int(os.cpu_count() or 1)
    gpu_name = ""
    total_vram_gb: float | None = None
    if effective_device == "cuda" and torch is not None:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu_name = str(properties.name)
        total_vram_gb = round(float(properties.total_memory) / float(1024**3), 2)
    warnings: list[str] = []
    if requested_device.startswith("cuda") and effective_device != "cuda":
        warnings.append("CUDA was requested but is unavailable; recommendations fall back to CPU-safe settings.")
    return {
        "requested_device": requested_device,
        "effective_device": effective_device,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "cpu_count": cpu_count,
        "warnings": warnings,
        "strong_gpu": bool(total_vram_gb is not None and total_vram_gb >= 40.0),
        "memory_constrained": bool(total_vram_gb is not None and total_vram_gb <= 16.0) or effective_device != "cuda",
    }


def recommend_notebook_training_params(
    base_params: Dict[str, Any],
    dataset_report: Dict[str, Any],
    hardware_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Recommend Notebook 2 parameter values from inspected dataset and hardware signals."""

    recommended = dict(base_params)
    reasons: Dict[str, str] = {}
    blockers = list(dataset_report.get("blockers", []))
    warnings = list(dataset_report.get("warnings", [])) + list(hardware_report.get("warnings", []))
    scale_bucket = str(dataset_report.get("dataset_scale_bucket", "small"))
    class_count = int(dataset_report.get("class_count", 0))
    continual_images = int(dict(dataset_report.get("split_totals", {})).get("continual", 0))
    noisy_or_high_risk = bool(dataset_report.get("noisy_or_high_risk", False))
    effective_device = str(hardware_report.get("effective_device", "cpu"))
    total_vram_gb = hardware_report.get("total_vram_gb")
    cpu_count = int(hardware_report.get("cpu_count", 1))

    batch_size = _gpu_batch_size(total_vram_gb if isinstance(total_vram_gb, (int, float)) else None, effective_device=effective_device)
    recommended["BATCH_SIZE"] = batch_size
    reasons["BATCH_SIZE"] = (
        "Derived from the active hardware tier "
        f"(device={effective_device}, vram_gb={total_vram_gb if total_vram_gb is not None else 'unknown'})."
    )

    target_effective_batch = _effective_batch_target(scale_bucket)
    grad_accum_steps = max(1, int(math.ceil(float(target_effective_batch) / float(max(1, batch_size)))))
    recommended["GRAD_ACCUM_STEPS"] = grad_accum_steps
    reasons["GRAD_ACCUM_STEPS"] = (
        f"Targets an effective batch size of {target_effective_batch} for a {scale_bucket} continual split "
        f"({continual_images} image(s))."
    )

    epochs, validation_every = _epoch_schedule(scale_bucket)
    recommended["EPOCHS"] = epochs
    recommended["VALIDATION_EVERY_N_EPOCHS"] = validation_every
    reasons["EPOCHS"] = f"Uses the fixed schedule for a {scale_bucket} continual split ({continual_images} image(s))."
    reasons["VALIDATION_EVERY_N_EPOCHS"] = f"Keeps validation cadence aligned with the {scale_bucket} epoch schedule."

    if noisy_or_high_risk or scale_bucket == "tiny":
        learning_rate = 1e-4
    elif scale_bucket == "small":
        learning_rate = 1.5e-4
    else:
        learning_rate = 2e-4
    recommended["LEARNING_RATE"] = learning_rate
    reasons["LEARNING_RATE"] = (
        "Lowered for tiny/high-risk datasets and kept more aggressive for medium/large clean datasets."
    )

    if scale_bucket == "tiny":
        lora_r = 8
        lora_dropout = 0.2
    elif scale_bucket == "small":
        lora_r = 16
        lora_dropout = 0.2 if noisy_or_high_risk else 0.1
    elif scale_bucket == "medium":
        lora_r = 24
        lora_dropout = 0.1 if not noisy_or_high_risk else 0.2
    elif bool(hardware_report.get("strong_gpu")) and class_count >= 8 and not noisy_or_high_risk:
        lora_r = 32
        lora_dropout = 0.05
    else:
        lora_r = 24
        lora_dropout = 0.1 if not noisy_or_high_risk else 0.2
    recommended["LORA_R"] = lora_r
    recommended["LORA_ALPHA"] = lora_r
    recommended["LORA_DROPOUT"] = lora_dropout
    reasons["LORA_R"] = (
        f"Uses the fixed rank schedule for a {scale_bucket} dataset with class_count={class_count} and strong_gpu={bool(hardware_report.get('strong_gpu'))}."
    )
    reasons["LORA_ALPHA"] = "Kept equal to the recommended LoRA rank for a predictable notebook surface."
    reasons["LORA_DROPOUT"] = "Raised on smaller/noisier datasets and relaxed only for larger clean datasets."

    if effective_device != "cuda" and cpu_count <= 4:
        num_workers = 0
    elif effective_device != "cuda":
        num_workers = min(4, max(2, cpu_count // 2))
    else:
        num_workers = min(8, max(2, cpu_count // 2))
    recommended["NUM_WORKERS"] = num_workers
    reasons["NUM_WORKERS"] = f"Derived from device={effective_device} and cpu_count={cpu_count}."

    recommended["PREFETCH"] = 0 if num_workers <= 0 else (8 if bool(hardware_report.get("strong_gpu")) else 4)
    reasons["PREFETCH"] = "Enabled only when workers are active; stronger GPU sessions get the higher notebook prefetch value."

    recommended["PIN_MEMORY"] = effective_device == "cuda"
    reasons["PIN_MEMORY"] = "Pinned host memory only helps on CUDA-backed sessions."

    recommended["USE_CACHE"] = bool(effective_device == "cuda" and not hardware_report.get("memory_constrained"))
    recommended["CACHE_TRAIN_SPLIT"] = bool(
        effective_device == "cuda"
        and not hardware_report.get("memory_constrained")
        and scale_bucket in {"tiny", "small", "medium"}
    )
    reasons["USE_CACHE"] = "Kept for stronger CUDA sessions and disabled for memory-constrained environments."
    reasons["CACHE_TRAIN_SPLIT"] = "Disabled when the runtime looks memory-constrained or the continual split is large."

    if bool(base_params.get("BER_ENABLED")) and str(base_params.get("LOSS_NAME", "")).strip().lower() != "cross_entropy":
        blockers.append("BER_ENABLED requires LOSS_NAME='cross_entropy'; the notebook will not auto-correct this combination.")

    changes: Dict[str, Dict[str, Any]] = {}
    for key, recommended_value in recommended.items():
        current_value = base_params.get(key)
        if current_value != recommended_value:
            changes[key] = {
                "current": current_value,
                "recommended": recommended_value,
                "reason": reasons.get(key, ""),
            }

    return {
        "recommended_params": recommended,
        "changes": changes,
        "change_count": len(changes),
        "has_changes": bool(changes),
        "blockers": blockers,
        "warnings": warnings,
        "scale_bucket": scale_bucket,
        "summary": {
            "continual_images": continual_images,
            "class_count": class_count,
            "real_ood_present": bool(dataset_report.get("real_ood_present")),
            "effective_device": effective_device,
            "total_vram_gb": total_vram_gb,
            "target_effective_batch": target_effective_batch,
        },
    }


def resolve_effective_notebook_params(
    base_params: Dict[str, Any],
    recommendation_report: Dict[str, Any],
    manual_overrides: Dict[str, Any],
    accepted: bool,
) -> Dict[str, Any]:
    """Resolve the final Notebook 2 parameters after recommendation confirmation."""

    resolved = dict(base_params)
    if bool(accepted):
        resolved.update(dict(recommendation_report.get("recommended_params", {})))
    for key, raw_value in dict(manual_overrides or {}).items():
        if key in resolved:
            resolved[key] = _coerce_like(resolved[key], raw_value)
        else:
            resolved[str(key)] = raw_value
    return resolved


__all__ = [
    "inspect_runtime_dataset",
    "inspect_runtime_hardware",
    "recommend_notebook_training_params",
    "resolve_effective_notebook_params",
]
