"""Class-support policy and effective-number class-balanced weighting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from src.shared.json_utils import read_json

MIN_SUPPORTED_CLASS_SAMPLES = 100
CLASS_BALANCE_ELIGIBLE_MAX_SAMPLES = 200
CLASS_BALANCE_BETA = 0.9999


def normalize_class_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def compute_effective_number_weights(
    counts: Sequence[int],
    *,
    beta: float = CLASS_BALANCE_BETA,
) -> list[float]:
    resolved_counts = [int(count) for count in counts]
    if not resolved_counts:
        return []
    raw_weights: list[float] = []
    for count in resolved_counts:
        if count <= 0:
            raise ValueError("Class-balanced weights require positive class counts.")
        effective_num = 1.0 - (float(beta) ** float(count))
        raw_weights.append((1.0 - float(beta)) / effective_num)
    mean_weight = sum(raw_weights) / float(len(raw_weights))
    if mean_weight <= 0.0:
        raise ValueError("Class-balanced mean weight must be positive.")
    return [float(weight / mean_weight) for weight in raw_weights]


def _manifest_payload(crop_root: Path) -> tuple[str, Dict[str, Any]]:
    filename = "split_manifest.json"
    path = crop_root / filename
    if not path.exists():
        return "", {}
    payload = read_json(path, default={}, expect_type=dict)
    if isinstance(payload, dict):
        return filename, dict(payload)
    return "", {}


def _manifest_class_counts(crop_root: Path) -> tuple[str, Dict[str, int], list[str]]:
    filename, payload = _manifest_payload(crop_root)
    if not filename:
        return "", {}, []
    classes = payload.get("classes", [])
    if not isinstance(classes, list):
        return filename, {}, []
    counts: Dict[str, int] = {}
    skipped: list[str] = []
    for entry in classes:
        if not isinstance(entry, dict):
            continue
        class_name = normalize_class_name(entry.get("class_name", ""))
        if not class_name:
            continue
        raw_count = entry.get("image_count")
        try:
            counts[class_name] = int(raw_count)
        except (TypeError, ValueError):
            skipped.append(class_name)
    return filename, counts, skipped


def resolve_reference_class_counts(
    *,
    crop_name: str,
    data_dir: str | Path,
    detected_classes: Sequence[str],
    split_class_counts: Dict[str, Dict[str, int]],
    dataset_key: str | None = None,
) -> Dict[str, Any]:
    crop_root = Path(data_dir) / str(dataset_key or crop_name)
    filename, manifest_counts, skipped = _manifest_class_counts(crop_root)
    normalized_detected = [normalize_class_name(name) for name in list(detected_classes)]
    resolved_counts: Dict[str, int] = {}
    missing: list[str] = []
    if manifest_counts:
        for class_name, normalized in zip(detected_classes, normalized_detected):
            if normalized in manifest_counts:
                resolved_counts[str(class_name)] = int(manifest_counts[normalized])
            else:
                missing.append(str(class_name))
        return {
            "count_source": f"manifest:{filename}",
            "count_source_path": str((crop_root / filename).resolve()),
            "resolved_class_counts": resolved_counts,
            "missing_classes": missing,
            "skipped_manifest_classes": list(skipped),
        }

    raw_train_counts = {
        str(class_name): int(count)
        for class_name, count in dict(split_class_counts.get("train", {})).items()
        if int(count) > 0
    }
    train_counts = {
        normalize_class_name(class_name): int(count)
        for class_name, count in raw_train_counts.items()
    }
    if train_counts:
        for class_name, normalized in zip(detected_classes, normalized_detected):
            if normalized in train_counts:
                resolved_counts[str(class_name)] = int(train_counts[normalized])
            else:
                missing.append(str(class_name))
        return {
            "count_source": "train_split",
            "count_source_path": "",
            "resolved_class_counts": resolved_counts,
            "missing_classes": missing,
            "skipped_manifest_classes": [],
        }

    return {
        "count_source": "unavailable",
        "count_source_path": "",
        "resolved_class_counts": {},
        "missing_classes": [str(class_name) for class_name in detected_classes],
        "skipped_manifest_classes": list(skipped),
    }


def build_class_balance_runtime(
    *,
    crop_name: str,
    data_dir: str | Path,
    detected_classes: Sequence[str],
    split_class_counts: Dict[str, Dict[str, int]],
    dataset_key: str | None = None,
    beta: float = CLASS_BALANCE_BETA,
    min_supported_samples: int = MIN_SUPPORTED_CLASS_SAMPLES,
    eligible_max_samples: int = CLASS_BALANCE_ELIGIBLE_MAX_SAMPLES,
) -> Dict[str, Any]:
    resolved = resolve_reference_class_counts(
        crop_name=crop_name,
        data_dir=data_dir,
        detected_classes=detected_classes,
        split_class_counts=split_class_counts,
        dataset_key=dataset_key,
    )
    resolved_counts = {
        str(class_name): int(count)
        for class_name, count in dict(resolved.get("resolved_class_counts", {})).items()
    }
    missing_classes = [str(class_name) for class_name in list(resolved.get("missing_classes", []))]
    under_min_classes = [
        str(class_name)
        for class_name, count in resolved_counts.items()
        if int(count) < int(min_supported_samples)
    ]
    eligible_classes = [
        str(class_name)
        for class_name, count in resolved_counts.items()
        if int(min_supported_samples) <= int(count) <= int(eligible_max_samples)
    ]
    all_classes_resolved = len(resolved_counts) == len(list(detected_classes)) and not missing_classes
    active = bool(all_classes_resolved and not under_min_classes and eligible_classes)
    weights_by_class: Dict[str, float] = {}
    if active:
        ordered_counts = [int(resolved_counts[str(class_name)]) for class_name in detected_classes]
        normalized_weights = compute_effective_number_weights(ordered_counts, beta=float(beta))
        weights_by_class = {
            str(class_name): float(weight)
            for class_name, weight in zip(detected_classes, normalized_weights)
        }

    return {
        "count_source": str(resolved.get("count_source", "unavailable")),
        "count_source_path": str(resolved.get("count_source_path", "") or ""),
        "resolved_class_counts": resolved_counts,
        "missing_classes": missing_classes,
        "skipped_manifest_classes": [str(name) for name in list(resolved.get("skipped_manifest_classes", []))],
        "min_supported_samples": int(min_supported_samples),
        "eligible_range": [int(min_supported_samples), int(eligible_max_samples)],
        "beta": float(beta),
        "eligible_classes": eligible_classes,
        "under_min_classes": under_min_classes,
        "all_classes_resolved": bool(all_classes_resolved),
        "active": bool(active),
        "weights_by_class": weights_by_class,
    }


def format_under_min_class_error(class_balance_runtime: Dict[str, Any]) -> str:
    counts = dict(class_balance_runtime.get("resolved_class_counts", {}))
    under_min = [str(name) for name in list(class_balance_runtime.get("under_min_classes", []))]
    minimum = int(class_balance_runtime.get("min_supported_samples", MIN_SUPPORTED_CLASS_SAMPLES))
    details = ", ".join(f"{name}={int(counts.get(name, 0))}" for name in under_min)
    return (
        f"Supported classes below minimum reference count of {minimum}: {details}. "
        "Remove them from the supported label set before training this adapter."
    )
