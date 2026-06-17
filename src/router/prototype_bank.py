"""Visual prototype-bank builder for router handoff evidence."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.router.taxonomy_registry import SUPPORTED_SPLITS, artifact_sha256, now_utc_timestamp, split_target_id
from src.shared.json_utils import ensure_parent, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BACKEND = "image_stats_v1"


@dataclass(frozen=True)
class ImageFeatureRecord:
    path: Path
    target_id: str
    split: str
    class_label: str
    vector: tuple[float, ...]
    sha256: str


def iter_dataset_images(
    dataset_root: Path,
    *,
    splits: Iterable[str] = SUPPORTED_SPLITS,
    include_ood: bool = False,
    max_images_per_class: int | None = None,
) -> Iterable[tuple[str, str, str, Path]]:
    if not dataset_root.exists():
        return []

    split_names = tuple(splits)
    for target_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir() and "__" in path.name):
        split_target_id(target_dir.name)
        for split_name in split_names:
            if not include_ood and split_name.lower() in {"ood", "oe"}:
                continue
            split_dir = target_dir / split_name
            if not split_dir.exists():
                continue
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                emitted = 0
                for image_path in sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
                    yield target_dir.name, split_name, class_dir.name, image_path
                    emitted += 1
                    if max_images_per_class is not None and emitted >= max_images_per_class:
                        break


def image_stats_vector(image_path: Path) -> tuple[float, ...]:
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:  # pragma: no cover - exercised only in runtimes without Pillow
        raise RuntimeError("Pillow is required to build image_stats_v1 router prototypes") from exc

    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((64, 64))
        stat = ImageStat.Stat(rgb)
        width, height = image.size
        means = [float(value) / 255.0 for value in stat.mean]
        stddev = [float(value) / 255.0 for value in stat.stddev]
        hist = rgb.histogram()
        channel_features: list[float] = []
        for channel in range(3):
            values = hist[channel * 256 : (channel + 1) * 256]
            bins = [sum(values[index : index + 16]) / float(64 * 64) for index in range(0, 256, 16)]
            channel_features.extend(bins)
        aspect_ratio = float(width) / float(height) if height else 0.0
    return tuple(round(value, 8) for value in (*means, *stddev, aspect_ratio, *channel_features))


def centroid(vectors: Iterable[tuple[float, ...]]) -> tuple[float, ...]:
    materialized = list(vectors)
    if not materialized:
        return ()
    width = len(materialized[0])
    totals = [0.0] * width
    for vector in materialized:
        if len(vector) != width:
            raise ValueError("All vectors must share the same width")
        for index, value in enumerate(vector):
            totals[index] += float(value)
    return tuple(round(total / len(materialized), 8) for total in totals)


def euclidean_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError("Vector widths differ")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def dispersion(vectors: Iterable[tuple[float, ...]], center: tuple[float, ...]) -> float:
    materialized = list(vectors)
    if not materialized or not center:
        return 0.0
    return round(sum(euclidean_distance(vector, center) for vector in materialized) / len(materialized), 8)


def _summarize_records(records: list[ImageFeatureRecord]) -> dict[str, Any]:
    by_target: dict[str, list[ImageFeatureRecord]] = defaultdict(list)
    by_class: dict[tuple[str, str], list[ImageFeatureRecord]] = defaultdict(list)
    for record in records:
        by_target[record.target_id].append(record)
        by_class[(record.target_id, record.class_label)].append(record)

    target_prototypes: dict[str, Any] = {}
    for target_id, target_records in sorted(by_target.items()):
        crop, part = split_target_id(target_id)
        target_vectors = [record.vector for record in target_records]
        target_center = centroid(target_vectors)
        split_counts: dict[str, int] = defaultdict(int)
        for record in target_records:
            split_counts[record.split] += 1
        class_labels = sorted({record.class_label for record in target_records})
        target_prototypes[target_id] = {
            "target_id": target_id,
            "crop": crop,
            "part": part,
            "sample_count": len(target_records),
            "split_counts": dict(sorted(split_counts.items())),
            "class_labels": class_labels,
            "centroid": list(target_center),
            "dispersion": dispersion(target_vectors, target_center),
        }

    class_prototypes: dict[str, Any] = {}
    for (target_id, class_label), class_records in sorted(by_class.items()):
        class_vectors = [record.vector for record in class_records]
        class_center = centroid(class_vectors)
        key = f"{target_id}::{class_label}"
        split_counts: dict[str, int] = defaultdict(int)
        for record in class_records:
            split_counts[record.split] += 1
        class_prototypes[key] = {
            "target_id": target_id,
            "class_label": class_label,
            "sample_count": len(class_records),
            "split_counts": dict(sorted(split_counts.items())),
            "centroid": list(class_center),
            "dispersion": dispersion(class_vectors, class_center),
        }

    return {
        "target_prototypes": target_prototypes,
        "class_prototypes": class_prototypes,
    }


def build_prototype_bank(
    *,
    dataset_root: Path = Path("data/prepared_runtime_datasets"),
    embedding_backend: str = DEFAULT_BACKEND,
    splits: Iterable[str] = SUPPORTED_SPLITS,
    include_ood: bool = False,
    max_images_per_class: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if embedding_backend != DEFAULT_BACKEND:
        raise ValueError(f"Unsupported prototype backend {embedding_backend!r}; only {DEFAULT_BACKEND!r} is available locally")

    records: list[ImageFeatureRecord] = []
    skipped: list[dict[str, str]] = []
    for target_id, split_name, class_label, image_path in iter_dataset_images(
        dataset_root,
        splits=splits,
        include_ood=include_ood,
        max_images_per_class=max_images_per_class,
    ):
        try:
            vector = image_stats_vector(image_path)
            digest = artifact_sha256(image_path)
        except Exception as exc:
            skipped.append({"path": str(image_path), "reason": str(exc)})
            continue
        records.append(
            ImageFeatureRecord(
                path=image_path,
                target_id=target_id,
                split=split_name,
                class_label=class_label,
                vector=vector,
                sha256=digest,
            )
        )

    summaries = _summarize_records(records)
    payload = {
        "schema_version": "router_prototype_bank.v1",
        "created_at": created_at or now_utc_timestamp(),
        "embedding_backend": embedding_backend,
        "source_roots": {
            "dataset_root": str(dataset_root),
            "splits": list(splits),
            "include_ood": include_ood,
            "max_images_per_class": max_images_per_class,
        },
        "summary": {
            "target_count": len(summaries["target_prototypes"]),
            "class_prototype_count": len(summaries["class_prototypes"]),
            "sample_count": len(records),
            "skipped_count": len(skipped),
            "targets": sorted(summaries["target_prototypes"]),
        },
        **summaries,
        "source_hashes": {
            f"{record.target_id}::{record.split}::{record.class_label}::{record.path.name}": record.sha256
            for record in records
        },
        "skipped": skipped,
    }
    return payload


def write_prototype_bank(payload: dict[str, Any], output_path: Path) -> Path:
    return write_json(output_path, payload, ensure_ascii=False, sort_keys=False)


def write_router_prototype_summary(
    *,
    output_path: Path,
    registry_payload: dict[str, Any] | None,
    prototype_payload: dict[str, Any],
) -> Path:
    registry_summary = registry_payload.get("summary", {}) if isinstance(registry_payload, dict) else {}
    prototype_summary = prototype_payload.get("summary", {})
    lines = [
        "# Router Prototype Artifact Summary",
        "",
        f"- Registry targets: {registry_summary.get('target_count', 0)}",
        f"- Registry unresolved crops: {registry_summary.get('unresolved_count', 0)}",
        f"- Prototype targets: {prototype_summary.get('target_count', 0)}",
        f"- Class prototypes: {prototype_summary.get('class_prototype_count', 0)}",
        f"- Prototype samples: {prototype_summary.get('sample_count', 0)}",
        f"- Skipped images: {prototype_summary.get('skipped_count', 0)}",
        f"- Embedding backend: {prototype_payload.get('embedding_backend')}",
        "",
        "Targets:",
    ]
    for target_id in prototype_summary.get("targets", []):
        target = prototype_payload.get("target_prototypes", {}).get(target_id, {})
        lines.append(f"- `{target_id}`: {target.get('sample_count', 0)} samples")
    resolved = ensure_parent(output_path)
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved
