#!/usr/bin/env python3
"""Duplicate-aware grouped dataset preparation for Colab and local workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors

from scripts.colab_dataset_layout import (
    IMAGE_EXTENSIONS,
    _class_name_aliases,
    _materialize_image,
    estimate_split_counts,
    normalize_class_name,
)
from src.shared.json_utils import read_json, write_json


DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_BIOCLIP_MODEL_ID = "imageomics/bioclip-2.5-vith14"
DEFAULT_RUNTIME_ROOT = Path("data") / "prepared_runtime_datasets"
DEFAULT_ARTIFACT_ROOT = Path("outputs") / "colab_notebook_data_prep"
PHASH_SIZE = 8
PHASH_AUTO_MAX_DISTANCE = 4
PHASH_REVIEW_MAX_DISTANCE = 8
DINO_AUTO_MIN = 0.985
DINO_REVIEW_MIN = 0.965
BIOCLIP_AUTO_MIN = 0.970
BIOCLIP_REVIEW_MIN = 0.950
DINO_CROSS_CLASS_BLOCK_MIN = 0.990
BIOCLIP_CROSS_CLASS_BLOCK_MIN = 0.980
DEFAULT_NEIGHBORS = 8
SOURCE_HINT_UNKNOWN = "unknown"
QUALITY_WARN_MIN_SIZE = 224
QUALITY_CRITICAL_MIN_SIZE = 160
SYNTHETIC_HINT_KEYWORDS = (
    "aug",
    "augment",
    "augmented",
    "roboflow",
    "gan",
    "synthetic",
    "flip",
    "rot",
    "rotate",
    "bright",
    "contrast",
    "noise",
    "pca",
)


@dataclass
class ImageRecord:
    relative_path: str
    absolute_path: str
    raw_class_name: str
    normalized_class_name: str
    source_hint: str
    synthetic_hint: bool
    readable: bool
    width: int
    height: int
    blur_score: float
    brightness_mean: float
    exact_hash: str
    phash_hex: str
    excluded_reason: str = ""


@dataclass
class ReviewPair:
    pair_type: str
    class_a: str
    class_b: str
    path_a: str
    path_b: str
    exact_match: bool
    phash_distance: int
    dino_cosine: float
    bioclip_cosine: float
    decision: str
    reason: str


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _load_taxonomy_expected_classes(crop_name: str, taxonomy_path: Optional[Path]) -> set[str]:
    if taxonomy_path is None or not Path(taxonomy_path).exists():
        return set()
    payload = read_json(Path(taxonomy_path), default={}, expect_type=dict)
    if not isinstance(payload, dict):
        return set()
    crop_key = normalize_class_name(crop_name)
    crop_specific = payload.get("crop_specific_diseases", {})
    if not isinstance(crop_specific, dict):
        return set()
    expected = {
        normalize_class_name(item)
        for item in crop_specific.get(crop_key, [])
        if normalize_class_name(item)
    }
    expected.add("healthy")
    return expected


def normalize_prepared_class_name(raw_class_name: str, *, crop_name: str, expected_classes: set[str]) -> str:
    normalized = normalize_class_name(raw_class_name)
    if not expected_classes:
        return normalized
    aliases = _class_name_aliases(normalized, crop_name=normalize_class_name(crop_name))
    matches = sorted(expected_classes & aliases)
    if len(matches) == 1:
        return matches[0]
    return normalized


def _infer_source_hint(class_dir: Path, image_path: Path) -> str:
    relative = image_path.relative_to(class_dir)
    if len(relative.parts) <= 1:
        return SOURCE_HINT_UNKNOWN
    hint = normalize_class_name(relative.parts[0])
    return hint or SOURCE_HINT_UNKNOWN


def _has_synthetic_hint(path_like: str) -> bool:
    normalized = normalize_class_name(path_like)
    return any(token in normalized for token in SYNTHETIC_HINT_KEYWORDS)


def _compute_exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_phash_hex(image: Image.Image) -> str:
    img = ImageOps.grayscale(image).resize((32, 32))
    pixels = np.asarray(img, dtype=np.float32)
    dct = np.fft.fft2(pixels)
    low_freq = np.abs(dct[: PHASH_SIZE + 1, : PHASH_SIZE + 1])
    med = np.median(low_freq[1:, 1:])
    bits = low_freq > med
    packed = 0
    for bit in bits.flatten()[: PHASH_SIZE * PHASH_SIZE]:
        packed = (packed << 1) | int(bool(bit))
    width = PHASH_SIZE * PHASH_SIZE // 4
    return f"{packed:0{width}x}"


def _phash_distance(a: str, b: str) -> int:
    return int((int(a, 16) ^ int(b, 16)).bit_count())


def _compute_blur_and_brightness(image: Image.Image) -> tuple[float, float]:
    grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    brightness_mean = float(grayscale.mean() / 255.0) if grayscale.size else 0.0
    if grayscale.shape[0] < 3 or grayscale.shape[1] < 3:
        return 0.0, brightness_mean
    center = grayscale[1:-1, 1:-1]
    lap = (
        -4.0 * center
        + grayscale[:-2, 1:-1]
        + grayscale[2:, 1:-1]
        + grayscale[1:-1, :-2]
        + grayscale[1:-1, 2:]
    )
    blur_score = float(lap.var())
    return blur_score, brightness_mean


def scan_class_root_dataset(
    *,
    class_root: Path,
    crop_name: str,
    taxonomy_path: Optional[Path] = None,
) -> tuple[List[ImageRecord], Dict[str, Any]]:
    expected_classes = _load_taxonomy_expected_classes(crop_name, taxonomy_path)
    records: List[ImageRecord] = []
    normalization_report: Dict[str, Any] = {
        "crop_name": str(crop_name),
        "expected_classes": sorted(expected_classes),
        "raw_to_normalized": {},
        "unmatched_raw_classes": [],
    }

    class_dirs = sorted([path for path in class_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    for class_dir in class_dirs:
        normalized_class_name = normalize_prepared_class_name(
            class_dir.name,
            crop_name=crop_name,
            expected_classes=expected_classes,
        )
        normalization_report["raw_to_normalized"][class_dir.name] = normalized_class_name
        if expected_classes and normalized_class_name not in expected_classes:
            normalization_report["unmatched_raw_classes"].append(class_dir.name)
        for image_path in sorted(class_dir.rglob("*"), key=lambda path: str(path).lower()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_path = image_path.relative_to(class_root).as_posix()
            source_hint = _infer_source_hint(class_dir, image_path)
            synthetic_hint = _has_synthetic_hint(relative_path)
            readable = False
            width = 0
            height = 0
            blur_score = 0.0
            brightness_mean = 0.0
            exact_hash = ""
            phash_hex = "0" * (PHASH_SIZE * PHASH_SIZE // 4)
            excluded_reason = ""
            try:
                with Image.open(image_path) as raw:
                    image = ImageOps.exif_transpose(raw.convert("RGB"))
                    width, height = image.size
                    blur_score, brightness_mean = _compute_blur_and_brightness(image)
                    phash_hex = _compute_phash_hex(image)
                    readable = True
            except Exception:
                excluded_reason = "unreadable"
            try:
                exact_hash = _compute_exact_hash(image_path)
            except Exception:
                exact_hash = ""
                if not excluded_reason:
                    excluded_reason = "unhashable"
            records.append(
                ImageRecord(
                    relative_path=relative_path,
                    absolute_path=str(image_path.resolve()),
                    raw_class_name=class_dir.name,
                    normalized_class_name=normalized_class_name,
                    source_hint=source_hint,
                    synthetic_hint=synthetic_hint,
                    readable=readable,
                    width=int(width),
                    height=int(height),
                    blur_score=float(blur_score),
                    brightness_mean=float(brightness_mean),
                    exact_hash=exact_hash,
                    phash_hex=phash_hex,
                    excluded_reason=excluded_reason,
                )
            )
    normalization_report["unmatched_raw_classes"] = sorted(set(normalization_report["unmatched_raw_classes"]))
    return records, normalization_report


def _load_dinov3_components(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return processor, model


def _load_bioclip_components(model_id: str) -> tuple[Any, Any]:
    import open_clip

    hub_model_id = f"hf-hub:{model_id}" if not str(model_id).startswith("hf-hub:") else str(model_id)
    model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
    model.eval()
    return preprocess_val, model


def _encode_dinov3(paths: Sequence[Path], *, model_id: str, batch_size: int, device: str) -> np.ndarray:
    import torch

    processor, model = _load_dinov3_components(model_id)
    model.to(device)
    embeddings: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as raw:
                images.append(ImageOps.exif_transpose(raw.convert("RGB")))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch = outputs.pooler_output
            else:
                batch = outputs.last_hidden_state[:, 0]
        batch = torch.nn.functional.normalize(batch, dim=-1)
        embeddings.append(batch.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)


def _encode_bioclip(paths: Sequence[Path], *, model_id: str, batch_size: int, device: str) -> np.ndarray:
    import torch

    preprocess_val, model = _load_bioclip_components(model_id)
    model.to(device)
    embeddings: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw.convert("RGB"))
            tensors.append(preprocess_val(image))
        image_tensor = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            batch = model.encode_image(image_tensor)
        batch = torch.nn.functional.normalize(batch, dim=-1)
        embeddings.append(batch.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)


def _compute_neighbor_pairs(
    embeddings: np.ndarray,
    *,
    paths: Sequence[str],
    normalized_class_name: str,
    neighbors: int,
    model_name: str,
) -> Dict[tuple[str, str], float]:
    if embeddings.size == 0 or len(paths) < 2:
        return {}
    neigh = NearestNeighbors(
        n_neighbors=min(len(paths), max(2, int(neighbors))),
        metric="cosine",
        algorithm="brute",
    )
    neigh.fit(embeddings)
    distances, indices = neigh.kneighbors(embeddings)
    pairs: Dict[tuple[str, str], float] = {}
    for row_index, path_a in enumerate(paths):
        for distance, col_index in zip(distances[row_index][1:], indices[row_index][1:]):
            path_b = paths[int(col_index)]
            if path_a == path_b:
                continue
            pair = tuple(sorted((path_a, path_b)))
            cosine = 1.0 - float(distance)
            pairs[pair] = max(cosine, pairs.get(pair, float("-inf")))
    return pairs


def _family_targets(records: Sequence[ImageRecord]) -> Dict[str, tuple[int, int, int]]:
    totals: Dict[str, int] = defaultdict(int)
    for record in records:
        if not record.excluded_reason:
            totals[record.normalized_class_name] += 1
    return {class_name: estimate_split_counts(total) for class_name, total in totals.items()}


def _assign_splits_for_class(
    *,
    families: List[tuple[str, List[ImageRecord]]],
    targets: tuple[int, int, int],
) -> Dict[str, str]:
    if len(families) < 3:
        raise ValueError("Need at least 3 families for grouped split assignment.")
    desired = {
        "continual": int(targets[0]),
        "val": int(targets[1]),
        "test": int(targets[2]),
    }
    used = {"continual": 0, "val": 0, "test": 0}
    assignments: Dict[str, str] = {}
    ordered = sorted(families, key=lambda item: (-len(item[1]), item[0]))
    # Seed one family per split to avoid empty split artifacts.
    for split_name, (family_id, family_records) in zip(("continual", "val", "test"), ordered[:3]):
        assignments[family_id] = split_name
        used[split_name] += len(family_records)
    for family_id, family_records in ordered[3:]:
        size = len(family_records)
        deficits = {
            split_name: desired[split_name] - used[split_name]
            for split_name in ("continual", "val", "test")
        }
        preferred = sorted(
            deficits.items(),
            key=lambda item: (item[1] < 0, -item[1], used[item[0]], item[0]),
        )[0][0]
        assignments[family_id] = preferred
        used[preferred] += size
    return assignments


def build_grouped_dataset_plan(
    *,
    class_root: Path,
    crop_name: str,
    artifact_root: Path,
    taxonomy_path: Optional[Path] = None,
    dino_model_id: str = DEFAULT_DINOV3_MODEL_ID,
    bioclip_model_id: str = DEFAULT_BIOCLIP_MODEL_ID,
    device: str = "cpu",
    batch_size: int = 16,
    neighbors: int = DEFAULT_NEIGHBORS,
) -> Dict[str, Any]:
    records, normalization_report = scan_class_root_dataset(
        class_root=class_root,
        crop_name=crop_name,
        taxonomy_path=taxonomy_path,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    valid_records = [record for record in records if not record.excluded_reason]
    class_to_records: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        class_to_records[record.normalized_class_name].append(record)

    dino_scores: Dict[tuple[str, str], float] = {}
    bioclip_scores: Dict[tuple[str, str], float] = {}
    for normalized_class_name, class_records in class_to_records.items():
        paths = [Path(record.absolute_path) for record in class_records]
        path_keys = [record.relative_path for record in class_records]
        if len(paths) < 2:
            continue
        dino_embeddings = _encode_dinov3(paths, model_id=dino_model_id, batch_size=batch_size, device=device)
        bioclip_embeddings = _encode_bioclip(paths, model_id=bioclip_model_id, batch_size=batch_size, device=device)
        dino_scores.update(
            _compute_neighbor_pairs(
                dino_embeddings,
                paths=path_keys,
                normalized_class_name=normalized_class_name,
                neighbors=neighbors,
                model_name="dinov3",
            )
        )
        bioclip_scores.update(
            _compute_neighbor_pairs(
                bioclip_embeddings,
                paths=path_keys,
                normalized_class_name=normalized_class_name,
                neighbors=neighbors,
                model_name="bioclip",
            )
        )

    uf = UnionFind()
    by_hash: Dict[tuple[str, str], List[ImageRecord]] = defaultdict(list)
    review_pairs: List[ReviewPair] = []
    blocking_conflicts: List[ReviewPair] = []
    for record in valid_records:
        uf.find(record.relative_path)
        by_hash[(record.normalized_class_name, record.exact_hash)].append(record)

    for (_, _), group in by_hash.items():
        if len(group) < 2:
            continue
        base = group[0].relative_path
        for other in group[1:]:
            uf.union(base, other.relative_path)

    record_by_path = {record.relative_path: record for record in valid_records}
    relative_paths = sorted(record_by_path)
    for index, path_a in enumerate(relative_paths):
        record_a = record_by_path[path_a]
        for path_b in relative_paths[index + 1 :]:
            record_b = record_by_path[path_b]
            phash_distance = _phash_distance(record_a.phash_hex, record_b.phash_hex)
            dino_cosine = dino_scores.get((path_a, path_b), dino_scores.get((path_b, path_a), float("-inf")))
            bioclip_cosine = bioclip_scores.get(
                (path_a, path_b), bioclip_scores.get((path_b, path_a), float("-inf"))
            )
            exact_match = bool(record_a.exact_hash and record_a.exact_hash == record_b.exact_hash)
            same_class = record_a.normalized_class_name == record_b.normalized_class_name

            if not same_class and (
                exact_match
                or phash_distance <= PHASH_AUTO_MAX_DISTANCE
                or (dino_cosine >= DINO_CROSS_CLASS_BLOCK_MIN and bioclip_cosine >= BIOCLIP_CROSS_CLASS_BLOCK_MIN)
            ):
                blocking_conflicts.append(
                    ReviewPair(
                        pair_type="cross_class_conflict",
                        class_a=record_a.normalized_class_name,
                        class_b=record_b.normalized_class_name,
                        path_a=path_a,
                        path_b=path_b,
                        exact_match=exact_match,
                        phash_distance=phash_distance,
                        dino_cosine=float(dino_cosine if math.isfinite(dino_cosine) else -1.0),
                        bioclip_cosine=float(bioclip_cosine if math.isfinite(bioclip_cosine) else -1.0),
                        decision="block",
                        reason="cross-class duplicate or near-duplicate",
                    )
                )
                continue

            if not same_class:
                continue

            should_auto_merge = (
                exact_match
                or phash_distance <= PHASH_AUTO_MAX_DISTANCE
                or (dino_cosine >= DINO_AUTO_MIN and bioclip_cosine >= BIOCLIP_AUTO_MIN)
            )
            should_review = (
                phash_distance <= PHASH_REVIEW_MAX_DISTANCE
                or dino_cosine >= DINO_REVIEW_MIN
                or bioclip_cosine >= BIOCLIP_REVIEW_MIN
            )
            if should_auto_merge:
                uf.union(path_a, path_b)
            elif should_review:
                review_pairs.append(
                    ReviewPair(
                        pair_type="same_class_review",
                        class_a=record_a.normalized_class_name,
                        class_b=record_b.normalized_class_name,
                        path_a=path_a,
                        path_b=path_b,
                        exact_match=exact_match,
                        phash_distance=phash_distance,
                        dino_cosine=float(dino_cosine if math.isfinite(dino_cosine) else -1.0),
                        bioclip_cosine=float(bioclip_cosine if math.isfinite(bioclip_cosine) else -1.0),
                        decision="review",
                        reason="borderline same-class similarity",
                    )
                )

    families: Dict[tuple[str, str], List[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        root_id = uf.find(record.relative_path)
        families[(record.normalized_class_name, root_id)].append(record)

    split_targets = _family_targets(valid_records)
    family_assignments: Dict[str, str] = {}
    class_health: Dict[str, Any] = {}
    blocking_issues: List[str] = []
    for class_name, target in split_targets.items():
        class_families = [
            (family_root, items)
            for (family_class, family_root), items in families.items()
            if family_class == class_name
        ]
        class_health[class_name] = {
            "image_count": sum(len(items) for _, items in class_families),
            "family_count": len(class_families),
            "targets": {"continual": target[0], "val": target[1], "test": target[2]},
            "synthetic_hint_count": sum(
                1 for _, items in class_families for item in items if item.synthetic_hint
            ),
        }
        if len(class_families) < 3:
            blocking_issues.append(
                f"Class '{class_name}' has only {len(class_families)} independent family/families after grouping."
            )
            continue
        assignments = _assign_splits_for_class(families=class_families, targets=target)
        family_assignments.update(assignments)

    manifest_rows: List[Dict[str, Any]] = []
    for (class_name, family_root), items in sorted(families.items(), key=lambda item: (item[0][0], item[0][1])):
        family_id = f"{class_name}__{family_root[:12]}"
        split_name = family_assignments.get(family_root, "")
        for item in sorted(items, key=lambda record: record.relative_path):
            manifest_rows.append(
                {
                    "relative_path": item.relative_path,
                    "raw_class_name": item.raw_class_name,
                    "normalized_class_name": item.normalized_class_name,
                    "source_hint": item.source_hint,
                    "synthetic_hint": item.synthetic_hint,
                    "width": item.width,
                    "height": item.height,
                    "blur_score": round(item.blur_score, 4),
                    "brightness_mean": round(item.brightness_mean, 4),
                    "exact_hash": item.exact_hash,
                    "phash_hex": item.phash_hex,
                    "family_id": family_id,
                    "split": split_name,
                }
            )

    summary = {
        "schema_version": "v1_grouped_data_prep",
        "crop_name": str(crop_name),
        "source_root": str(class_root.resolve()),
        "summary": {
            "total_images": len(records),
            "readable_images": len(valid_records),
            "excluded_images": len([record for record in records if record.excluded_reason]),
            "same_class_review_pairs": len(review_pairs),
            "cross_class_conflicts": len(blocking_conflicts),
            "blocking_issues": len(blocking_issues),
        },
        "normalization_report": normalization_report,
        "class_health": class_health,
        "blocking_issues": blocking_issues,
        "runtime_ready": not blocking_issues and not blocking_conflicts,
        "prepared_runtime_root": str((DEFAULT_RUNTIME_ROOT / str(crop_name)).resolve()),
        "ood_handoff_checklist": {
            "status": "pending",
            "message": "Prepare a separate runtime_dataset/<crop>/ood tree after ID-side prep completes.",
        },
    }

    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "prep_summary.json", summary, ensure_ascii=False)
    write_json(artifact_root / "label_normalization_report.json", normalization_report, ensure_ascii=False)
    write_json(artifact_root / "class_health_report.json", class_health, ensure_ascii=False)
    write_json(
        artifact_root / "proposed_split_manifest.json",
        {
            "schema_version": "v1_grouped_split_manifest",
            "crop_name": str(crop_name),
            "source_root": str(class_root.resolve()),
            "blocking_issues": list(blocking_issues),
            "runtime_ready": bool(summary["runtime_ready"]),
            "rows": manifest_rows,
        },
        ensure_ascii=False,
    )
    write_json(
        artifact_root / "ood_handoff_checklist.json",
        summary["ood_handoff_checklist"],
        ensure_ascii=False,
    )
    _write_csv(artifact_root / "dataset_manifest.csv", [asdict(record) for record in records])
    _write_csv(artifact_root / "family_manifest.csv", manifest_rows)
    _write_csv(artifact_root / "same_class_review_candidates.csv", [asdict(pair) for pair in review_pairs])
    _write_csv(artifact_root / "cross_class_conflicts.csv", [asdict(pair) for pair in blocking_conflicts])
    _write_csv(
        artifact_root / "exact_duplicates.csv",
        [
            {
                "normalized_class_name": key[0],
                "exact_hash": key[1],
                "relative_paths": "|".join(sorted(record.relative_path for record in values)),
                "count": len(values),
            }
            for key, values in sorted(by_hash.items())
            if len(values) > 1
        ],
    )
    return summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def materialize_grouped_runtime_dataset(
    *,
    class_root: Path,
    crop_name: str,
    artifact_root: Path,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    materialization_strategy: str = "auto",
) -> Path:
    manifest = read_json(artifact_root / "proposed_split_manifest.json", default={}, expect_type=dict)
    if not isinstance(manifest, dict):
        raise RuntimeError("Grouped split manifest is missing or invalid.")
    if manifest.get("blocking_issues"):
        raise RuntimeError("Grouped split manifest contains blocking issues. Resolve them before materializing.")
    crop_root = Path(runtime_root) / str(crop_name)
    if crop_root.exists():
        shutil.rmtree(crop_root)
    crop_root.mkdir(parents=True, exist_ok=True)
    rows = list(manifest.get("rows", []))
    for row in rows:
        split_name = str(row.get("split", "")).strip()
        if split_name not in {"continual", "val", "test"}:
            continue
        relative_path = Path(str(row.get("relative_path", "")))
        class_name = str(row.get("normalized_class_name", "")).strip()
        raw_class_name = str(row.get("raw_class_name", "")).strip()
        source_path = Path(class_root) / relative_path
        try:
            destination_relative = relative_path.relative_to(raw_class_name)
        except Exception:
            destination_relative = Path(relative_path.name)
        destination_path = crop_root / split_name / class_name / destination_relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        _materialize_image(source_path, destination_path, materialization_strategy)
    write_json(
        crop_root / "split_manifest.json",
        {
            "schema_version": "v1_grouped_runtime_layout",
            "crop_name": str(crop_name),
            "source_root": str(class_root.resolve()),
            "artifact_root": str(artifact_root.resolve()),
            "split_policy": "grouped_family_80_10_10",
            "rows": rows,
        },
        ensure_ascii=False,
    )
    return Path(runtime_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a grouped runtime dataset with duplicate-aware audit.")
    parser.add_argument("--root", type=Path, required=True, help="Flat class-root dataset.")
    parser.add_argument("--crop", type=str, required=True, help="Crop name.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=f"Artifact output root (default: {DEFAULT_ARTIFACT_ROOT})",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=Path("config") / "plant_taxonomy.json",
        help="Taxonomy path used for class normalization.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Embedding device (cpu or cuda).")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS, help="Neighbors per image.")
    parser.add_argument("--materialize", action="store_true", help="Materialize runtime dataset if no blockers.")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=DEFAULT_RUNTIME_ROOT,
        help=f"Prepared runtime dataset root (default: {DEFAULT_RUNTIME_ROOT})",
    )
    args = parser.parse_args()

    summary = build_grouped_dataset_plan(
        class_root=args.root,
        crop_name=args.crop,
        artifact_root=args.artifact_root,
        taxonomy_path=args.taxonomy_path,
        device=args.device,
        batch_size=args.batch_size,
        neighbors=args.neighbors,
    )
    print(json.dumps(summary, indent=2))
    if args.materialize:
        materialize_grouped_runtime_dataset(
            class_root=args.root,
            crop_name=args.crop,
            artifact_root=args.artifact_root,
            runtime_root=args.runtime_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
