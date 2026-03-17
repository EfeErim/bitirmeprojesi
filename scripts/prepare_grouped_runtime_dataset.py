#!/usr/bin/env python3
"""Duplicate-aware grouped dataset preparation for Colab and local workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
    class_order_index: int
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
    adjacency_distance: Optional[int]
    review_rank: int
    decision: str
    reason: str
    cluster_id: str = ""
    triage_resolution: str = ""
    triage_reason: str = ""


@dataclass
class ReviewCluster:
    cluster_id: str
    normalized_class_name: str
    image_count: int
    pair_count: int
    source_hint_count: int
    source_hints: str
    synthetic_hint_count: int
    min_phash_distance: int
    max_phash_distance: int
    max_dino_cosine: float
    max_bioclip_cosine: float
    min_adjacency_distance: Optional[int]
    resolution: str
    reason: str
    relative_paths: str


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
        class_image_paths = sorted(class_dir.rglob("*"), key=lambda path: str(path).lower())
        class_order_index = 0
        for image_path in class_image_paths:
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
                    class_order_index=int(class_order_index),
                    excluded_reason=excluded_reason,
                )
            )
            class_order_index += 1
    normalization_report["unmatched_raw_classes"] = sorted(set(normalization_report["unmatched_raw_classes"]))
    return records, normalization_report


def _refresh_record_availability(records: Sequence[ImageRecord]) -> Dict[str, int]:
    excluded_counts = {
        "missing_after_scan": 0,
        "unreadable_after_scan": 0,
    }
    for record in records:
        if record.excluded_reason:
            continue
        image_path = Path(record.absolute_path)
        if not image_path.is_file():
            record.readable = False
            record.excluded_reason = "missing_after_scan"
            excluded_counts["missing_after_scan"] += 1
            continue
        try:
            with Image.open(image_path) as raw:
                _ = ImageOps.exif_transpose(raw.convert("RGB"))
        except Exception:
            record.readable = False
            record.excluded_reason = "unreadable_after_scan"
            excluded_counts["unreadable_after_scan"] += 1
    return excluded_counts


def _resolve_amp_dtype(device: str) -> Any:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _load_dinov3_components(model_id: str, *, device: str = "cpu") -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModel

    load_kwargs: Dict[str, Any] = {}
    amp_dtype = _resolve_amp_dtype(device)
    if amp_dtype is not None:
        load_kwargs["dtype"] = amp_dtype
    processor = AutoImageProcessor.from_pretrained(model_id)
    try:
        model = AutoModel.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(model_id)
    model.eval()
    return processor, model


def _autocast_context(*, device: str, amp_dtype: Any) -> Any:
    import torch

    enabled = amp_dtype is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _load_bioclip_components(model_id: str, *, device: str = "cpu") -> tuple[Any, Any]:
    import open_clip
    import torch

    hub_model_id = f"hf-hub:{model_id}" if not str(model_id).startswith("hf-hub:") else str(model_id)
    create_kwargs: Dict[str, Any] = {}
    amp_dtype = _resolve_amp_dtype(device)
    if amp_dtype is not None:
        create_kwargs["precision"] = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
    try:
        model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id, **create_kwargs)
    except TypeError:
        model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
    model.eval()
    return preprocess_val, model


def _encode_dinov3(paths: Sequence[Path], *, model_id: str, batch_size: int, device: str) -> np.ndarray:
    processor, model = _load_dinov3_components(model_id, device=device)
    return _encode_dinov3_with_components(
        paths,
        processor=processor,
        model=model,
        batch_size=batch_size,
        device=device,
        amp_dtype=_resolve_amp_dtype(device),
    )


def _encode_dinov3_with_components(
    paths: Sequence[Path],
    *,
    processor: Any,
    model: Any,
    batch_size: int,
    device: str,
    amp_dtype: Any = None,
) -> np.ndarray:
    import torch

    embeddings: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as raw:
                images.append(ImageOps.exif_transpose(raw.convert("RGB")))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        with torch.inference_mode():
            with _autocast_context(device=device, amp_dtype=amp_dtype):
                outputs = model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch = outputs.pooler_output
                else:
                    batch = outputs.last_hidden_state[:, 0]
        batch = torch.nn.functional.normalize(batch, dim=-1).to(dtype=torch.float32)
        embeddings.append(batch.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)


def _encode_bioclip(paths: Sequence[Path], *, model_id: str, batch_size: int, device: str) -> np.ndarray:
    preprocess_val, model = _load_bioclip_components(model_id, device=device)
    return _encode_bioclip_with_components(
        paths,
        preprocess_val=preprocess_val,
        model=model,
        batch_size=batch_size,
        device=device,
        amp_dtype=_resolve_amp_dtype(device),
    )


def _encode_bioclip_with_components(
    paths: Sequence[Path],
    *,
    preprocess_val: Any,
    model: Any,
    batch_size: int,
    device: str,
    amp_dtype: Any = None,
) -> np.ndarray:
    import torch

    embeddings: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw.convert("RGB"))
            tensors.append(preprocess_val(image))
        image_tensor = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            with _autocast_context(device=device, amp_dtype=amp_dtype):
                batch = model.encode_image(image_tensor)
        batch = torch.nn.functional.normalize(batch, dim=-1).to(dtype=torch.float32)
        embeddings.append(batch.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)


def _compute_neighbor_pairs(
    embeddings: np.ndarray,
    *,
    paths: Sequence[str],
    neighbors: int,
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


def _progress(progress_fn: Optional[Callable[[str], None]], message: str) -> None:
    if callable(progress_fn):
        progress_fn(str(message))


def _adjacency_distance(a: ImageRecord, b: ImageRecord) -> Optional[int]:
    if a.normalized_class_name != b.normalized_class_name:
        return None
    return abs(int(a.class_order_index) - int(b.class_order_index))


def _review_rank(a: ImageRecord, b: ImageRecord) -> int:
    distance = _adjacency_distance(a, b)
    if distance is None:
        return 10**9
    return int(distance)


def _classify_review_cluster(
    *,
    records: Sequence[ImageRecord],
    pairs: Sequence[ReviewPair],
) -> tuple[str, str]:
    cluster_size = len(records)
    source_hints = {record.source_hint for record in records if record.source_hint and record.source_hint != SOURCE_HINT_UNKNOWN}
    synthetic_count = sum(1 for record in records if record.synthetic_hint)
    min_adjacency = min(
        (pair.adjacency_distance for pair in pairs if pair.adjacency_distance is not None),
        default=None,
    )
    max_phash = max((int(pair.phash_distance) for pair in pairs), default=PHASH_REVIEW_MAX_DISTANCE + 1)
    max_dino = max((float(pair.dino_cosine) for pair in pairs), default=-1.0)
    max_bioclip = max((float(pair.bioclip_cosine) for pair in pairs), default=-1.0)

    if cluster_size > 6:
        return "manual_review", "cluster size exceeds conservative auto-resolve threshold"
    if len(source_hints) > 1:
        return "manual_review", "mixed source hints in same-class review cluster"

    strong_similarity = max_dino >= DINO_AUTO_MIN and (max_bioclip < 0 or max_bioclip >= BIOCLIP_REVIEW_MIN)
    adjacency_supported = min_adjacency is not None and int(min_adjacency) <= 2 and cluster_size <= 3
    synthetic_supported = synthetic_count > 0 and cluster_size <= 4

    if synthetic_supported:
        return "auto_resolve", "small same-class cluster has synthetic lineage hints"
    if adjacency_supported:
        return "auto_resolve", "very small same-class cluster is adjacency-backed"
    if max_phash > PHASH_REVIEW_MAX_DISTANCE:
        return "manual_review", "perceptual hash disagreement is too large"

    if strong_similarity and cluster_size <= 4:
        return "auto_resolve", "same-class low-risk cluster with strong lineage/similarity support"
    return "manual_review", "same-class cluster lacks strong heuristic support for auto-resolution"


def _triage_review_clusters(
    *,
    review_pairs: Sequence[ReviewPair],
    record_lookup: Dict[str, ImageRecord],
    uf: UnionFind,
) -> tuple[List[ReviewPair], List[ReviewCluster], List[ReviewCluster]]:
    if not review_pairs:
        return [], [], []

    cluster_uf = UnionFind()
    for pair in review_pairs:
        key_a = f"{pair.class_a}::{pair.path_a}"
        key_b = f"{pair.class_b}::{pair.path_b}"
        cluster_uf.union(key_a, key_b)

    cluster_pairs: Dict[str, List[ReviewPair]] = defaultdict(list)
    cluster_paths: Dict[str, set[str]] = defaultdict(set)
    for pair in review_pairs:
        root = cluster_uf.find(f"{pair.class_a}::{pair.path_a}")
        cluster_pairs[root].append(pair)
        cluster_paths[root].add(pair.path_a)
        cluster_paths[root].add(pair.path_b)

    cluster_rows: List[ReviewCluster] = []
    unresolved_pairs: List[ReviewPair] = []
    auto_resolved_clusters: List[ReviewCluster] = []

    for cluster_index, root in enumerate(sorted(cluster_pairs.keys()), start=1):
        pairs = cluster_pairs[root]
        class_name = pairs[0].class_a
        records = [record_lookup[path] for path in sorted(cluster_paths[root])]
        resolution, reason = _classify_review_cluster(records=records, pairs=pairs)
        cluster_id = f"{class_name}__review_cluster_{cluster_index:04d}"
        source_hints = sorted(
            {record.source_hint for record in records if record.source_hint and record.source_hint != SOURCE_HINT_UNKNOWN}
        )
        min_adjacency = min(
            (pair.adjacency_distance for pair in pairs if pair.adjacency_distance is not None),
            default=None,
        )
        cluster_row = ReviewCluster(
            cluster_id=cluster_id,
            normalized_class_name=class_name,
            image_count=len(records),
            pair_count=len(pairs),
            source_hint_count=len(source_hints),
            source_hints="|".join(source_hints),
            synthetic_hint_count=sum(1 for record in records if record.synthetic_hint),
            min_phash_distance=min(int(pair.phash_distance) for pair in pairs),
            max_phash_distance=max(int(pair.phash_distance) for pair in pairs),
            max_dino_cosine=max(float(pair.dino_cosine) for pair in pairs),
            max_bioclip_cosine=max(float(pair.bioclip_cosine) for pair in pairs),
            min_adjacency_distance=min_adjacency,
            resolution=resolution,
            reason=reason,
            relative_paths="|".join(record.relative_path for record in records),
        )
        cluster_rows.append(cluster_row)

        if resolution == "auto_resolve":
            base = records[0].relative_path
            for record in records[1:]:
                uf.union(base, record.relative_path)
            auto_resolved_clusters.append(cluster_row)
            for pair in pairs:
                pair.cluster_id = cluster_id
                pair.triage_resolution = resolution
                pair.triage_reason = reason
        else:
            for pair in pairs:
                pair.cluster_id = cluster_id
                pair.triage_resolution = resolution
                pair.triage_reason = reason
                unresolved_pairs.append(pair)

    high_risk_clusters = [row for row in cluster_rows if row.resolution != "auto_resolve"]
    return unresolved_pairs, cluster_rows, auto_resolved_clusters


def _build_hash_groups(records: Sequence[ImageRecord], *, key_name: str) -> Dict[str, List[ImageRecord]]:
    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        key = str(getattr(record, key_name, "") or "").strip()
        if key:
            groups[key].append(record)
    return groups


def _register_cross_class_conflicts(
    *,
    groups: Dict[str, List[ImageRecord]],
    conflict_reason: str,
    blocking_conflicts: List[ReviewPair],
    seen_pairs: set[tuple[str, str]],
) -> None:
    for _, values in groups.items():
        grouped_by_class: Dict[str, List[ImageRecord]] = defaultdict(list)
        for record in values:
            grouped_by_class[record.normalized_class_name].append(record)
        if len(grouped_by_class) < 2:
            continue
        for records_a, records_b in itertools.combinations(grouped_by_class.values(), 2):
            for record_a in records_a:
                for record_b in records_b:
                    pair = tuple(sorted((record_a.relative_path, record_b.relative_path)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    blocking_conflicts.append(
                        ReviewPair(
                            pair_type="cross_class_conflict",
                            class_a=record_a.normalized_class_name,
                            class_b=record_b.normalized_class_name,
                            path_a=record_a.relative_path,
                            path_b=record_b.relative_path,
                            exact_match=record_a.exact_hash == record_b.exact_hash,
                            phash_distance=_phash_distance(record_a.phash_hex, record_b.phash_hex),
                            dino_cosine=-1.0,
                            bioclip_cosine=-1.0,
                            adjacency_distance=None,
                            review_rank=10**9,
                            decision="block",
                            reason=conflict_reason,
                        )
                    )


def _compute_pair_similarity(
    embeddings: np.ndarray,
    *,
    path_keys: Sequence[str],
    pairs: Iterable[tuple[str, str]],
) -> Dict[tuple[str, str], float]:
    if embeddings.size == 0:
        return {}
    index_by_path = {path: idx for idx, path in enumerate(path_keys)}
    similarities: Dict[tuple[str, str], float] = {}
    for path_a, path_b in pairs:
        idx_a = index_by_path.get(path_a)
        idx_b = index_by_path.get(path_b)
        if idx_a is None or idx_b is None:
            continue
        score = float(np.dot(embeddings[idx_a], embeddings[idx_b]))
        similarities[tuple(sorted((path_a, path_b)))] = score
    return similarities


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
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    import gc

    import torch

    amp_dtype = _resolve_amp_dtype(device)
    effective_batch_size = max(1, int(batch_size))

    _progress(progress_fn, "Scanning class-root dataset and computing hashes.")
    records, normalization_report = scan_class_root_dataset(
        class_root=class_root,
        crop_name=crop_name,
        taxonomy_path=taxonomy_path,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    availability_exclusions = _refresh_record_availability(records)
    if availability_exclusions["missing_after_scan"]:
        _progress(
            progress_fn,
            f"Excluded {availability_exclusions['missing_after_scan']} image(s) that disappeared after the initial scan.",
        )
    if availability_exclusions["unreadable_after_scan"]:
        _progress(
            progress_fn,
            f"Excluded {availability_exclusions['unreadable_after_scan']} image(s) that became unreadable after the initial scan.",
        )
    valid_records = [record for record in records if not record.excluded_reason]
    class_to_records: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        class_to_records[record.normalized_class_name].append(record)

    dino_scores: Dict[tuple[str, str], float] = {}
    bioclip_scores: Dict[tuple[str, str], float] = {}
    review_pairs: List[ReviewPair] = []
    blocking_conflicts: List[ReviewPair] = []
    class_dino_pairs_by_class: Dict[str, Dict[tuple[str, str], float]] = {}
    seen_blocking_pairs: set[tuple[str, str]] = set()
    global_exact_hash_groups = _build_hash_groups(valid_records, key_name="exact_hash")
    global_phash_groups = _build_hash_groups(valid_records, key_name="phash_hex")
    _register_cross_class_conflicts(
        groups=global_exact_hash_groups,
        conflict_reason="cross-class exact duplicate",
        blocking_conflicts=blocking_conflicts,
        seen_pairs=seen_blocking_pairs,
    )
    _register_cross_class_conflicts(
        groups=global_phash_groups,
        conflict_reason="cross-class identical perceptual hash",
        blocking_conflicts=blocking_conflicts,
        seen_pairs=seen_blocking_pairs,
    )

    uf = UnionFind()
    for record in valid_records:
        uf.find(record.relative_path)

    by_hash: Dict[tuple[str, str], List[ImageRecord]] = defaultdict(list)
    for record in valid_records:
        by_hash[(record.normalized_class_name, record.exact_hash)].append(record)
    for (_, _), group in by_hash.items():
        if len(group) < 2:
            continue
        base = group[0].relative_path
        for other in group[1:]:
            uf.union(base, other.relative_path)

    _progress(progress_fn, "Loading DINOv3 once for the full audit run.")
    dino_processor, dino_model = _load_dinov3_components(dino_model_id, device=device)
    dino_model.to(device)
    for normalized_class_name, class_records in class_to_records.items():
        _progress(progress_fn, f"Processing class '{normalized_class_name}' ({len(class_records)} readable images).")
        paths = [Path(record.absolute_path) for record in class_records]
        path_keys = [record.relative_path for record in class_records]
        if len(paths) < 2:
            continue
        _progress(progress_fn, f"Encoding DINOv3 embeddings for class '{normalized_class_name}'.")
        dino_embeddings = _encode_dinov3_with_components(
            paths,
            processor=dino_processor,
            model=dino_model,
            batch_size=effective_batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )
        class_dino_pairs = _compute_neighbor_pairs(
            dino_embeddings,
            paths=path_keys,
            neighbors=neighbors,
        )
        dino_scores.update(class_dino_pairs)
        class_dino_pairs_by_class[normalized_class_name] = class_dino_pairs

    dino_model.to("cpu")
    del dino_model
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    bioclip_needed = False
    for normalized_class_name, class_records in class_to_records.items():
        class_dino_pairs = class_dino_pairs_by_class.get(normalized_class_name, {})
        if not class_dino_pairs:
            continue
        record_by_path = {record.relative_path: record for record in class_records}
        for pair, dino_cosine in class_dino_pairs.items():
            record_a = record_by_path[pair[0]]
            record_b = record_by_path[pair[1]]
            phash_distance = _phash_distance(record_a.phash_hex, record_b.phash_hex)
            if dino_cosine >= DINO_REVIEW_MIN and phash_distance > PHASH_AUTO_MAX_DISTANCE:
                bioclip_needed = True
                break
        if bioclip_needed:
            break

    bioclip_preprocess = None
    bioclip_model = None
    if bioclip_needed:
        _progress(progress_fn, "Loading BioCLIP once for candidate refinement.")
        bioclip_preprocess, bioclip_model = _load_bioclip_components(bioclip_model_id, device=device)
        bioclip_model.to(device)

    for normalized_class_name, class_records in class_to_records.items():
        class_dino_pairs = class_dino_pairs_by_class.get(normalized_class_name, {})
        if not class_dino_pairs:
            continue
        record_by_path = {record.relative_path: record for record in class_records}
        bioclip_candidate_pairs = []
        for pair, dino_cosine in class_dino_pairs.items():
            record_a = record_by_path[pair[0]]
            record_b = record_by_path[pair[1]]
            phash_distance = _phash_distance(record_a.phash_hex, record_b.phash_hex)
            if dino_cosine >= DINO_REVIEW_MIN and phash_distance > PHASH_AUTO_MAX_DISTANCE:
                bioclip_candidate_pairs.append(pair)

        unique_bioclip_paths = sorted({path for pair in bioclip_candidate_pairs for path in pair})
        if unique_bioclip_paths and bioclip_preprocess is not None and bioclip_model is not None:
            _progress(
                progress_fn,
                f"Encoding BioCLIP refinement embeddings for class '{normalized_class_name}' ({len(unique_bioclip_paths)} images).",
            )
            bioclip_embeddings = _encode_bioclip_with_components(
                [Path(record_by_path[path].absolute_path) for path in unique_bioclip_paths],
                preprocess_val=bioclip_preprocess,
                model=bioclip_model,
                batch_size=effective_batch_size,
                device=device,
                amp_dtype=amp_dtype,
            )
            bioclip_scores.update(
                _compute_pair_similarity(
                    bioclip_embeddings,
                    path_keys=unique_bioclip_paths,
                    pairs=bioclip_candidate_pairs,
                )
            )

        for path_a, path_b in sorted(class_dino_pairs.keys()):
            record_a = record_by_path[path_a]
            record_b = record_by_path[path_b]
            phash_distance = _phash_distance(record_a.phash_hex, record_b.phash_hex)
            dino_cosine = class_dino_pairs.get((path_a, path_b), float("-inf"))
            bioclip_cosine = bioclip_scores.get((path_a, path_b), float("-inf"))
            exact_match = bool(record_a.exact_hash and record_a.exact_hash == record_b.exact_hash)

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
                        adjacency_distance=_adjacency_distance(record_a, record_b),
                        review_rank=_review_rank(record_a, record_b),
                        decision="review",
                        reason="borderline same-class similarity; adjacency only affects review ordering",
                    )
                )

    if bioclip_model is not None:
        bioclip_model.to("cpu")
        del bioclip_model
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    record_lookup = {record.relative_path: record for record in valid_records}
    review_pairs_total = len(review_pairs)
    review_pairs, review_clusters, auto_resolved_review_clusters = _triage_review_clusters(
        review_pairs=review_pairs,
        record_lookup=record_lookup,
        uf=uf,
    )
    high_risk_review_clusters = [row for row in review_clusters if row.resolution != "auto_resolve"]

    review_pairs = sorted(
        review_pairs,
        key=lambda item: (
            item.class_a,
            item.review_rank,
            -max(float(item.dino_cosine), float(item.bioclip_cosine)),
            item.phash_distance,
            item.path_a,
            item.path_b,
        ),
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
            "same_class_review_pairs_total": review_pairs_total,
            "same_class_review_clusters_total": len(review_clusters),
            "same_class_auto_resolved_clusters": len(auto_resolved_review_clusters),
            "same_class_high_risk_clusters": len(high_risk_review_clusters),
            "cross_class_conflicts": len(blocking_conflicts),
            "blocking_issues": len(blocking_issues),
            "excluded_reason_breakdown": {
                "unreadable": len([record for record in records if record.excluded_reason == "unreadable"]),
                "unhashable": len([record for record in records if record.excluded_reason == "unhashable"]),
                "missing_after_scan": availability_exclusions["missing_after_scan"],
                "unreadable_after_scan": availability_exclusions["unreadable_after_scan"],
            },
            "adjacency_used_for_review_ranking_only": True,
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
    _write_csv(artifact_root / "same_class_review_clusters.csv", [asdict(cluster) for cluster in review_clusters])
    _write_csv(
        artifact_root / "same_class_auto_resolved_clusters.csv",
        [asdict(cluster) for cluster in auto_resolved_review_clusters],
    )
    _write_csv(
        artifact_root / "same_class_high_risk_clusters.csv",
        [asdict(cluster) for cluster in high_risk_review_clusters],
    )
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
