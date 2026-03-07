#!/usr/bin/env python3
"""Helpers for converting flat class-root datasets into runtime split layouts."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.shared.json_utils import read_json, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MATERIALIZATION_STRATEGIES = {"auto", "copy", "symlink", "hardlink"}


def normalize_class_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def estimate_split_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total < 3:
        return total, 0, 0

    train_count = max(1, int(total * 0.8))
    val_count = max(1, int(total * 0.1))
    if train_count + val_count >= total:
        val_count = 1
        train_count = max(1, total - 2)

    test_count = total - train_count - val_count
    if test_count < 1:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1
    return train_count, val_count, test_count


def _fingerprint_paths(paths: Iterable[Path], *, root: Path) -> str:
    digest = hashlib.sha1()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _public_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    classes: List[Dict[str, Any]] = []
    for entry in manifest.get("classes", []):
        classes.append({key: value for key, value in entry.items() if not str(key).startswith("_")})

    public_manifest = dict(manifest)
    public_manifest["classes"] = classes
    return public_manifest


def _resolve_materialization_attempts(strategy: str) -> List[str]:
    normalized = str(strategy or "auto").strip().lower()
    if normalized not in MATERIALIZATION_STRATEGIES:
        raise ValueError(f"Unsupported materialization strategy: {strategy}")

    if normalized == "auto":
        if os.name != "nt":
            return ["symlink", "hardlink", "copy"]
        return ["copy"]
    return [normalized]


def _materialize_image(source_path: Path, dest_path: Path, strategy: str) -> str:
    attempts = _resolve_materialization_attempts(strategy)
    last_error: Optional[Exception] = None

    for attempt in attempts:
        try:
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()

            if attempt == "copy":
                shutil.copy2(source_path, dest_path)
            elif attempt == "symlink":
                dest_path.symlink_to(source_path.resolve())
            elif attempt == "hardlink":
                os.link(source_path, dest_path)
            else:
                raise ValueError(f"Unsupported materialization strategy: {attempt}")
            return attempt
        except OSError as exc:
            last_error = exc
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to materialize dataset image.")


def build_runtime_split_manifest(
    *,
    class_root: Path,
    crop_name: str,
    seed: int,
    allowed: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    class_dirs = sorted([path for path in class_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower())
    allowed_set = {normalize_class_name(item) for item in list(allowed or []) if normalize_class_name(item)}
    classes: List[Dict[str, Any]] = []
    total_images = 0

    for class_dir in class_dirs:
        normalized = normalize_class_name(class_dir.name)
        if not normalized or (allowed_set and normalized not in allowed_set):
            continue
        images = sorted(
            [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda path: str(path).lower(),
        )
        train_count, val_count, test_count = estimate_split_counts(len(images))
        classes.append(
            {
                "source_class_name": class_dir.name,
                "class_name": normalized,
                "image_count": len(images),
                "image_fingerprint": _fingerprint_paths(images, root=class_root),
                "estimated_continual": train_count,
                "estimated_val": val_count,
                "estimated_test": test_count,
                "split_counts": {
                    "continual": train_count,
                    "val": val_count,
                    "test": test_count,
                },
                "_relative_image_paths": [path.relative_to(class_root).as_posix() for path in images],
            }
        )
        total_images += len(images)

    return {
        "schema_version": "v1_runtime_split_manifest",
        "source_root": str(class_root.resolve()),
        "crop": str(crop_name),
        "seed": int(seed),
        "allowed_classes": sorted(allowed_set),
        "split_policy": "80/10/10",
        "summary": {
            "num_classes": len(classes),
            "total_images": int(total_images),
        },
        "classes": classes,
    }


def prepare_runtime_dataset_layout(
    class_root: Path,
    crop_name: str,
    *,
    seed: int = 42,
    allowed: Optional[Iterable[str]] = None,
    runtime_root: Optional[Path] = None,
    materialization_strategy: str = "copy",
) -> Path:
    """Split class-root data into `continual/val/test` under the runtime dataset root."""
    runtime_dataset_root = runtime_root or (Path(__file__).resolve().parents[1] / "data" / "runtime_notebook_datasets")
    crop_root = runtime_dataset_root / str(crop_name)
    split_manifest_path = crop_root / "split_manifest.json"
    legacy_manifest_path = crop_root / "_split_metadata.json"

    source_manifest = build_runtime_split_manifest(
        class_root=Path(class_root),
        crop_name=str(crop_name),
        seed=int(seed),
        allowed=allowed,
    )
    comparison_manifest = _public_manifest(source_manifest)

    if crop_root.exists() and split_manifest_path.exists():
        try:
            if read_json(split_manifest_path, default={}) == comparison_manifest:
                return runtime_dataset_root
        except Exception:
            pass

    if crop_root.exists():
        shutil.rmtree(crop_root)
    crop_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed))
    for class_entry in source_manifest.get("classes", []):
        class_name = str(class_entry.get("class_name", ""))
        if not class_name:
            continue
        relative_image_paths = [Path(item) for item in class_entry.get("_relative_image_paths", [])]
        images = [Path(class_root) / rel_path for rel_path in relative_image_paths]
        rng.shuffle(images)

        continual_count, val_count, _ = estimate_split_counts(len(images))
        splits = {
            "continual": images[:continual_count],
            "val": images[continual_count:continual_count + val_count],
            "test": images[continual_count + val_count:],
        }
        for split_name, files in splits.items():
            dst_dir = crop_root / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for source_path in files:
                _materialize_image(source_path, dst_dir / source_path.name, materialization_strategy)

    public_manifest = _public_manifest(source_manifest)
    write_json(split_manifest_path, public_manifest, ensure_ascii=False)
    write_json(legacy_manifest_path, public_manifest, ensure_ascii=False)
    return runtime_dataset_root
