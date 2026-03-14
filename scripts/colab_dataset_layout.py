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
_CLASS_ALIAS_GROUPS = (
    {"healthy", "healthy_leaf"},
    {"gray_mold", "botrytis_gray_mold"},
    {"yellow_leaf_curl", "yellow_leaf_curl_virus"},
    {"spotted_wilt_virus", "tomato_spotted_wilt_virus"},
)


def normalize_class_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _build_alias_lookup() -> Dict[str, set[str]]:
    lookup: Dict[str, set[str]] = {}
    for group in _CLASS_ALIAS_GROUPS:
        resolved_group = {normalize_class_name(item) for item in group if normalize_class_name(item)}
        for alias in resolved_group:
            lookup[alias] = set(resolved_group)
    return lookup


_CLASS_ALIAS_LOOKUP = _build_alias_lookup()


def _class_name_aliases(name: str, *, crop_name: str) -> set[str]:
    normalized = normalize_class_name(name)
    crop_key = normalize_class_name(crop_name)
    aliases = {normalized}

    if crop_key:
        crop_prefix = f"{crop_key}_"
        if normalized.startswith(crop_prefix):
            stripped = normalized[len(crop_prefix):]
            if stripped:
                aliases.add(stripped)
        aliases.add(crop_prefix + normalized if not normalized.startswith(crop_prefix) else normalized)

    if normalized.endswith("_leaf"):
        aliases.add(normalized[: -len("_leaf")])
    if normalized in {"healthy_leaf", f"{crop_key}_healthy_leaf", f"{crop_key}_healthy"}:
        aliases.add("healthy")

    expanded = set(aliases)
    for alias in list(aliases):
        expanded.update(_CLASS_ALIAS_LOOKUP.get(alias, set()))
    return {item for item in expanded if item}


def resolve_notebook_training_classes(
    *,
    available_classes: Iterable[str],
    crop_name: str,
    taxonomy: Optional[Dict[str, Any]] = None,
    taxonomy_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_available = sorted(
        {
            normalize_class_name(name)
            for name in list(available_classes)
            if normalize_class_name(name)
        }
    )
    crop_key = normalize_class_name(crop_name)
    resolution: Dict[str, Any] = {
        "selected_classes": list(resolved_available),
        "used_taxonomy_filter": False,
        "reason": "no_available_classes" if not resolved_available else "taxonomy_unavailable",
        "expected_classes": [],
        "matched_classes": [],
        "unmatched_classes": [],
    }
    if not resolved_available:
        return resolution

    taxonomy_payload: Dict[str, Any] = {}
    if isinstance(taxonomy, dict):
        taxonomy_payload = dict(taxonomy)
    elif taxonomy_path is not None and Path(taxonomy_path).exists():
        loaded = read_json(Path(taxonomy_path), default={}, expect_type=dict)
        if isinstance(loaded, dict):
            taxonomy_payload = dict(loaded)

    expected = sorted(
        {
            normalize_class_name(item)
            for item in (
                taxonomy_payload.get("crop_specific_diseases", {}).get(crop_key, [])
                if isinstance(taxonomy_payload.get("crop_specific_diseases", {}), dict)
                else []
            )
            if normalize_class_name(item)
        }
        | {"healthy"}
    )
    resolution["expected_classes"] = list(expected)
    if not expected:
        resolution["reason"] = "crop_not_in_taxonomy"
        return resolution

    expected_set = set(expected)
    matched: List[str] = []
    unmatched: List[str] = []
    for class_name in resolved_available:
        aliases = _class_name_aliases(class_name, crop_name=crop_key)
        if aliases & expected_set:
            matched.append(class_name)
        else:
            unmatched.append(class_name)

    resolution["matched_classes"] = list(matched)
    resolution["unmatched_classes"] = list(unmatched)
    if matched and not unmatched:
        resolution["selected_classes"] = list(matched)
        resolution["used_taxonomy_filter"] = True
        resolution["reason"] = "full_taxonomy_alignment"
        return resolution

    resolution["reason"] = "partial_taxonomy_alignment_fallback"
    return resolution


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
    ood_root: Optional[Path] = None,
    runtime_root: Optional[Path] = None,
    materialization_strategy: str = "auto",
) -> Path:
    """Split class-root data into runtime layout and optionally materialize `ood/`."""
    runtime_dataset_root = runtime_root or (Path(__file__).resolve().parents[1] / "data" / "runtime_notebook_datasets")
    crop_root = runtime_dataset_root / str(crop_name)
    split_manifest_path = crop_root / "split_manifest.json"
    legacy_manifest_path = crop_root / "_split_metadata.json"
    resolved_ood_root = Path(ood_root) if ood_root is not None else None

    source_manifest = build_runtime_split_manifest(
        class_root=Path(class_root),
        crop_name=str(crop_name),
        seed=int(seed),
        allowed=allowed,
    )
    comparison_manifest = _public_manifest(source_manifest)
    comparison_manifest["ood"] = None
    ood_images: List[Path] = []
    if resolved_ood_root is not None:
        if not resolved_ood_root.exists():
            raise FileNotFoundError(f"OOD root not found: {resolved_ood_root}")
        if not resolved_ood_root.is_dir():
            raise NotADirectoryError(f"OOD root is not a directory: {resolved_ood_root}")
        ood_images = sorted(
            [
                path
                for path in resolved_ood_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda path: str(path).lower(),
        )
        comparison_manifest["ood"] = {
            "source_root": str(resolved_ood_root.resolve()),
            "image_count": len(ood_images),
            "image_fingerprint": _fingerprint_paths(ood_images, root=resolved_ood_root),
        }

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
        source_class_name = str(class_entry.get("source_class_name", ""))
        if not class_name:
            continue
        relative_image_paths = [Path(item) for item in class_entry.get("_relative_image_paths", [])]
        images: List[tuple[Path, Path]] = []
        for rel_path in relative_image_paths:
            source_path = Path(class_root) / rel_path
            try:
                destination_relative_path = rel_path.relative_to(source_class_name)
            except Exception:
                destination_relative_path = Path(rel_path.name)
            images.append((source_path, destination_relative_path))
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
            for source_path, destination_relative_path in files:
                destination_path = dst_dir / destination_relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                _materialize_image(source_path, destination_path, materialization_strategy)

    public_manifest = _public_manifest(source_manifest)
    public_manifest["ood"] = comparison_manifest["ood"]
    if resolved_ood_root is not None:
        ood_dir = crop_root / "ood"
        ood_dir.mkdir(parents=True, exist_ok=True)
        for source_path in ood_images:
            destination_relative_path = source_path.relative_to(resolved_ood_root)
            destination_path = ood_dir / destination_relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            _materialize_image(source_path, destination_path, materialization_strategy)
    write_json(split_manifest_path, public_manifest, ensure_ascii=False)
    write_json(legacy_manifest_path, public_manifest, ensure_ascii=False)
    return runtime_dataset_root
