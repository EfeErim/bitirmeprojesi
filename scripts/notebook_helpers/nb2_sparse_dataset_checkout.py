"""Notebook 2 sparse-checkout helpers for repo-tracked runtime datasets."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


DATASET_SPARSE_PATHS: dict[str, tuple[str, ...]] = {
    "grape__fruit": (
        "data/prepared_runtime_datasets/grape__fruit",
        "data/ood_dataset/final/grape__fruit_ood_final",
        "data/oe_dataset/grape_fruit_oe_from_leaf",
    ),
    "grape__leaf": (
        "data/prepared_runtime_datasets/grape__leaf",
        "data/ood_dataset/final/grape__leaf_ood_final",
        "data/oe_dataset/grape_leaf_oe_unsupported_leaf_candidates",
    ),
    "strawberry__fruit": (
        "data/prepared_runtime_datasets/strawberry__fruit",
        "data/ood_dataset/final/strawberry__fruit_ood_final",
        "data/oe_dataset/strawberry_fruit_oe_candidates",
    ),
    "strawberry__leaf": (
        "data/prepared_runtime_datasets/strawberry__leaf",
        "data/ood_dataset/final/strawberry__leaf_ood_final",
        "data/oe_dataset/strawberry_leaf_oe_from_blossom_candidates",
    ),
    "tomato__fruit": (
        "data/prepared_runtime_datasets/tomato__fruit",
        "data/ood_dataset/final/tomato__fruit_ood_final",
        "data/oe_dataset/tomato_fruit_oe_from_leaf",
    ),
    "tomato__leaf": (
        "data/prepared_runtime_datasets/tomato__leaf",
        "data/ood_dataset/final/tomato__leaf_ood_final",
        "data/oe_dataset/tomato_leaf_oe_from_fruit",
    ),
    "apricot__fruit": (
        "data/prepared_runtime_datasets/apricot__fruit",
        "data/ood_dataset/final/apricot__fruit_ood_final",
    ),
    "apricot__leaf": (
        "data/prepared_runtime_datasets/apricot__leaf",
        "data/ood_dataset/final/apricot__leaf_ood_final",
        "data/oe_dataset/apricot_leaf_oe_unsupported_leaf_candidates",
    ),
}


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _requested_dataset_keys(crop_name: str, part_name: str, dataset_name: str) -> list[str]:
    requested = str(dataset_name or "").strip()
    if requested:
        return [requested]

    crop_key = _slug(crop_name)
    part_key = _slug(part_name)
    if not crop_key:
        return []
    if part_key and part_key not in {"unspecified", "all", "both"}:
        return [f"{crop_key}__{part_key}"]
    return [key for key in sorted(DATASET_SPARSE_PATHS) if key.startswith(f"{crop_key}__")]


def _repo_relative_path(repo_root: Path, raw_path: str) -> str | None:
    value = str(raw_path or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return None
    return path.as_posix()


def ensure_notebook2_dataset_sparse_checkout(
    repo_root: Path,
    *,
    crop_name: str,
    part_name: str,
    dataset_name: str,
    ood_root: str,
    oe_root: str,
    oe_enabled: bool,
    print_fn: Callable[[str], None] = print,
) -> list[str]:
    """Fetch only the Notebook 2 dataset paths needed by the selected adapter."""
    root = Path(repo_root)
    if not (root / ".git").exists():
        return []

    paths: list[str] = []
    requested_keys = _requested_dataset_keys(crop_name, part_name, dataset_name)
    for key in requested_keys:
        paths.extend(DATASET_SPARSE_PATHS.get(key, ()))

    # If a dataset key is new and has no static mapping yet, still fetch its prepared runtime root.
    for key in requested_keys:
        runtime_dataset_path = f"data/prepared_runtime_datasets/{key}"
        if runtime_dataset_path not in paths:
            paths.append(runtime_dataset_path)

    explicit_ood = _repo_relative_path(root, ood_root)
    if explicit_ood:
        paths.append(explicit_ood)
    explicit_oe = _repo_relative_path(root, oe_root) if bool(oe_enabled) else None
    if explicit_oe:
        paths.append(explicit_oe)

    unique_paths = list(dict.fromkeys(path for path in paths if path))
    missing = [path for path in unique_paths if not (root / path).exists()]
    if not missing:
        return []

    print_fn(f"[BOOTSTRAP] Fetching selected Notebook 2 dataset paths: {missing}")
    subprocess.run(
        ["git", "sparse-checkout", "add", *missing],
        cwd=str(root),
        check=True,
    )
    return missing
