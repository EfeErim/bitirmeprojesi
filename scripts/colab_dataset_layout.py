"""Notebook dataset-layout wrapper around the canonical src surface."""

from __future__ import annotations

import src.data.dataset_layout as _impl

IMAGE_EXTENSIONS = _impl.IMAGE_EXTENSIONS
MATERIALIZATION_STRATEGIES = _impl.MATERIALIZATION_STRATEGIES
read_json = _impl.read_json
class_name_aliases = _impl.class_name_aliases
materialize_image = _impl.materialize_image
estimate_split_counts = _impl.estimate_split_counts
list_repo_dataset_directories = _impl.list_repo_dataset_directories
normalize_class_name = _impl.normalize_class_name
resolve_notebook_training_classes = _impl.resolve_notebook_training_classes
resolve_repo_dataset_directory = _impl.resolve_repo_dataset_directory
resolve_repo_relative_root = _impl.resolve_repo_relative_root


def _sync_impl() -> None:
    _impl.read_json = read_json


def build_runtime_split_manifest(*args, **kwargs):
    _sync_impl()
    return _impl.build_runtime_split_manifest(*args, **kwargs)


def prepare_runtime_dataset_layout(*args, **kwargs):
    _sync_impl()
    return _impl.prepare_runtime_dataset_layout(*args, **kwargs)


def main() -> int:
    _sync_impl()
    return _impl.main()


__all__ = [
    "IMAGE_EXTENSIONS",
    "MATERIALIZATION_STRATEGIES",
    "build_runtime_split_manifest",
    "class_name_aliases",
    "estimate_split_counts",
    "list_repo_dataset_directories",
    "main",
    "materialize_image",
    "normalize_class_name",
    "prepare_runtime_dataset_layout",
    "read_json",
    "resolve_notebook_training_classes",
    "resolve_repo_dataset_directory",
    "resolve_repo_relative_root",
]


if __name__ == "__main__":
    raise SystemExit(main())
