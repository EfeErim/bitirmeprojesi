"""Compatibility wrapper for direct adapter smoke-test helpers."""

from __future__ import annotations

import src.pipeline.adapter_smoke as _impl

IMAGE_SUFFIXES = _impl.IMAGE_SUFFIXES
DEFAULT_ROBUST_VIEWS = _impl.DEFAULT_ROBUST_VIEWS
DEFAULT_EXPLANATION_METHOD = _impl.DEFAULT_EXPLANATION_METHOD
SUPPORTED_EXPLANATION_METHODS = _impl.SUPPORTED_EXPLANATION_METHODS
IMAGE_MEAN_PAD_RGB = _impl.IMAGE_MEAN_PAD_RGB
DEFAULT_DISCOVERY_ROOTS = _impl.DEFAULT_DISCOVERY_ROOTS
SKIP_DISCOVERY_DIR_NAMES = _impl.SKIP_DISCOVERY_DIR_NAMES
preprocess_image = _impl.preprocess_image
_build_adapter = _impl._build_adapter
_target_size = _impl._target_size
_prepare_view_tensor = _impl._prepare_view_tensor
_iter_adapter_meta_paths = _impl._iter_adapter_meta_paths


def _sync_impl() -> None:
    _impl.preprocess_image = preprocess_image
    _impl._build_adapter = _build_adapter
    _impl._target_size = _target_size
    _impl._prepare_view_tensor = _prepare_view_tensor
    _impl._iter_adapter_meta_paths = _iter_adapter_meta_paths


def discover_adapter_candidates(*args, **kwargs):
    _sync_impl()
    return _impl.discover_adapter_candidates(*args, **kwargs)


def load_adapter_summary(*args, **kwargs):
    _sync_impl()
    return _impl.load_adapter_summary(*args, **kwargs)


def predict_single_image(*args, **kwargs):
    _sync_impl()
    return _impl.predict_single_image(*args, **kwargs)


def predict_image_folder(*args, **kwargs):
    _sync_impl()
    return _impl.predict_image_folder(*args, **kwargs)


def build_prediction_visualization_images(*args, **kwargs):
    _sync_impl()
    return _impl.build_prediction_visualization_images(*args, **kwargs)


__all__ = [
    "DEFAULT_DISCOVERY_ROOTS",
    "DEFAULT_EXPLANATION_METHOD",
    "DEFAULT_ROBUST_VIEWS",
    "IMAGE_MEAN_PAD_RGB",
    "IMAGE_SUFFIXES",
    "SKIP_DISCOVERY_DIR_NAMES",
    "SUPPORTED_EXPLANATION_METHODS",
    "_build_adapter",
    "_iter_adapter_meta_paths",
    "_prepare_view_tensor",
    "_target_size",
    "build_prediction_visualization_images",
    "discover_adapter_candidates",
    "load_adapter_summary",
    "predict_image_folder",
    "predict_single_image",
    "preprocess_image",
]
