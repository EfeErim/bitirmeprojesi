#!/usr/bin/env python3
"""Compatibility facade over the split data-loading modules."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler

from src.data.cache import LRUCache
from src.data.datasets import (
    IMAGE_EXTENSIONS,
    VALID_ERROR_POLICIES,
)
from src.data.datasets import (
    CropDataset as _CropDataset,
)
from src.data.datasets import (
    default_crop_classes as _default_crop_classes_impl,
)
from src.data.datasets import (
    infer_crop_classes_from_layout as _infer_crop_classes_from_layout_impl,
)
from src.data.datasets import (
    normalize_split as _normalize_split_impl,
)
from src.data.loaders import (
    VALID_SAMPLERS,
)
from src.data.loaders import (
    build_weighted_sampler as _build_weighted_sampler_impl,
)
from src.data.loaders import (
    create_training_loaders as _create_training_loaders_impl,
)
from src.data.loaders import (
    dict_collate_fn as _dict_collate_fn_impl,
)
from src.data.loaders import (
    seed_worker_factory as _seed_worker_factory_impl,
)
from src.data.transforms import build_image_transform as _build_image_transform_impl
from src.data.transforms import preprocess_image as _preprocess_image_impl

CropDataset = _CropDataset


def _normalize_split(split: str) -> str:
    return _normalize_split_impl(split)


def _default_crop_classes(crop: str) -> List[str]:
    return _default_crop_classes_impl(crop)


def _image_transform(target_size: int, training: bool):
    return _build_image_transform_impl(target_size, training)


def _seed_worker_factory(base_seed: int) -> Any:
    return _seed_worker_factory_impl(base_seed)


def preprocess_image(image: Union[np.ndarray, Image.Image], target_size: int = 224) -> torch.Tensor:
    return _preprocess_image_impl(image, target_size=target_size)


def infer_crop_classes_from_layout(data_dir: str, crop: str) -> List[str]:
    return _infer_crop_classes_from_layout_impl(data_dir=data_dir, crop=crop)


def dict_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    return _dict_collate_fn_impl(batch)


def _build_weighted_sampler(dataset: CropDataset, seed: int) -> WeightedRandomSampler:
    return _build_weighted_sampler_impl(dataset, seed)


def create_training_loaders(
    data_dir: str,
    crop: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_size: int = 1000,
    target_size: int = 224,
    error_policy: str = "tolerant",
    sampler: str = "shuffle",
    seed: int = 42,
    validate_images_on_init: bool = True,
    **dataloader_kwargs: Any,
):
    return _create_training_loaders_impl(
        data_dir=data_dir,
        crop=crop,
        batch_size=batch_size,
        num_workers=num_workers,
        use_cache=use_cache,
        cache_size=cache_size,
        target_size=target_size,
        error_policy=error_policy,
        sampler=sampler,
        seed=seed,
        validate_images_on_init=validate_images_on_init,
        dataset_cls=CropDataset,
        infer_classes_fn=infer_crop_classes_from_layout,
        collate_fn=dict_collate_fn,
        sampler_builder=_build_weighted_sampler,
        worker_seed_factory=_seed_worker_factory,
        **dataloader_kwargs,
    )


__all__ = [
    "CropDataset",
    "IMAGE_EXTENSIONS",
    "LRUCache",
    "VALID_ERROR_POLICIES",
    "VALID_SAMPLERS",
    "_build_weighted_sampler",
    "_default_crop_classes",
    "_image_transform",
    "_normalize_split",
    "_seed_worker_factory",
    "create_training_loaders",
    "dict_collate_fn",
    "infer_crop_classes_from_layout",
    "preprocess_image",
]
