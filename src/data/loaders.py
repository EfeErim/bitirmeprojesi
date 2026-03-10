"""DataLoader assembly helpers."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import CropDataset, infer_crop_classes_from_layout

VALID_SAMPLERS = {"shuffle", "weighted"}


def seed_worker_factory(base_seed: int) -> Any:
    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

    return _seed_worker


def dict_collate_fn(batch: List[tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {"images": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
    images, labels = zip(*batch)
    return {
        "images": torch.stack(list(images), dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_weighted_sampler(dataset: CropDataset, seed: int) -> WeightedRandomSampler:
    counts = Counter(dataset.labels)
    weights = [1.0 / float(counts[label]) for label in dataset.labels]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def create_training_loaders(
    data_dir: str,
    crop: str,
    class_names: List[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_size: int = 1000,
    target_size: int = 224,
    error_policy: str = "tolerant",
    sampler: str = "shuffle",
    seed: int = 42,
    validate_images_on_init: bool = True,
    *,
    dataset_cls: type[CropDataset] = CropDataset,
    infer_classes_fn: Callable[[str, str], List[str]] = infer_crop_classes_from_layout,
    collate_fn: Callable[[List[tuple[torch.Tensor, int]]], Dict[str, torch.Tensor]] = dict_collate_fn,
    sampler_builder: Callable[[CropDataset, int], WeightedRandomSampler] = build_weighted_sampler,
    worker_seed_factory: Callable[[int], Any] = seed_worker_factory,
    **dataloader_kwargs: Any,
) -> Dict[str, DataLoader]:
    sampler_name = str(sampler).strip().lower()
    if sampler_name not in VALID_SAMPLERS:
        raise ValueError(f"Unsupported sampler: {sampler}")

    inferred_classes = [str(name) for name in infer_classes_fn(data_dir, crop)]
    resolved_classes = [str(name) for name in class_names] if class_names is not None else inferred_classes
    if inferred_classes and class_names is not None and set(resolved_classes) != set(inferred_classes):
        missing = sorted(set(inferred_classes) - set(resolved_classes))
        extra = sorted(set(resolved_classes) - set(inferred_classes))
        details: List[str] = []
        if missing:
            details.append(f"missing from override: {', '.join(missing)}")
        if extra:
            details.append(f"not found on disk: {', '.join(extra)}")
        raise ValueError("Provided class_names do not match the dataset layout (" + "; ".join(details) + ").")

    pin_memory = bool(dataloader_kwargs.pop("pin_memory", True))
    persistent_workers = bool(dataloader_kwargs.pop("persistent_workers", num_workers > 0))
    prefetch_factor = dataloader_kwargs.pop("prefetch_factor", None)

    loaders: Dict[str, DataLoader] = {}
    worker_init_fn = worker_seed_factory(int(seed))
    for split in ("train", "val", "test"):
        dataset = dataset_cls(
            data_dir=data_dir,
            crop=crop,
            split=split,
            class_names=resolved_classes,
            transform=(split == "train"),
            target_size=target_size,
            use_cache=use_cache,
            cache_size=cache_size,
            error_policy=error_policy,
            validate_images_on_init=validate_images_on_init,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed) + (0 if split == "train" else 10 if split == "val" else 20))

        split_sampler = None
        shuffle = split == "train"
        drop_last = split == "train"
        if split == "train" and sampler_name == "weighted" and len(dataset) > 0:
            split_sampler = sampler_builder(dataset, int(seed))
            shuffle = False

        extra_kwargs = dict(dataloader_kwargs)
        if num_workers <= 0:
            extra_kwargs.pop("prefetch_factor", None)
            persistent_workers = False
        elif prefetch_factor is not None:
            extra_kwargs["prefetch_factor"] = int(prefetch_factor)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if split_sampler is None else False,
            sampler=split_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=loader_generator,
            drop_last=drop_last,
            **extra_kwargs,
        )
        setattr(loaders[split], "_seed_base", int(seed) + (0 if split == "train" else 10 if split == "val" else 20))
        setattr(loaders[split], "_sampler_seed_base", int(seed) + (100 if split == "train" else 110 if split == "val" else 120))

    ood_root = Path(data_dir) / crop / "ood"
    if ood_root.exists():
        ood_dataset = dataset_cls(
            data_dir=data_dir,
            crop=crop,
            split="ood",
            class_names=resolved_classes,
            transform=False,
            target_size=target_size,
            use_cache=use_cache,
            cache_size=cache_size,
            error_policy=error_policy,
            validate_images_on_init=validate_images_on_init,
        )
        ood_generator = torch.Generator()
        ood_generator.manual_seed(int(seed) + 30)
        ood_extra_kwargs = dict(dataloader_kwargs)
        if num_workers <= 0:
            ood_extra_kwargs.pop("prefetch_factor", None)
        elif prefetch_factor is not None:
            ood_extra_kwargs["prefetch_factor"] = int(prefetch_factor)

        loaders["ood"] = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=ood_generator,
            **ood_extra_kwargs,
        )
        setattr(loaders["ood"], "_seed_base", int(seed) + 30)
        setattr(loaders["ood"], "_sampler_seed_base", int(seed) + 130)
    return loaders
