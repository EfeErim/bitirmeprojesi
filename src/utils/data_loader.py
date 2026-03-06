#!/usr/bin/env python3
"""Small dataset helpers used by notebook 2 and adapter inference."""

from __future__ import annotations

from collections import Counter, OrderedDict
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VALID_ERROR_POLICIES = {"tolerant", "strict"}
VALID_SAMPLERS = {"shuffle", "weighted"}


class LRUCache:
    """Tiny TTL-capable cache for preprocessed validation images."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = int(max(1, capacity))
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.ttl_seconds: Optional[float] = None

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        if self.ttl_seconds is not None:
            age = time.time() - self.timestamps.get(key, 0.0)
            if age > self.ttl_seconds:
                self.__delitem__(key)
                return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        self.timestamps[key] = time.time()
        while len(self.cache) > self.capacity:
            oldest, _ = self.cache.popitem(last=False)
            self.timestamps.pop(oldest, None)

    def clear(self) -> None:
        self.cache.clear()
        self.timestamps.clear()

    def set_ttl(self, seconds: Optional[float]) -> None:
        self.ttl_seconds = seconds

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, key: str) -> Optional[Any]:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)


def _normalize_split(split: str) -> str:
    if split == "train":
        return "continual"
    if split in {"val", "test", "continual"}:
        return split
    raise ValueError(f"Unsupported split: {split}")


def _default_crop_classes(crop: str) -> List[str]:
    defaults = {
        "tomato": ["healthy", "early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot"],
        "pepper": ["healthy", "bell_pepper_bacterial_spot"],
        "corn": ["healthy", "common_rust", "northern_leaf_blight"],
    }
    return defaults.get(str(crop).strip().lower(), ["healthy"])


def _image_transform(target_size: int, training: bool) -> transforms.Compose:
    steps: List[Any] = [transforms.Resize((target_size, target_size))]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def _seed_worker_factory(base_seed: int) -> Any:
    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)

    return _seed_worker


class CropDataset(Dataset):
    """Dataset over `data/<crop>/{continual,val,test}/<class>/<image>`."""

    def __init__(
        self,
        data_dir: str,
        crop: str,
        split: str = "train",
        class_names: Optional[Sequence[str]] = None,
        transform: bool = True,
        target_size: int = 224,
        use_cache: bool = True,
        cache_size: int = 1000,
        error_policy: str = "tolerant",
        validate_images_on_init: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.crop = str(crop)
        self.split = _normalize_split(split)
        self.target_size = int(target_size)
        self.use_cache = bool(use_cache)
        self.cache = LRUCache(cache_size) if self.use_cache else None
        self.error_policy = str(error_policy).strip().lower()
        if self.error_policy not in VALID_ERROR_POLICIES:
            raise ValueError(f"Unsupported error policy: {error_policy}")
        self.validate_images_on_init = bool(validate_images_on_init)
        self.load_errors: List[Dict[str, str]] = []
        self.skipped_files: List[str] = []

        inferred = [str(name) for name in class_names] if class_names is not None else infer_crop_classes_from_layout(
            data_dir=str(self.data_dir),
            crop=self.crop,
        )
        self.classes = inferred or _default_crop_classes(self.crop)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.image_paths, self.labels = self._load_data()
        if self.validate_images_on_init:
            self._validate_image_paths()
        self.transform = _image_transform(self.target_size, training=bool(transform))

    def _load_data(self) -> Tuple[List[Path], List[int]]:
        base_dir = self.data_dir / self.crop / self.split
        if not base_dir.exists():
            return [], []

        image_paths: List[Path] = []
        labels: List[int] = []
        for class_name in self.classes:
            class_dir = base_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image_paths.append(image_path)
                labels.append(self.class_to_idx[class_name])
        return image_paths, labels

    def _record_error(self, img_path: Path, exc: Exception) -> None:
        payload = {"path": str(img_path), "error": str(exc)}
        self.load_errors.append(payload)
        self.skipped_files.append(str(img_path))
        logger.error("Failed to load %s: %s", img_path, exc)

    def _validate_image_paths(self) -> None:
        if not self.image_paths:
            return

        valid_paths: List[Path] = []
        valid_labels: List[int] = []
        failed: List[Dict[str, str]] = []
        for img_path, label in zip(self.image_paths, self.labels):
            try:
                self._decode_image(img_path)
                valid_paths.append(img_path)
                valid_labels.append(label)
            except Exception as exc:
                failed.append({"path": str(img_path), "error": str(exc)})

        if failed and self.error_policy == "strict":
            first = failed[0]
            raise ValueError(f"Failed to validate dataset image {first['path']}: {first['error']}")

        if failed:
            self.load_errors.extend(failed)
            self.skipped_files.extend(row["path"] for row in failed)
            self.image_paths = valid_paths
            self.labels = valid_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _decode_image(img_path: Path) -> Image.Image:
        raw = cv2.imread(str(img_path))
        if raw is None:
            raise ValueError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _load_image(self, img_path: Path) -> Image.Image:
        if self.use_cache and self.split != "continual" and self.cache is not None:
            cached = self.cache.get(str(img_path))
            if cached is not None:
                return cached.copy()

        image = self._decode_image(img_path)

        if self.use_cache and self.split != "continual" and self.cache is not None:
            self.cache.put(str(img_path), image.copy())
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if not self.image_paths:
            raise IndexError("CropDataset is empty.")

        attempts = 0
        current_idx = int(idx % len(self.image_paths))
        while attempts < len(self.image_paths):
            img_path = self.image_paths[current_idx]
            label = self.labels[current_idx]
            try:
                image = self.transform(self._load_image(img_path))
                return image, label
            except Exception as exc:
                self._record_error(img_path, exc)
                if self.error_policy == "strict":
                    raise
                attempts += 1
                current_idx = (current_idx + 1) % len(self.image_paths)

        raise RuntimeError("Unable to load any valid images from dataset.")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache) if self.cache is not None else 0,
            "cache_capacity": self.cache.capacity if self.cache is not None else 0,
            "load_error_count": len(self.load_errors),
            "skipped_files": list(self.skipped_files),
        }


def preprocess_image(image: Union[np.ndarray, Image.Image], target_size: int = 224) -> torch.Tensor:
    """Normalize a single image into an ImageNet-style tensor."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"Unsupported channel count: {channels}")
        else:
            raise ValueError(f"Unsupported array shape: {image.shape}")
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return _image_transform(int(target_size), training=False)(image)


def infer_crop_classes_from_layout(data_dir: str, crop: str) -> List[str]:
    """Infer class names from crop split folders."""
    crop_root = Path(data_dir) / crop
    class_names: set[str] = set()
    for split in ("continual", "val", "test"):
        split_root = crop_root / split
        if not split_root.exists():
            continue
        for class_dir in split_root.iterdir():
            if class_dir.is_dir() and class_dir.name:
                class_names.add(class_dir.name)
    return sorted(class_names)


def dict_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {"images": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)}
    images, labels = zip(*batch)
    return {
        "images": torch.stack(list(images), dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _build_weighted_sampler(dataset: CropDataset, seed: int) -> WeightedRandomSampler:
    counts = Counter(dataset.labels)
    weights = [1.0 / float(counts[label]) for label in dataset.labels]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


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
) -> Dict[str, DataLoader]:
    """Create trainer-compatible loaders for notebook 2."""
    sampler_name = str(sampler).strip().lower()
    if sampler_name not in VALID_SAMPLERS:
        raise ValueError(f"Unsupported sampler: {sampler}")

    class_names = infer_crop_classes_from_layout(data_dir=data_dir, crop=crop)
    pin_memory = bool(dataloader_kwargs.pop("pin_memory", True))
    persistent_workers = bool(dataloader_kwargs.pop("persistent_workers", num_workers > 0))
    prefetch_factor = dataloader_kwargs.pop("prefetch_factor", None)

    loaders: Dict[str, DataLoader] = {}
    worker_init_fn = _seed_worker_factory(int(seed))
    for split in ("train", "val", "test"):
        dataset = CropDataset(
            data_dir=data_dir,
            crop=crop,
            split=split,
            class_names=class_names,
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
        if split == "train" and sampler_name == "weighted" and len(dataset) > 0:
            split_sampler = _build_weighted_sampler(dataset, seed=int(seed))
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
            collate_fn=dict_collate_fn,
            worker_init_fn=worker_init_fn,
            generator=loader_generator,
            **extra_kwargs,
        )
    return loaders
