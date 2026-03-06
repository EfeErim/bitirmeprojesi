#!/usr/bin/env python3
"""Small dataset helpers used by notebook 2 and adapter inference."""

from __future__ import annotations

from collections import OrderedDict
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.crop = str(crop)
        self.split = _normalize_split(split)
        self.target_size = int(target_size)
        self.use_cache = bool(use_cache)
        self.cache = LRUCache(cache_size) if self.use_cache else None

        inferred = [str(name) for name in class_names] if class_names is not None else infer_crop_classes_from_layout(
            data_dir=str(self.data_dir),
            crop=self.crop,
        )
        self.classes = inferred or _default_crop_classes(self.crop)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.image_paths, self.labels = self._load_data()
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

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, img_path: Path) -> Image.Image:
        if self.use_cache and self.split != "continual" and self.cache is not None:
            cached = self.cache.get(str(img_path))
            if cached is not None:
                return cached.copy()

        raw = cv2.imread(str(img_path))
        if raw is None:
            raise ValueError(f"Failed to load image: {img_path}")
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        if self.use_cache and self.split != "continual" and self.cache is not None:
            self.cache.put(str(img_path), image.copy())
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = self.transform(self._load_image(img_path))
            return image, label
        except Exception as exc:
            logger.error("Failed to load %s: %s", img_path, exc)
            return torch.zeros(3, self.target_size, self.target_size), label

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self.cache) if self.cache is not None else 0,
            "cache_capacity": self.cache.capacity if self.cache is not None else 0,
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


def create_training_loaders(
    data_dir: str,
    crop: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_size: int = 1000,
    **dataloader_kwargs: Any,
) -> Dict[str, DataLoader]:
    """Create trainer-compatible loaders for notebook 2."""
    class_names = infer_crop_classes_from_layout(data_dir=data_dir, crop=crop)
    pin_memory = bool(dataloader_kwargs.pop("pin_memory", True))

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = CropDataset(
            data_dir=data_dir,
            crop=crop,
            split=split,
            class_names=class_names,
            transform=(split == "train"),
            use_cache=use_cache,
            cache_size=cache_size,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dict_collate_fn,
            **dataloader_kwargs,
        )
    return loaders
