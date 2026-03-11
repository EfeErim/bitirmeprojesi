"""Dataset primitives for training and evaluation layouts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cache import LRUCache
from .transforms import build_image_transform

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VALID_ERROR_POLICIES = {"tolerant", "strict"}


def normalize_split(split: str) -> str:
    if split == "train":
        return "continual"
    if split in {"val", "test", "continual", "ood"}:
        return split
    raise ValueError(f"Unsupported split: {split}")


def default_crop_classes(crop: str) -> List[str]:
    defaults = {
        "tomato": ["healthy", "early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot"],
        "pepper": ["healthy", "bell_pepper_bacterial_spot"],
        "corn": ["healthy", "common_rust", "northern_leaf_blight"],
    }
    return defaults.get(str(crop).strip().lower(), ["healthy"])


def infer_crop_classes_from_layout(data_dir: str, crop: str) -> List[str]:
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
        cache_train_split: bool = False,
        error_policy: str = "tolerant",
        validate_images_on_init: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.crop = str(crop)
        self.split = normalize_split(split)
        self.target_size = int(target_size)
        self.use_cache = bool(use_cache)
        self.cache = LRUCache(cache_size) if self.use_cache else None
        self.cache_train_split = bool(cache_train_split)
        self.error_policy = str(error_policy).strip().lower()
        if self.error_policy not in VALID_ERROR_POLICIES:
            raise ValueError(f"Unsupported error policy: {error_policy}")
        self.validate_images_on_init = bool(validate_images_on_init)
        self.load_errors: List[Dict[str, str]] = []
        self.skipped_files: List[str] = []
        self._retired_indices: set[int] = set()

        inferred = [str(name) for name in class_names] if class_names is not None else infer_crop_classes_from_layout(
            data_dir=str(self.data_dir),
            crop=self.crop,
        )
        self.classes = inferred or default_crop_classes(self.crop)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.image_paths, self.labels = self._load_data()
        if self.validate_images_on_init:
            self._validate_image_paths()
        self.transform = build_image_transform(self.target_size, training=bool(transform))

    def _load_data(self) -> Tuple[List[Path], List[int]]:
        base_dir = self.data_dir / self.crop / self.split
        if not base_dir.exists():
            return [], []

        if self.split == "ood":
            ood_paths = sorted(
                path
                for path in base_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            return ood_paths, [-1] * len(ood_paths)

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
        cache = self.cache
        cache_enabled_for_split = self.use_cache and cache is not None and (
            self.split != "continual" or self.cache_train_split
        )
        if cache_enabled_for_split and cache is not None:
            cached = cache.get(str(img_path))
            if cached is not None:
                return cached.copy()

        image = self._decode_image(img_path)

        if cache_enabled_for_split and cache is not None:
            cache.put(str(img_path), image.copy())
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if not self.image_paths:
            raise IndexError("CropDataset is empty.")

        attempts = 0
        current_idx = int(idx % len(self.image_paths))
        while attempts < len(self.image_paths):
            if current_idx in self._retired_indices:
                attempts += 1
                current_idx = (current_idx + 1) % len(self.image_paths)
                continue
            img_path = self.image_paths[current_idx]
            label = self.labels[current_idx]
            try:
                image = self.transform(self._load_image(img_path))
                return image, label
            except Exception as exc:
                self._record_error(img_path, exc)
                if self.error_policy == "strict":
                    raise
                self._retired_indices.add(current_idx)
                attempts += 1
                current_idx = (current_idx + 1) % len(self.image_paths)

        raise RuntimeError("Unable to load any valid images from dataset.")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache) if self.cache is not None else 0,
            "cache_capacity": self.cache.capacity if self.cache is not None else 0,
            "cache_train_split": bool(self.cache_train_split),
            "load_error_count": len(self.load_errors),
            "skipped_files": list(self.skipped_files),
        }
