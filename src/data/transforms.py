"""Image transform helpers shared by training and inference."""

from __future__ import annotations

from typing import Any, List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def build_image_transform(target_size: int, training: bool) -> transforms.Compose:
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
    return build_image_transform(int(target_size), training=False)(image)
