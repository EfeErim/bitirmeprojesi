"""Image transform helpers shared by training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from torchvision import transforms


def build_image_transform(target_size: int, training: bool) -> "transforms.Compose":
    from torchvision import transforms

    if training:
        steps: List[Any] = [
            transforms.RandomResizedCrop(
                size=(target_size, target_size),
                scale=(0.65, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(25),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.35,
                hue=0.15,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.2,
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    else:
        steps = [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    return transforms.Compose(steps)


def preprocess_image(image: Union[np.ndarray, Image.Image], target_size: int = 224) -> torch.Tensor:
    """Normalize a single image into an ImageNet-style tensor."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                image = np.repeat(image, 3, axis=2)
            elif channels == 3:
                image = image[:, :, ::-1]
            elif channels == 4:
                image = image[:, :, [2, 1, 0]]
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
