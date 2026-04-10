"""Image transform helpers shared by training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from torchvision import transforms


VALID_AUGMENTATION_POLICIES = {"none", "basic", "randaugment"}


def normalize_augmentation_policy(value: str | None) -> str:
    policy = str(value or "randaugment").strip().lower()
    if policy not in VALID_AUGMENTATION_POLICIES:
        raise ValueError(
            "Unsupported augmentation_policy: "
            f"{value}. Expected one of: {', '.join(sorted(VALID_AUGMENTATION_POLICIES))}."
        )
    return policy


def _normalize_randaugment_num_ops(value: int) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError("randaugment_num_ops must be at least 1.")
    return resolved


def _normalize_randaugment_magnitude(value: int) -> int:
    resolved = int(value)
    if not 0 <= resolved <= 30:
        raise ValueError("randaugment_magnitude must be between 0 and 30.")
    return resolved


def _normalization_steps() -> List[Any]:
    from torchvision import transforms

    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


def build_image_transform(
    target_size: int,
    training: bool,
    augmentation_policy: str = "randaugment",
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 7,
) -> "transforms.Compose":
    from torchvision import transforms

    if training:
        policy = normalize_augmentation_policy(augmentation_policy)
        if policy == "none":
            steps = [
                transforms.Resize((target_size, target_size)),
                *_normalization_steps(),
            ]
        else:
            steps = [
                transforms.RandomResizedCrop(
                    size=(target_size, target_size),
                    scale=(0.8, 1.0),
                    ratio=(0.95, 1.05),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(12),
            ]
            if policy == "randaugment":
                steps.append(
                    transforms.RandAugment(
                        num_ops=_normalize_randaugment_num_ops(randaugment_num_ops),
                        magnitude=_normalize_randaugment_magnitude(randaugment_magnitude),
                    )
                )
            else:
                steps.append(
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                    )
                )
            steps.extend(
                [
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))], p=0.1),
                    *_normalization_steps(),
                ]
            )
    else:
        steps = [
            transforms.Resize((target_size, target_size)),
            *_normalization_steps(),
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
