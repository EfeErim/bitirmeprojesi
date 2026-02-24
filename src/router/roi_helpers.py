#!/usr/bin/env python3
"""ROI/image utility helpers for VLM router."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

BoundingBox = List[float]
Detection = Dict[str, Any]

__all__ = [
    'tensor_to_pil',
    'coerce_image_input',
    'extract_roi',
    'sanitize_bbox',
    'bbox_area_ratio',
    'bbox_iou',
    'suppress_overlapping_detections',
    'select_best_detection',
    'unique_nonempty',
]


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert CHW/NCHW tensor image to RGB PIL image."""
    tensor = image_tensor.detach().cpu()

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims (C,H,W), got shape {tuple(tensor.shape)}")

    if tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(2, 0, 1)

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")

    tensor_min = float(tensor.min())
    tensor_max = float(tensor.max())

    if tensor_max <= 1.0 and tensor_min >= 0.0:
        normalized = tensor
    else:
        normalized = tensor.clone()
        if tensor_min < 0.0:
            normalized = (normalized + 1.0) / 2.0
        normalized = normalized.clamp(0.0, 1.0)

    uint8_img = (normalized * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(uint8_img)


def coerce_image_input(image_input: Any) -> Tuple[Image.Image, Tuple[int, int, int]]:
    """Normalize supported image input into RGB PIL image and size tuple."""
    if isinstance(image_input, torch.Tensor):
        pil = tensor_to_pil(image_input)
        return pil, tuple(image_input.shape)

    if isinstance(image_input, (str, Path)):
        pil = Image.open(str(image_input)).convert('RGB')
        width, height = pil.size
        return pil, (3, height, width)

    if isinstance(image_input, Image.Image):
        pil = image_input.convert('RGB')
        width, height = pil.size
        return pil, (3, height, width)

    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported ndarray shape for image input: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert('RGB')
        height, width = arr.shape[:2]
        return pil, (3, height, width)

    raise TypeError(
        f"Unsupported image_input type: {type(image_input).__name__}. "
        "Expected torch.Tensor, str/path, PIL.Image, or numpy.ndarray."
    )


def extract_roi(image: Image.Image, bbox: Optional[BoundingBox], pad_ratio: float = 0.08) -> Image.Image:
    """Extract padded ROI from bbox; return original image for invalid boxes."""
    if bbox is None or len(bbox) != 4:
        return image

    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox]

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio

    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))


def sanitize_bbox(bbox: Optional[BoundingBox], image_width: int, image_height: int) -> Optional[BoundingBox]:
    """Clamp bbox to image bounds and return None when invalid."""
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def bbox_area_ratio(bbox: Optional[BoundingBox], image_width: int, image_height: int) -> float:
    """Compute bbox area divided by full image area."""
    if bbox is None or image_width <= 0 or image_height <= 0:
        return 0.0

    x1, y1, x2, y2 = bbox
    box_w = max(0.0, float(x2) - float(x1))
    box_h = max(0.0, float(y2) - float(y1))
    image_area = float(image_width * image_height)
    if image_area <= 0.0:
        return 0.0

    return (box_w * box_h) / image_area


def bbox_iou(box_a: Optional[BoundingBox], box_b: Optional[BoundingBox]) -> float:
    """Compute IoU overlap for two xyxy bounding boxes."""
    if box_a is None or box_b is None or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def suppress_overlapping_detections(
    detections: List[Detection],
    iou_threshold: float = 0.75,
    same_crop_only: bool = True,
) -> List[Detection]:
    """Greedily drop overlapping detections, optionally crop-scoped."""
    if not detections:
        return []

    kept: List[Detection] = []
    for candidate in detections:
        candidate_bbox = candidate.get('bbox')
        candidate_crop = str(candidate.get('crop', '')).strip().lower()
        should_drop = False

        for existing in kept:
            if same_crop_only:
                existing_crop = str(existing.get('crop', '')).strip().lower()
                if existing_crop != candidate_crop:
                    continue

            overlap = bbox_iou(candidate_bbox, existing.get('bbox'))
            if overlap >= iou_threshold:
                should_drop = True
                break

        if not should_drop:
            kept.append(candidate)

    return kept


def select_best_detection(detections: List[Detection]) -> Optional[Detection]:
    """Select highest-score detection from list, else None."""
    if not detections:
        return None
    return max(detections, key=lambda detection: float(detection.get('score', 0.0)))


def unique_nonempty(values: List[Optional[str]]) -> List[str]:
    """Return order-preserving, case-insensitive unique non-empty strings."""
    seen = set()
    result: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result
