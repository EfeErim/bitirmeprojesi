#!/usr/bin/env python3
"""
Interactive Colab VLM test helper.

Usage in Colab:
    from scripts.colab_interactive_vlm_test import run_interactive_vlm_test
    run_interactive_vlm_test(config_path='/content/drive/MyDrive/aads_ulora/config/colab.json')
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def run_single_prediction(
    image_path: str,
    config_path: str,
    pipeline: Optional[IndependentMultiCropPipeline] = None,
) -> Dict[str, Any]:
    """Run a single image prediction and return normalized output."""
    if pipeline is None:
        config = _load_config(config_path)
        pipeline = IndependentMultiCropPipeline(config=config, device="cuda" if torch.cuda.is_available() else "cpu")
        pipeline.initialize_router()

    image = _open_image(Path(image_path))
    result = pipeline.process_image(image=image)

    return {
        "image": str(image_path),
        "plant_type": result.get("crop"),
        "plant_part": result.get("part"),
        "router_confidence": result.get("router_confidence", 0.0),
        "status": result.get("status", "unknown"),
        "raw": result,
    }


def run_interactive_vlm_test(config_path: str) -> None:
    """Interactive upload loop for Google Colab."""
    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError("This helper is intended for Google Colab only.") from error

    config = _load_config(config_path)
    pipeline = IndependentMultiCropPipeline(config=config, device="cuda" if torch.cuda.is_available() else "cpu")
    pipeline.initialize_router()

    print("Interactive VLM test started. Upload images to classify plant type and part.")
    print("Stop by sending an empty upload action in Colab.")

    while True:
        uploaded = files.upload()
        if not uploaded:
            print("No files uploaded. Stopping interactive test.")
            break

        for filename in uploaded:
            try:
                result = run_single_prediction(
                    image_path=filename,
                    config_path=config_path,
                    pipeline=pipeline,
                )
                print(
                    f"{result['image']}: plant={result['plant_type']}, "
                    f"part={result['plant_part']}, confidence={result['router_confidence']:.4f}, "
                    f"status={result['status']}"
                )
            except Exception as error:
                print(f"{filename}: failed -> {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Colab VLM testing helper")
    parser.add_argument("--config", required=True, help="Path to configuration JSON")
    parser.add_argument("--image", default=None, help="Optional image for single prediction")

    arguments = parser.parse_args()

    if arguments.image:
        prediction = run_single_prediction(image_path=arguments.image, config_path=arguments.config)
        print(prediction)
    else:
        run_interactive_vlm_test(config_path=arguments.config)
