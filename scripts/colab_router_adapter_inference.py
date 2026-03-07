#!/usr/bin/env python3
"""Thin entrypoint for router-driven adapter inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from src.workflows import InferenceWorkflow


def run_inference(
    image_path: str | Path,
    *,
    config_env: Optional[str] = "colab",
    crop_hint: Optional[str] = None,
    part_hint: Optional[str] = None,
    adapter_root: Optional[str | Path] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    workflow = InferenceWorkflow(environment=config_env, adapter_root=adapter_root, device=device)
    image = Image.open(image_path).convert("RGB")
    return workflow.predict(image, crop_hint=crop_hint, part_hint=part_hint, return_ood=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run router -> adapter inference for a single image.")
    parser.add_argument("image", type=Path, help="Image path")
    parser.add_argument("--config-env", default="colab", help="Config environment override (default: colab)")
    parser.add_argument("--crop", dest="crop_hint", help="Optional crop hint to bypass the router")
    parser.add_argument("--part", dest="part_hint", help="Optional part hint")
    parser.add_argument("--adapter-root", type=Path, help="Override adapter root directory")
    parser.add_argument("--device", default="cuda", help="Torch device preference")
    args = parser.parse_args()

    result = run_inference(
        args.image,
        config_env=args.config_env,
        crop_hint=args.crop_hint,
        part_hint=args.part_hint,
        adapter_root=args.adapter_root,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
