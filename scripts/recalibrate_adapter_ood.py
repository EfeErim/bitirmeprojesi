#!/usr/bin/env python3
"""Recalibrate OOD state for an existing adapter export and save it back out."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from scripts.colab_adapter_smoke_test import (
    _resolve_adapter_dir,
    _resolve_crop_name,
    discover_adapter_candidates,
)
from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.loaders import create_training_loaders


def _loader_size(loader: Any) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except Exception:
            return 0
    try:
        return int(len(loader))
    except Exception:
        return 0


def _select_calibration_loader(loaders: Dict[str, Any]) -> tuple[str, Any]:
    val_loader = loaders.get("val")
    if val_loader is not None and _loader_size(val_loader) > 0:
        return "val", val_loader

    train_loader = loaders.get("train")
    if train_loader is not None and _loader_size(train_loader) > 0:
        return "train", train_loader

    raise ValueError("No usable calibration loader found. Expected a non-empty val or train split.")


def _copy_tree_overwrite(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def _save_adapter_in_place(adapter: IndependentCropAdapter, resolved_adapter_dir: Path) -> Path:
    if resolved_adapter_dir.name == "continual_sd_lora_adapter":
        return adapter.save_adapter(str(resolved_adapter_dir.parent))

    temp_root = resolved_adapter_dir.parent / f".tmp_recalibrated_adapter_{uuid.uuid4().hex}"
    try:
        saved_adapter_dir = adapter.save_adapter(str(temp_root))
        _copy_tree_overwrite(saved_adapter_dir, resolved_adapter_dir)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
    return resolved_adapter_dir


def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[int, int, float]:
    adapter_dir = Path(str(candidate.get("adapter_dir", "")))
    meta_path = adapter_dir / "adapter_meta.json"
    try:
        mtime = float(meta_path.stat().st_mtime)
    except OSError:
        mtime = -1.0
    calibration_version = int(candidate.get("ood_calibration_version", 0))
    calibrated = 1 if calibration_version > 0 else 0
    return (calibrated, calibration_version, mtime)


def _discover_adapter_dir(
    *,
    adapter_ref: str | Path,
    crop_name: Optional[str],
) -> Path:
    candidates = discover_adapter_candidates(
        search_roots=[adapter_ref],
        crop_name=crop_name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any adapter bundles under {Path(adapter_ref)}."
        )
    selected = max(candidates, key=_candidate_sort_key)
    return Path(str(selected["adapter_dir"]))


def _resolve_adapter_dir_with_search(
    *,
    adapter_ref: str | Path,
    crop_name: Optional[str],
    config_env: Optional[str],
) -> Path:
    try:
        return _resolve_adapter_dir(
            crop_name,
            adapter_dir=adapter_ref,
            config_env=config_env,
        )
    except FileNotFoundError:
        search_root = Path(adapter_ref)
        if search_root.is_dir():
            return _discover_adapter_dir(adapter_ref=search_root, crop_name=crop_name)
        raise


def run_recalibration(
    *,
    adapter_ref: str | Path,
    data_dir: str | Path,
    crop_name: Optional[str],
    output_dir: Optional[str | Path] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    config = get_config(environment=config_env)
    training_cfg = dict(config.get("training", {}).get("continual", {}))
    data_cfg = dict(training_cfg.get("data", {}))
    colab_cfg = dict(config.get("colab", {}).get("training", {}))

    resolved_adapter_dir = _resolve_adapter_dir_with_search(
        adapter_ref=adapter_ref,
        crop_name=crop_name,
        config_env=config_env,
    )
    crop_key = _resolve_crop_name(crop_name, adapter_dir=resolved_adapter_dir)

    loaders = create_training_loaders(
        data_dir=str(data_dir),
        crop=crop_key,
        batch_size=int(training_cfg.get("batch_size", 8)),
        num_workers=int(num_workers if num_workers is not None else colab_cfg.get("num_workers", 2)),
        use_cache=bool(use_cache),
        cache_size=int(data_cfg.get("cache_size", 1000)),
        target_size=int(data_cfg.get("target_size", 224)),
        error_policy=str(data_cfg.get("loader_error_policy", "tolerant")),
        sampler=str(data_cfg.get("sampler", "shuffle")),
        seed=int(training_cfg.get("seed", 42)),
        validate_images_on_init=bool(data_cfg.get("validate_images_on_init", True)),
        pin_memory=bool(colab_cfg.get("pin_memory", True) if pin_memory is None else pin_memory),
    )

    selected_split, calibration_loader = _select_calibration_loader(loaders)
    adapter = IndependentCropAdapter(crop_name=crop_key, device=device)
    adapter.load_adapter(str(resolved_adapter_dir))
    calibration_result = adapter.calibrate_ood(calibration_loader)

    overwritten_in_place = output_dir is None
    if output_dir is None:
        saved_adapter_dir = _save_adapter_in_place(adapter, resolved_adapter_dir)
    else:
        output_root = Path(output_dir)
        saved_adapter_dir = adapter.save_adapter(str(output_root))
    return {
        "status": "success",
        "crop_name": crop_key,
        "adapter_input_dir": str(resolved_adapter_dir),
        "adapter_output_dir": str(saved_adapter_dir),
        "overwritten_in_place": bool(overwritten_in_place),
        "data_dir": str(Path(data_dir)),
        "calibration_split": selected_split,
        "ood_calibration": dict(calibration_result.get("ood_calibration", {})),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recalibrate OOD state for an existing adapter export.")
    parser.add_argument(
        "adapter_ref",
        help="Adapter bundle dir, export dir, telemetry dir, adapter_meta.json path, or a parent folder to search recursively",
    )
    parser.add_argument("data_dir", help="Runtime dataset root containing train/val/test splits for the crop")
    parser.add_argument("--crop", dest="crop_name", help="Crop name (for example: tomato)")
    parser.add_argument("--output-dir", help="Directory where the recalibrated adapter should be saved")
    parser.add_argument("--config-env", default="colab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pin-memory", choices=("true", "false"))
    parser.add_argument("--no-cache", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    pin_memory = None
    if args.pin_memory is not None:
        pin_memory = args.pin_memory == "true"

    result = run_recalibration(
        adapter_ref=args.adapter_ref,
        data_dir=args.data_dir,
        crop_name=args.crop_name,
        output_dir=args.output_dir,
        config_env=args.config_env,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        use_cache=not bool(args.no_cache),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
