"""Internal helpers for the public training workflow facade."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


def loader_size(loader: Any) -> int:
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


def build_loader_sizes(loaders: Dict[str, Any]) -> Dict[str, int]:
    return {name: loader_size(loader) for name, loader in loaders.items()}


def stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_paths(item) for item in value]
    return value


@dataclass
class TrainingRunSetup:
    run_id: str
    training_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    colab_cfg: Dict[str, Any]
    loaders: Dict[str, Any]
    loader_sizes: Dict[str, int]
    detected_classes: List[str]
    adapter: Any


def prepare_training_run(
    *,
    config: Dict[str, Any],
    device: str,
    crop_name: str,
    data_dir: str | Path,
    class_names: Optional[Sequence[str]],
    num_workers: Optional[int],
    pin_memory: Optional[bool],
    use_cache: bool,
    sampler: Optional[str],
    error_policy: Optional[str],
    run_id: str,
    loader_factory: Callable[..., Dict[str, Any]],
    adapter_factory: Callable[..., Any],
) -> TrainingRunSetup:
    training_cfg = dict(config.get("training", {}).get("continual", {}))
    data_cfg = dict(training_cfg.get("data", {}))
    colab_cfg = dict(config.get("colab", {}).get("training", {}))
    resolved_run_id = str(run_id or f"{crop_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    loaders = loader_factory(
        data_dir=str(data_dir),
        crop=crop_name,
        batch_size=int(training_cfg.get("batch_size", 8)),
        num_workers=int(num_workers if num_workers is not None else colab_cfg.get("num_workers", 2)),
        use_cache=bool(use_cache),
        cache_size=int(data_cfg.get("cache_size", 1000)),
        target_size=int(data_cfg.get("target_size", 224)),
        error_policy=str(error_policy or data_cfg.get("loader_error_policy", "tolerant")),
        sampler=str(sampler or data_cfg.get("sampler", "shuffle")),
        seed=int(training_cfg.get("seed", 42)),
        validate_images_on_init=bool(data_cfg.get("validate_images_on_init", True)),
        pin_memory=bool(colab_cfg.get("pin_memory", True) if pin_memory is None else pin_memory),
    )
    loader_sizes = build_loader_sizes(loaders)

    detected_classes = list(class_names or getattr(loaders["train"].dataset, "classes", []))
    if not detected_classes:
        raise ValueError(f"No classes found for crop '{crop_name}' in {data_dir}")

    adapter = adapter_factory(
        crop_name=crop_name,
        model_name=str(
            training_cfg.get(
                "backbone",
                {},
            ).get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")
        ),
        device=device,
    )
    adapter.initialize_engine(class_names=detected_classes, config=config)

    return TrainingRunSetup(
        run_id=resolved_run_id,
        training_cfg=training_cfg,
        data_cfg=data_cfg,
        colab_cfg=colab_cfg,
        loaders=loaders,
        loader_sizes=loader_sizes,
        detected_classes=detected_classes,
        adapter=adapter,
    )


def select_calibration_loader(loaders: Dict[str, Any], loader_sizes: Dict[str, int]) -> Any:
    val_loader = loaders.get("val")
    if val_loader is not None and int(loader_sizes.get("val", 0)) > 0:
        return val_loader
    return loaders.get("train")


def build_artifact_payload(
    *,
    training_artifacts: Dict[str, Any],
    validation_artifacts: Dict[str, Any],
    test_artifacts: Dict[str, Any],
    ood_benchmark: Dict[str, Any],
    readiness_artifacts: Dict[str, Any],
    summary_artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    return stringify_paths(
        {
            "training": training_artifacts,
            "validation": validation_artifacts.get("paths", {}),
            "test": test_artifacts.get("paths", {}),
            "ood_benchmark": dict(ood_benchmark.get("paths", {})),
            "production_readiness": {"readiness_json": readiness_artifacts.get("readiness_json")},
            "summary": summary_artifacts,
        }
    )
