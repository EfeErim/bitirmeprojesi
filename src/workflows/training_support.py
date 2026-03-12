"""Internal helpers for the public training workflow facade."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


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


def loader_batch_count(loader: Any) -> int:
    try:
        return int(len(loader))
    except Exception:
        return 0


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
    loader_batch_counts: Dict[str, int]
    detected_classes: List[str]
    split_class_counts: Dict[str, Dict[str, int]]
    adapter: Any


def _dataset_class_counts(loader: Any) -> Dict[str, int]:
    dataset = getattr(loader, "dataset", None)
    classes = [str(name) for name in getattr(dataset, "classes", [])]
    labels = getattr(dataset, "labels", None)
    if not classes or not isinstance(labels, list):
        return {}
    counts = Counter(int(label) for label in labels)
    return {
        class_name: int(counts.get(class_index, 0))
        for class_index, class_name in enumerate(classes)
    }


def _validate_training_layout(
    *,
    crop_name: str,
    data_dir: str | Path,
    batch_size: int,
    loaders: Dict[str, Any],
    loader_sizes: Dict[str, int],
    loader_batch_counts: Dict[str, int],
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    train_loader = loaders.get("train")
    if train_loader is None:
        raise ValueError("Training loader was not created.")

    detected_classes = [str(name) for name in getattr(train_loader.dataset, "classes", [])]
    if not detected_classes:
        raise ValueError(f"No classes found for crop '{crop_name}' in {data_dir}")

    train_samples = int(loader_sizes.get("train", 0))
    if train_samples <= 0:
        raise ValueError(
            f"Training split '{crop_name}/continual' is empty under {data_dir}. "
            "At least one in-distribution training sample is required."
        )

    train_batches = int(loader_batch_counts.get("train", 0))
    if train_batches <= 0:
        raise ValueError(
            "Training produced zero batches before the first optimizer step. "
            f"train_samples={train_samples}, batch_size={batch_size}. "
            "Reduce batch_size or increase the continual split size."
        )

    split_class_counts = {
        split_name: _dataset_class_counts(loader)
        for split_name, loader in loaders.items()
        if split_name in {"train", "val", "test"}
    }
    train_counts = dict(split_class_counts.get("train", {}))
    eval_only_classes: Dict[str, List[str]] = {}
    for split_name in ("val", "test"):
        split_counts = dict(split_class_counts.get(split_name, {}))
        leaked = [
            name
            for name, count in split_counts.items()
            if int(count) > 0 and int(train_counts.get(name, 0)) <= 0
        ]
        if leaked:
            eval_only_classes[split_name] = sorted(leaked)
    if eval_only_classes:
        details = "; ".join(f"{split}: {', '.join(names)}" for split, names in eval_only_classes.items())
        raise ValueError(
            "Validation/test splits contain classes that have no continual-train samples: "
            + details
        )

    if train_counts:
        missing_train_classes = [name for name in detected_classes if int(train_counts.get(name, 0)) <= 0]
        if missing_train_classes:
            raise ValueError(
                "Detected classes without training samples: "
                + ", ".join(sorted(missing_train_classes))
            )

    return detected_classes, split_class_counts


def select_calibration_source(loaders: Dict[str, Any], loader_sizes: Dict[str, int]) -> tuple[str, Any]:
    val_loader = loaders.get("val")
    if val_loader is not None and int(loader_sizes.get("val", 0)) > 0:
        return "val", val_loader
    return "train", loaders.get("train")


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
    resolved_run_id = str(run_id or f"{crop_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}")

    loaders = loader_factory(
        data_dir=str(data_dir),
        crop=crop_name,
        class_names=class_names,
        batch_size=int(training_cfg.get("batch_size", 8)),
        num_workers=int(num_workers if num_workers is not None else colab_cfg.get("num_workers", 2)),
        use_cache=bool(use_cache),
        cache_size=int(data_cfg.get("cache_size", 1000)),
        cache_train_split=bool(data_cfg.get("cache_train_split", False)),
        target_size=int(data_cfg.get("target_size", 224)),
        error_policy=str(error_policy or data_cfg.get("loader_error_policy", "tolerant")),
        sampler=str(sampler or data_cfg.get("sampler", "shuffle")),
        seed=int(training_cfg.get("seed", 42)),
        validate_images_on_init=bool(data_cfg.get("validate_images_on_init", True)),
        pin_memory=bool(colab_cfg.get("pin_memory", True) if pin_memory is None else pin_memory),
    )
    loader_sizes = build_loader_sizes(loaders)
    loader_batch_counts = {name: loader_batch_count(loader) for name, loader in loaders.items()}
    detected_classes, split_class_counts = _validate_training_layout(
        crop_name=crop_name,
        data_dir=data_dir,
        batch_size=int(training_cfg.get("batch_size", 8)),
        loaders=loaders,
        loader_sizes=loader_sizes,
        loader_batch_counts=loader_batch_counts,
    )

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
        loader_batch_counts=loader_batch_counts,
        detected_classes=detected_classes,
        split_class_counts=split_class_counts,
        adapter=adapter,
    )

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
