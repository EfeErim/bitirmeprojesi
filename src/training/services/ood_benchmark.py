"""Held-out OOD evidence helpers for production readiness."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from src.training.services.metrics import load_plan_targets, validate_ood_metrics
from src.training.services.reporting import (
    persist_ood_benchmark_artifacts,
    persist_validation_artifacts,
)
from src.training.validation import evaluate_model_with_artifact_metrics


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


def _dataset_labels(dataset: Any) -> List[int]:
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, list):
        return [int(label) for label in labels]
    resolved: List[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError("Dataset must return `(image, label)` tuples for OOD benchmarking.")
        resolved.append(int(sample[1]))
    return resolved


class FilteredLabelDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        *,
        classes: Sequence[str],
        label_map: Optional[Dict[int, int]] = None,
        constant_label: Optional[int] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.classes = [str(name) for name in classes]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self._label_map = {int(key): int(value) for key, value in dict(label_map or {}).items()}
        self._constant_label = int(constant_label) if constant_label is not None else None
        self.labels = [self._remap_label(_dataset_labels(base_dataset)[index]) for index in self.indices]

    def _remap_label(self, label: int) -> int:
        if self._constant_label is not None:
            return int(self._constant_label)
        if label not in self._label_map:
            raise KeyError(f"Label {label} is not present in the fold remapping.")
        return int(self._label_map[label])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[self.indices[int(index)]]
        del label
        return image, self.labels[int(index)]


def _build_weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    counts = Counter(int(label) for label in labels)
    weights = [1.0 / float(counts[int(label)]) for label in labels]
    generator = torch.Generator()
    generator.manual_seed(42)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=generator)


def _clone_loader(source_loader: DataLoader, dataset: Dataset, *, training_like: bool) -> DataLoader:
    batch_size = int(getattr(source_loader, "batch_size", 1) or 1)
    num_workers = int(getattr(source_loader, "num_workers", 0))
    pin_memory = bool(getattr(source_loader, "pin_memory", False))
    persistent_workers = bool(getattr(source_loader, "persistent_workers", False)) if num_workers > 0 else False
    collate_fn = getattr(source_loader, "collate_fn", None)
    worker_init_fn = getattr(source_loader, "worker_init_fn", None)
    generator = getattr(source_loader, "generator", None)
    drop_last = bool(getattr(source_loader, "drop_last", False)) if training_like else False
    timeout = float(getattr(source_loader, "timeout", 0.0))
    prefetch_factor = getattr(source_loader, "prefetch_factor", None)

    sampler = None
    shuffle = False
    source_sampler = getattr(source_loader, "sampler", None)
    if training_like and isinstance(source_sampler, WeightedRandomSampler) and hasattr(dataset, "labels"):
        sampler = _build_weighted_sampler(getattr(dataset, "labels", []))
    elif training_like and isinstance(source_sampler, RandomSampler):
        shuffle = True

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_fn,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
        "drop_last": drop_last,
        "timeout": timeout,
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_value = _mean(values)
    assert mean_value is not None
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(len(values))
    return float(variance ** 0.5)


def _select_eval_loader(loaders: Dict[str, Any]) -> tuple[str, Any]:
    test_loader = loaders.get("test")
    if test_loader is not None and _loader_size(test_loader) > 0:
        return "test", test_loader
    return "val", loaders.get("val")


def _select_calibration_loader(loaders: Dict[str, Any]) -> tuple[str, Any]:
    val_loader = loaders.get("val")
    if val_loader is not None and _loader_size(val_loader) > 0:
        return "val", val_loader
    return "train", loaders.get("train")


def run_leave_one_class_out_benchmark(
    *,
    crop_name: str,
    class_names: Sequence[str],
    loaders: Dict[str, Any],
    config: Dict[str, Any],
    device: str,
    artifact_root: Path,
    adapter_factory: Callable[..., Any],
    run_id: str,
    num_epochs: Optional[int] = None,
    telemetry: Any = None,
    emit_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    min_classes: int = 3,
) -> Dict[str, Any]:
    resolved_classes = [str(name) for name in class_names]
    target_values = load_plan_targets()
    artifact_root = Path(artifact_root)
    emit = emit_event or (lambda _event_type, _payload: None)
    train_loader = loaders.get("train")
    eval_split_name, eval_loader = _select_eval_loader(loaders)
    calibration_split_name, calibration_loader = _select_calibration_loader(loaders)

    base_context = {
        "crop_name": str(crop_name),
        "run_id": str(run_id),
        "eval_split_name": eval_split_name,
        "calibration_split_name": calibration_split_name,
        "class_count": len(resolved_classes),
    }

    if len(resolved_classes) < int(min_classes):
        payload = {
            "schema_version": "v6_ood_benchmark",
            "status": "failed",
            "passed": False,
            "ood_evidence_source": "held_out_benchmark",
            "reason": "insufficient_classes_for_fallback",
            "metrics": {name: None for name in ("accuracy", "ood_auroc", "ood_false_positive_rate", "sure_ds_f1", "conformal_empirical_coverage", "conformal_avg_set_size")},
            "metric_std": {},
            "evaluation": validate_ood_metrics({}, target_values, require_ood=True),
            "successful_folds": 0,
            "failed_folds": 0,
            "folds": [],
            "targets": target_values,
            "context": {**base_context, "required_min_classes": int(min_classes)},
        }
        artifacts = persist_ood_benchmark_artifacts(artifact_root=artifact_root, summary_payload=payload, telemetry=telemetry)
        payload["paths"] = {key: str(path) for key, path in artifacts.items()}
        return payload

    if train_loader is None or _loader_size(train_loader) <= 0:
        payload = {
            "schema_version": "v6_ood_benchmark",
            "status": "failed",
            "passed": False,
            "ood_evidence_source": "held_out_benchmark",
            "reason": "missing_train_loader",
            "metrics": {name: None for name in ("accuracy", "ood_auroc", "ood_false_positive_rate", "sure_ds_f1", "conformal_empirical_coverage", "conformal_avg_set_size")},
            "metric_std": {},
            "evaluation": validate_ood_metrics({}, target_values, require_ood=True),
            "successful_folds": 0,
            "failed_folds": 0,
            "folds": [],
            "targets": target_values,
            "context": base_context,
        }
        artifacts = persist_ood_benchmark_artifacts(artifact_root=artifact_root, summary_payload=payload, telemetry=telemetry)
        payload["paths"] = {key: str(path) for key, path in artifacts.items()}
        return payload

    train_dataset = train_loader.dataset
    calibration_dataset = getattr(calibration_loader, "dataset", None)
    eval_dataset = getattr(eval_loader, "dataset", None)
    if calibration_dataset is None or eval_dataset is None:
        payload = {
            "schema_version": "v6_ood_benchmark",
            "status": "failed",
            "passed": False,
            "ood_evidence_source": "held_out_benchmark",
            "reason": "missing_eval_loader",
            "metrics": {name: None for name in ("accuracy", "ood_auroc", "ood_false_positive_rate", "sure_ds_f1", "conformal_empirical_coverage", "conformal_avg_set_size")},
            "metric_std": {},
            "evaluation": validate_ood_metrics({}, target_values, require_ood=True),
            "successful_folds": 0,
            "failed_folds": 0,
            "folds": [],
            "targets": target_values,
            "context": base_context,
        }
        artifacts = persist_ood_benchmark_artifacts(artifact_root=artifact_root, summary_payload=payload, telemetry=telemetry)
        payload["paths"] = {key: str(path) for key, path in artifacts.items()}
        return payload

    class_to_idx = dict(getattr(train_dataset, "class_to_idx", {name: idx for idx, name in enumerate(resolved_classes)}))
    train_labels = _dataset_labels(train_dataset)
    calibration_labels = _dataset_labels(calibration_dataset)
    eval_labels = _dataset_labels(eval_dataset)
    folds: List[Dict[str, Any]] = []

    emit(
        "ood_benchmark_started",
        {
            **base_context,
            "fold_count": len(resolved_classes),
        },
    )

    for fold_index, held_out_class in enumerate(resolved_classes, start=1):
        seen_classes = [name for name in resolved_classes if name != held_out_class]
        held_out_label = int(class_to_idx[held_out_class])
        seen_label_map = {int(class_to_idx[name]): idx for idx, name in enumerate(seen_classes)}

        train_indices = [idx for idx, label in enumerate(train_labels) if label in seen_label_map]
        calibration_indices = [idx for idx, label in enumerate(calibration_labels) if label in seen_label_map]
        eval_id_indices = [idx for idx, label in enumerate(eval_labels) if label in seen_label_map]
        eval_ood_indices = [idx for idx, label in enumerate(eval_labels) if label == held_out_label]

        sample_counts = {
            "train_samples": len(train_indices),
            "calibration_samples": len(calibration_indices),
            "eval_in_distribution_samples": len(eval_id_indices),
            "eval_ood_samples": len(eval_ood_indices),
        }
        emit(
            "ood_benchmark_fold_started",
            {
                **base_context,
                "fold_index": fold_index,
                "fold_total": len(resolved_classes),
                "held_out_class": held_out_class,
                "seen_classes": list(seen_classes),
                "sample_counts": sample_counts,
            },
        )

        if (
            sample_counts["train_samples"] <= 0
            or sample_counts["calibration_samples"] <= 0
            or sample_counts["eval_in_distribution_samples"] <= 0
            or sample_counts["eval_ood_samples"] <= 0
        ):
            reason = "missing_fold_samples"
            fold_payload = {
                "held_out_class": held_out_class,
                "status": "failed",
                "reason": reason,
                "seen_classes": list(seen_classes),
                "metrics": {},
                "sample_counts": sample_counts,
                "paths": {},
            }
            folds.append(fold_payload)
            emit(
                "ood_benchmark_fold_completed",
                {
                    **base_context,
                    "fold_index": fold_index,
                    "fold_total": len(resolved_classes),
                    **fold_payload,
                },
            )
            continue

        fold_train_dataset = FilteredLabelDataset(
            train_dataset,
            train_indices,
            classes=seen_classes,
            label_map=seen_label_map,
        )
        fold_calibration_dataset = FilteredLabelDataset(
            calibration_dataset,
            calibration_indices,
            classes=seen_classes,
            label_map=seen_label_map,
        )
        fold_eval_dataset = FilteredLabelDataset(
            eval_dataset,
            eval_id_indices,
            classes=seen_classes,
            label_map=seen_label_map,
        )
        fold_ood_dataset = FilteredLabelDataset(
            eval_dataset,
            eval_ood_indices,
            classes=["__ood__"],
            constant_label=-1,
        )

        fold_train_loader = _clone_loader(train_loader, fold_train_dataset, training_like=True)
        fold_calibration_loader = _clone_loader(calibration_loader, fold_calibration_dataset, training_like=False)
        fold_eval_loader = _clone_loader(eval_loader, fold_eval_dataset, training_like=False)
        fold_ood_loader = _clone_loader(eval_loader, fold_ood_dataset, training_like=False)

        try:
            adapter = adapter_factory(
                crop_name=str(crop_name),
                model_name=str(
                    config.get("training", {})
                    .get("continual", {})
                    .get("backbone", {})
                    .get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")
                ),
                device=str(device),
            )
            adapter.initialize_engine(class_names=seen_classes, config=config)
            session = adapter.build_training_session(
                train_loader=fold_train_loader,
                num_epochs=num_epochs,
                val_loader=fold_calibration_loader,
                run_id=f"{run_id}_ood_fold_{held_out_class}",
                checkpoint_every_n_steps=0,
                checkpoint_on_exception=False,
            )
            session.run()
            adapter.calibrate_ood(fold_calibration_loader)
            trainer = getattr(session, "trainer", getattr(adapter, "_trainer", None))
            evaluation = evaluate_model_with_artifact_metrics(trainer, fold_eval_loader, ood_loader=fold_ood_loader)
            if evaluation is None:
                raise RuntimeError("No evaluation samples were produced for the fold.")
            artifacts = persist_validation_artifacts(
                artifact_root=artifact_root,
                y_true=evaluation.y_true,
                y_pred=evaluation.y_pred,
                classes=seen_classes,
                telemetry=telemetry,
                artifact_subdir=f"ood_benchmark/folds/{held_out_class}",
                require_ood=True,
                ood_labels=evaluation.ood_labels,
                ood_scores=evaluation.ood_scores,
                sure_ds_f1=evaluation.sure_ds_f1,
                conformal_empirical_coverage=evaluation.conformal_empirical_coverage,
                conformal_avg_set_size=evaluation.conformal_avg_set_size,
                context={
                    **base_context,
                    "held_out_class": held_out_class,
                    "seen_classes": list(seen_classes),
                    **sample_counts,
                    **dict(evaluation.context),
                },
            )
            metric_gate = dict(artifacts.get("metric_gate", {}))
            fold_payload = {
                "held_out_class": held_out_class,
                "status": "completed",
                "reason": "",
                "seen_classes": list(seen_classes),
                "metrics": dict(metric_gate.get("metrics", {})),
                "evaluation": dict(metric_gate.get("evaluation", {})),
                "sample_counts": sample_counts,
                "paths": {key: str(path) for key, path in dict(artifacts.get("paths", {})).items()},
            }
        except Exception as exc:
            fold_payload = {
                "held_out_class": held_out_class,
                "status": "failed",
                "reason": str(exc),
                "seen_classes": list(seen_classes),
                "metrics": {},
                "sample_counts": sample_counts,
                "paths": {},
            }

        folds.append(fold_payload)
        emit(
            "ood_benchmark_fold_completed",
            {
                **base_context,
                "fold_index": fold_index,
                "fold_total": len(resolved_classes),
                **fold_payload,
            },
        )

    successful_folds = [fold for fold in folds if fold.get("status") == "completed"]
    failed_folds = [fold for fold in folds if fold.get("status") != "completed"]
    metric_names = (
        "accuracy",
        "ood_auroc",
        "ood_false_positive_rate",
        "sure_ds_f1",
        "conformal_empirical_coverage",
        "conformal_avg_set_size",
    )
    aggregate_metrics: Dict[str, Optional[float]] = {}
    metric_std: Dict[str, Optional[float]] = {}
    for metric_name in metric_names:
        values = [
            float(fold["metrics"][metric_name])
            for fold in successful_folds
            if fold.get("metrics", {}).get(metric_name) is not None
        ]
        aggregate_metrics[metric_name] = _mean(values) if len(values) == len(successful_folds) else None
        metric_std[metric_name] = _std(values) if len(values) == len(successful_folds) else None

    ood_validation = validate_ood_metrics(aggregate_metrics, target_values, require_ood=True)
    passed = bool(not failed_folds and ood_validation["passed"])
    summary_payload = {
        "schema_version": "v6_ood_benchmark",
        "status": "completed" if not failed_folds else "failed",
        "passed": passed,
        "ood_evidence_source": "held_out_benchmark",
        "metrics": aggregate_metrics,
        "metric_std": metric_std,
        "evaluation": ood_validation,
        "successful_folds": len(successful_folds),
        "failed_folds": len(failed_folds),
        "folds": folds,
        "targets": target_values,
        "context": base_context,
    }
    artifacts = persist_ood_benchmark_artifacts(
        artifact_root=artifact_root,
        summary_payload=summary_payload,
        telemetry=telemetry,
    )
    summary_payload["paths"] = {key: str(path) for key, path in artifacts.items()}
    emit(
        "ood_benchmark_completed",
        {
            **base_context,
            "passed": summary_payload["passed"],
            "status": summary_payload["status"],
            "successful_folds": summary_payload["successful_folds"],
            "failed_folds": summary_payload["failed_folds"],
            "paths": summary_payload["paths"],
        },
    )
    return summary_payload
