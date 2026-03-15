"""Artifact writers for training telemetry, plots, and validation reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.shared.artifacts import ArtifactStore
from src.training.services.metrics import (
    build_production_readiness,
    compute_plan_metrics,
    write_plan_metric_artifact,
)

_HISTORY_KEYS = [
    "train_loss",
    "val_loss",
    "val_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "generalization_gap",
]

_BATCH_KEYS = [
    "epoch",
    "batch",
    "global_step",
    "optimizer_steps",
    "loss",
    "lr",
    "grad_norm",
    "step_time_sec",
    "samples_per_sec",
    "batch_size",
    "accumulation_step",
    "optimizer_step_applied",
    "ber_ce_loss",
    "ber_old_loss",
    "ber_new_loss",
    "epoch_progress",
    "advisory",
    "severity",
]

_BATCH_INT_KEYS = {
    "epoch",
    "batch",
    "global_step",
    "optimizer_steps",
    "batch_size",
    "accumulation_step",
}
_BATCH_FLOAT_KEYS = {
    "loss",
    "lr",
    "grad_norm",
    "step_time_sec",
    "samples_per_sec",
    "ber_ce_loss",
    "ber_old_loss",
    "ber_new_loss",
    "epoch_progress",
}
_BATCH_BOOL_KEYS = {"optimizer_step_applied"}


def _artifact_dir(root: Path, *parts: str) -> Path:
    target = Path(root)
    for part in parts:
        target /= part
    target.mkdir(parents=True, exist_ok=True)
    return target


def _copy_to_telemetry(telemetry: Any, source_path: Path, relative_path: str) -> None:
    if telemetry is None or not hasattr(telemetry, "copy_artifact_file"):
        return
    telemetry.copy_artifact_file(source_path, relative_path)


def _copy_artifacts_to_telemetry(telemetry: Any, artifacts: Iterable[tuple[Path, str]]) -> None:
    for source_path, relative_path in artifacts:
        _copy_to_telemetry(telemetry, source_path, relative_path)


def _epoch_rows(history_snapshot: Dict[str, Any]) -> List[List[Any]]:
    max_len = max((len(history_snapshot.get(key, [])) for key in _HISTORY_KEYS), default=0)
    rows: List[List[Any]] = []
    for index in range(max_len):
        row: List[Any] = [index + 1]
        for key in _HISTORY_KEYS:
            values = history_snapshot.get(key, [])
            row.append(values[index] if index < len(values) else "")
        rows.append(row)
    return rows


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(list(row))
    return path


def _coerce_batch_csv_value(key: str, value: str) -> Any:
    if value == "":
        return ""
    if key in _BATCH_INT_KEYS:
        return int(value)
    if key in _BATCH_FLOAT_KEYS:
        return float(value)
    if key in _BATCH_BOOL_KEYS:
        lowered = str(value).strip().lower()
        return lowered in {"1", "true", "yes"}
    return value


def load_batch_metrics_history(batch_metrics_csv: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(batch_metrics_csv)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _coerce_batch_csv_value(key, value) for key, value in row.items() if key is not None})
    return rows


class BatchMetricsRecorder:
    """Append batch metrics to CSV in buffered chunks."""

    def __init__(self, *, artifact_root: Path, flush_interval: int = 64) -> None:
        training_dir = _artifact_dir(Path(artifact_root), "training")
        self.output_path = _write_csv(training_dir / "batch_metrics.csv", _BATCH_KEYS, [])
        self._flush_interval = max(1, int(flush_interval))
        self._pending_rows: List[List[Any]] = []

    def append(self, payload: Dict[str, Any]) -> None:
        row = [payload.get(key, "") for key in _BATCH_KEYS]
        self._pending_rows.append(row)
        if len(self._pending_rows) >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._pending_rows:
            return
        with self.output_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(self._pending_rows)
        self._pending_rows.clear()

    def close(self) -> None:
        self.flush()


def _resolve_output_subdirs(artifact_subdir: str, telemetry_subdir: str | None) -> tuple[str, str]:
    resolved_artifact_subdir = str(artifact_subdir or "validation").strip().strip("/\\") or "validation"
    resolved_telemetry_subdir = (
        str(telemetry_subdir).strip().strip("/\\") if telemetry_subdir is not None else resolved_artifact_subdir
    ) or resolved_artifact_subdir
    return resolved_artifact_subdir, resolved_telemetry_subdir


def _resolve_context_metric(value: Any, context: Dict[str, Any], key: str) -> Any:
    if value is not None:
        return value
    context_value = context.get(key)
    return None if context_value is None else float(context_value)


def _build_ood_evidence_summary(
    *,
    split_name: str,
    metrics: Dict[str, Any],
    context: Dict[str, Any],
    ood_type_breakdown: Dict[str, Any] | None,
) -> Dict[str, Any]:
    sample_counts = {
        "classification_samples": (
            None if metrics.get("classification_samples") is None else int(metrics["classification_samples"])
        ),
        "in_distribution_samples": (
            None if metrics.get("in_distribution_samples") is None else int(metrics["in_distribution_samples"])
        ),
        "ood_samples": None if metrics.get("ood_samples") is None else int(metrics["ood_samples"]),
    }
    resolved_breakdown = dict(ood_type_breakdown or {})
    ood_types = sorted(str(name) for name in resolved_breakdown.keys())
    return {
        "split_name": str(split_name),
        "primary_score_method": str(context.get("ood_primary_score_method", "ensemble") or "ensemble"),
        "score_methods": [str(name) for name in list(context.get("ood_score_methods", []))],
        "metrics": {
            "ood_auroc": metrics.get("ood_auroc"),
            "ood_false_positive_rate": metrics.get("ood_false_positive_rate"),
            "sure_ds_f1": metrics.get("sure_ds_f1"),
            "conformal_empirical_coverage": metrics.get("conformal_empirical_coverage"),
            "conformal_avg_set_size": metrics.get("conformal_avg_set_size"),
        },
        "sample_counts": sample_counts,
        "ood_types": ood_types,
        "ood_type_count": int(len(ood_types)),
        "ood_type_breakdown": resolved_breakdown,
    }


def _write_per_class_metrics_csv(path: Path, resolved_classes: Sequence[str], report_dict: Dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for class_name in resolved_classes:
            row = report_dict.get(class_name, {})
            writer.writerow(
                [
                    class_name,
                    row.get("precision", 0.0),
                    row.get("recall", 0.0),
                    row.get("f1-score", 0.0),
                    row.get("support", 0),
                ]
            )
    return path


def _render_confusion_matrix(
    *,
    matrix: Any,
    output_path: Path,
    resolved_classes: Sequence[str],
    title: str,
    normalize: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues", vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(resolved_classes)), resolved_classes, rotation=45, ha="right")
    plt.yticks(range(len(resolved_classes)), resolved_classes)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            label = f"{matrix[row_idx, col_idx]:.2f}" if normalize else str(int(matrix[row_idx, col_idx]))
            plt.text(col_idx, row_idx, label, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def persist_training_history_artifacts(
    *,
    artifact_root: Path,
    history_snapshot: Dict[str, Any],
    telemetry: Any = None,
) -> Dict[str, Path]:
    training_dir = _artifact_dir(artifact_root, "training")
    store = ArtifactStore(training_dir)
    history_json = store.write_json("history.json", history_snapshot)
    epoch_rows = _epoch_rows(history_snapshot)
    results_csv = _write_csv(training_dir / "results.csv", ["epoch", *_HISTORY_KEYS], epoch_rows)
    history_csv = _write_csv(training_dir / "history.csv", ["epoch", *_HISTORY_KEYS], epoch_rows)

    _copy_artifacts_to_telemetry(
        telemetry,
        [
            (history_json, "training/history.json"),
            (results_csv, "training/results.csv"),
            (history_csv, "training/history.csv"),
        ],
    )
    return {
        "history_json": history_json,
        "results_csv": results_csv,
        "history_csv": history_csv,
    }


def persist_batch_metrics_artifacts(
    *,
    artifact_root: Path,
    batch_history: Sequence[Dict[str, Any]] | None = None,
    batch_metrics_csv: Path | str | None = None,
    telemetry: Any = None,
) -> Dict[str, Path]:
    training_dir = _artifact_dir(artifact_root, "training")
    if batch_metrics_csv is not None:
        batch_csv = Path(batch_metrics_csv)
    else:
        history_rows = list(batch_history or [])
        extras = sorted({str(key) for row in history_rows for key in row.keys()} - set(_BATCH_KEYS))
        headers = [*_BATCH_KEYS, *extras]
        rows: List[List[Any]] = []
        for row in history_rows:
            rows.append([row.get(key, "") for key in headers])
        batch_csv = _write_csv(training_dir / "batch_metrics.csv", headers, rows)
    _copy_artifacts_to_telemetry(telemetry, [(batch_csv, "training/batch_metrics.csv")])
    return {"batch_metrics_csv": batch_csv}


def _plot_epoch_series(
    axis: Any,
    history_snapshot: Dict[str, Any],
    keys: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    plotted = False
    for key in keys:
        values = history_snapshot.get(key, [])
        if not values:
            continue
        axis.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.8, markersize=3, label=key)
        plotted = True
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)


def _plot_batch_series(axis: Any, batch_history: Sequence[Dict[str, Any]], key: str, title: str, ylabel: str) -> None:
    if not batch_history:
        axis.set_title(title)
        axis.set_xlabel("Global Step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        return

    steps = [int(row.get("global_step", index + 1)) for index, row in enumerate(batch_history)]
    values = [row.get(key, 0.0) for row in batch_history]
    axis.plot(steps, values, linewidth=1.4)
    axis.set_title(title)
    axis.set_xlabel("Global Step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)


def persist_training_results_figure(
    *,
    artifact_root: Path,
    history_snapshot: Dict[str, Any],
    batch_history: Sequence[Dict[str, Any]],
    telemetry: Any = None,
) -> Dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training_dir = _artifact_dir(artifact_root, "training")
    results_png = training_dir / "results.png"

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    _plot_epoch_series(axes[0, 0], history_snapshot, ["train_loss", "val_loss"], "Loss", "Loss")
    _plot_epoch_series(axes[0, 1], history_snapshot, ["val_accuracy", "balanced_accuracy"], "Accuracy", "Score")
    _plot_epoch_series(
        axes[0, 2],
        history_snapshot,
        ["macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
        "Precision / Recall / F1",
        "Score",
    )
    _plot_epoch_series(axes[1, 0], history_snapshot, ["generalization_gap"], "Generalization Gap", "Loss Delta")
    _plot_batch_series(axes[1, 1], batch_history, "loss", "Batch Loss", "Loss")
    _plot_batch_series(axes[1, 2], batch_history, "lr", "Learning Rate", "LR")
    _plot_batch_series(axes[2, 0], batch_history, "grad_norm", "Gradient Norm", "Norm")
    _plot_batch_series(axes[2, 1], batch_history, "samples_per_sec", "Throughput", "Samples / sec")
    _plot_batch_series(axes[2, 2], batch_history, "step_time_sec", "Step Time", "Seconds")
    fig.suptitle("Training Results", fontsize=16)
    fig.tight_layout()
    fig.savefig(results_png, dpi=150)
    plt.close(fig)

    _copy_artifacts_to_telemetry(telemetry, [(results_png, "training/results.png")])
    return {"results_png": results_png}


def persist_training_summary_artifact(
    *,
    artifact_root: Path,
    summary_payload: Dict[str, Any],
    telemetry: Any = None,
) -> Dict[str, Path]:
    training_dir = _artifact_dir(artifact_root, "training")
    summary_json = ArtifactStore(training_dir).write_json("summary.json", summary_payload)
    _copy_artifacts_to_telemetry(telemetry, [(summary_json, "training/summary.json")])
    return {"summary_json": summary_json}


def persist_ood_benchmark_artifacts(
    *,
    artifact_root: Path,
    summary_payload: Dict[str, Any],
    telemetry: Any = None,
) -> Dict[str, Path]:
    benchmark_dir = _artifact_dir(artifact_root, "ood_benchmark")
    store = ArtifactStore(benchmark_dir)
    summary_json = store.write_json("summary.json", summary_payload)
    folds = list(summary_payload.get("folds", [])) if isinstance(summary_payload, dict) else []
    headers = [
        "held_out_class",
        "status",
        "reason",
        "primary_score_method",
        "accuracy",
        "ood_auroc",
        "ood_false_positive_rate",
        "ensemble_ood_auroc",
        "ensemble_ood_false_positive_rate",
        "energy_ood_auroc",
        "energy_ood_false_positive_rate",
        "knn_ood_auroc",
        "knn_ood_false_positive_rate",
        "sure_ds_f1",
        "conformal_empirical_coverage",
        "conformal_avg_set_size",
        "train_samples",
        "calibration_samples",
        "eval_in_distribution_samples",
        "eval_ood_samples",
        "resume_key",
    ]
    rows: List[List[Any]] = []
    for fold in folds:
        metrics = dict(fold.get("metrics", {}))
        method_metrics = dict(fold.get("method_metrics", {}))
        sample_counts = dict(fold.get("sample_counts", {}))
        rows.append(
            [
                fold.get("held_out_class", ""),
                fold.get("status", ""),
                fold.get("reason", ""),
                fold.get("primary_score_method", ""),
                metrics.get("accuracy", ""),
                metrics.get("ood_auroc", ""),
                metrics.get("ood_false_positive_rate", ""),
                dict(method_metrics.get("ensemble", {})).get("ood_auroc", ""),
                dict(method_metrics.get("ensemble", {})).get("ood_false_positive_rate", ""),
                dict(method_metrics.get("energy", {})).get("ood_auroc", ""),
                dict(method_metrics.get("energy", {})).get("ood_false_positive_rate", ""),
                dict(method_metrics.get("knn", {})).get("ood_auroc", ""),
                dict(method_metrics.get("knn", {})).get("ood_false_positive_rate", ""),
                metrics.get("sure_ds_f1", ""),
                metrics.get("conformal_empirical_coverage", ""),
                metrics.get("conformal_avg_set_size", ""),
                sample_counts.get("train_samples", ""),
                sample_counts.get("calibration_samples", ""),
                sample_counts.get("eval_in_distribution_samples", ""),
                sample_counts.get("eval_ood_samples", ""),
                fold.get("resume_key", ""),
            ]
        )
    per_fold_csv = _write_csv(benchmark_dir / "per_fold.csv", headers, rows)
    _copy_artifacts_to_telemetry(
        telemetry,
        [
            (summary_json, "ood_benchmark/summary.json"),
            (per_fold_csv, "ood_benchmark/per_fold.csv"),
        ],
    )
    return {"summary_json": summary_json, "per_fold_csv": per_fold_csv}


def persist_production_readiness_artifact(
    *,
    artifact_root: Path,
    classification_metric_gate: Dict[str, Any] | None,
    classification_split: str,
    ood_evidence_source: str | None,
    ood_metrics: Dict[str, Any] | None,
    targets: Dict[str, float] | None = None,
    context: Dict[str, Any] | None = None,
    require_ood: bool = True,
    telemetry: Any = None,
) -> Dict[str, Any]:
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = build_production_readiness(
        classification_metric_gate=classification_metric_gate,
        classification_split=classification_split,
        ood_evidence_source=ood_evidence_source,
        ood_metrics=ood_metrics,
        targets=targets,
        context=context,
        require_ood=require_ood,
    )
    readiness_json = ArtifactStore(artifact_root).write_json("production_readiness.json", payload)
    _copy_artifacts_to_telemetry(telemetry, [(readiness_json, "production_readiness.json")])
    return {"payload": payload, "readiness_json": readiness_json}


def persist_validation_artifacts(
    *,
    artifact_root: Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    telemetry: Any = None,
    artifact_subdir: str = "validation",
    telemetry_subdir: str | None = None,
    gate_targets: Dict[str, float] | None = None,
    require_ood: bool = False,
    emit_metric_gate: bool = True,
    ood_labels: Sequence[int] | None = None,
    ood_scores: Sequence[float] | None = None,
    sure_ds_f1: float | None = None,
    conformal_empirical_coverage: float | None = None,
    conformal_avg_set_size: float | None = None,
    ood_type_breakdown: Dict[str, Any] | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_classes = [str(name) for name in classes]
    labels = list(range(len(resolved_classes)))
    resolved_artifact_subdir, resolved_telemetry_subdir = _resolve_output_subdirs(
        artifact_subdir,
        telemetry_subdir,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=resolved_classes,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=resolved_classes,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1.0, None)

    validation_dir = _artifact_dir(artifact_root, resolved_artifact_subdir)
    store = ArtifactStore(validation_dir)
    report_txt = store.write_text("classification_report.txt", report_text)
    report_json = store.write_json("classification_report.json", report_dict)
    confusion_csv = validation_dir / "confusion_matrix.csv"
    np.savetxt(confusion_csv, cm, delimiter=",", fmt="%d")
    per_class_csv = validation_dir / "per_class_metrics.csv"
    confusion_png = validation_dir / "confusion_matrix.png"
    confusion_norm_png = validation_dir / "confusion_matrix_normalized.png"
    _write_per_class_metrics_csv(per_class_csv, resolved_classes, report_dict)
    _render_confusion_matrix(
        matrix=cm,
        output_path=confusion_png,
        resolved_classes=resolved_classes,
        title="Validation Confusion Matrix",
        normalize=False,
    )
    _render_confusion_matrix(
        matrix=cm_norm,
        output_path=confusion_norm_png,
        resolved_classes=resolved_classes,
        title="Validation Confusion Matrix (Normalized)",
        normalize=True,
    )

    paths = {
        "report_txt": report_txt,
        "report_json": report_json,
        "per_class_csv": per_class_csv,
        "cm_csv": confusion_csv,
        "cm_png": confusion_png,
        "cm_norm_png": confusion_norm_png,
    }
    _copy_artifacts_to_telemetry(
        telemetry,
        [
            (report_txt, f"{resolved_telemetry_subdir}/classification_report.txt"),
            (report_json, f"{resolved_telemetry_subdir}/classification_report.json"),
            (per_class_csv, f"{resolved_telemetry_subdir}/per_class_metrics.csv"),
            (confusion_csv, f"{resolved_telemetry_subdir}/confusion_matrix.csv"),
            (confusion_png, f"{resolved_telemetry_subdir}/confusion_matrix.png"),
            (confusion_norm_png, f"{resolved_telemetry_subdir}/confusion_matrix_normalized.png"),
        ],
    )

    if isinstance(ood_type_breakdown, dict) and ood_type_breakdown:
        breakdown_json = store.write_json("ood_type_breakdown.json", ood_type_breakdown)
        paths["ood_type_breakdown_json"] = breakdown_json
        _copy_artifacts_to_telemetry(
            telemetry,
            [(breakdown_json, f"{resolved_telemetry_subdir}/ood_type_breakdown.json")],
        )

    metric_context = dict(context or {"num_classes": len(resolved_classes)})
    if isinstance(ood_type_breakdown, dict) and ood_type_breakdown:
        metric_context["ood_type_breakdown"] = dict(ood_type_breakdown)
    metrics = compute_plan_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ood_labels=ood_labels,
        ood_scores=ood_scores,
        sure_ds_f1=_resolve_context_metric(sure_ds_f1, metric_context, "sure_ds_f1"),
        conformal_empirical_coverage=_resolve_context_metric(
            conformal_empirical_coverage,
            metric_context,
            "conformal_empirical_coverage",
        ),
        conformal_avg_set_size=_resolve_context_metric(
            conformal_avg_set_size,
            metric_context,
            "conformal_avg_set_size",
        ),
    )
    if metrics.get("ood_samples") is not None or metrics.get("in_distribution_samples") is not None:
        ood_evidence_summary = _build_ood_evidence_summary(
            split_name=resolved_artifact_subdir,
            metrics=metrics,
            context=metric_context,
            ood_type_breakdown=ood_type_breakdown,
        )
        summary_json = store.write_json("ood_evidence_summary.json", ood_evidence_summary)
        paths["ood_evidence_summary_json"] = summary_json
        _copy_artifacts_to_telemetry(
            telemetry,
            [(summary_json, f"{resolved_telemetry_subdir}/ood_evidence_summary.json")],
        )

    metric_gate = write_plan_metric_artifact(
        output_path=validation_dir / "metric_gate.json",
        metrics=metrics,
        targets=gate_targets,
        require_ood=require_ood,
        context=metric_context,
    )
    metric_gate_json = validation_dir / "metric_gate.json"
    if emit_metric_gate:
        paths["metric_gate_json"] = metric_gate_json
        _copy_artifacts_to_telemetry(telemetry, [(metric_gate_json, f"{resolved_telemetry_subdir}/metric_gate.json")])
    else:
        metric_gate_json.unlink(missing_ok=True)

    return {
        "report_text": report_text,
        "report_dict": report_dict,
        "cm": cm,
        "cm_norm": cm_norm,
        "metric_gate": metric_gate,
        "paths": paths,
    }
