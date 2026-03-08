"""Artifact writers for training telemetry, plots, and validation reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.shared.artifacts import ArtifactStore
from src.training.services.metrics import compute_plan_metrics, write_plan_metric_artifact

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
    "epoch_progress",
    "advisory",
    "severity",
]


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

    _copy_to_telemetry(telemetry, history_json, "training/history.json")
    _copy_to_telemetry(telemetry, results_csv, "training/results.csv")
    _copy_to_telemetry(telemetry, history_csv, "training/history.csv")
    return {
        "history_json": history_json,
        "results_csv": results_csv,
        "history_csv": history_csv,
    }


def persist_batch_metrics_artifacts(
    *,
    artifact_root: Path,
    batch_history: Sequence[Dict[str, Any]],
    telemetry: Any = None,
) -> Dict[str, Path]:
    training_dir = _artifact_dir(artifact_root, "training")
    extras = sorted({str(key) for row in batch_history for key in row.keys()} - set(_BATCH_KEYS))
    headers = [*_BATCH_KEYS, *extras]
    rows: List[List[Any]] = []
    for row in batch_history:
        rows.append([row.get(key, "") for key in headers])

    batch_csv = _write_csv(training_dir / "batch_metrics.csv", headers, rows)
    _copy_to_telemetry(telemetry, batch_csv, "training/batch_metrics.csv")
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

    _copy_to_telemetry(telemetry, results_png, "training/results.png")
    return {"results_png": results_png}


def persist_training_summary_artifact(
    *,
    artifact_root: Path,
    summary_payload: Dict[str, Any],
    telemetry: Any = None,
) -> Dict[str, Path]:
    training_dir = _artifact_dir(artifact_root, "training")
    summary_json = ArtifactStore(training_dir).write_json("summary.json", summary_payload)
    _copy_to_telemetry(telemetry, summary_json, "training/summary.json")
    return {"summary_json": summary_json}


def persist_validation_artifacts(
    *,
    artifact_root: Path,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    telemetry: Any = None,
    gate_targets: Dict[str, float] | None = None,
    require_ood: bool = False,
    ood_labels: Sequence[int] | None = None,
    ood_scores: Sequence[float] | None = None,
    sure_ds_f1: float | None = None,
    conformal_empirical_coverage: float | None = None,
    conformal_avg_set_size: float | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_classes = [str(name) for name in classes]
    labels = list(range(len(resolved_classes)))
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

    validation_dir = _artifact_dir(artifact_root, "validation")
    store = ArtifactStore(validation_dir)
    report_txt = store.write_text("classification_report.txt", report_text)
    report_json = store.write_json("classification_report.json", report_dict)
    confusion_csv = validation_dir / "confusion_matrix.csv"
    np.savetxt(confusion_csv, cm, delimiter=",", fmt="%d")
    per_class_csv = validation_dir / "per_class_metrics.csv"
    metric_gate_json = validation_dir / "metric_gate.json"
    confusion_png = validation_dir / "confusion_matrix.png"
    confusion_norm_png = validation_dir / "confusion_matrix_normalized.png"

    with per_class_csv.open("w", encoding="utf-8", newline="") as handle:
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

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for matrix, path, title, normalize in (
        (cm, confusion_png, "Validation Confusion Matrix", False),
        (cm_norm, confusion_norm_png, "Validation Confusion Matrix (Normalized)", True),
    ):
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
        plt.savefig(path, dpi=150)
        plt.close()

    _copy_to_telemetry(telemetry, report_txt, "validation/classification_report.txt")
    _copy_to_telemetry(telemetry, report_json, "validation/classification_report.json")
    _copy_to_telemetry(telemetry, per_class_csv, "validation/per_class_metrics.csv")
    _copy_to_telemetry(telemetry, confusion_csv, "validation/confusion_matrix.csv")
    _copy_to_telemetry(telemetry, confusion_png, "validation/confusion_matrix.png")
    _copy_to_telemetry(telemetry, confusion_norm_png, "validation/confusion_matrix_normalized.png")

    metric_context = dict(context or {"num_classes": len(resolved_classes)})
    metrics = compute_plan_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ood_labels=ood_labels,
        ood_scores=ood_scores,
        sure_ds_f1=(
            sure_ds_f1
            if sure_ds_f1 is not None
            else (
                float(metric_context["sure_ds_f1"])
                if metric_context.get("sure_ds_f1") is not None
                else None
            )
        ),
        conformal_empirical_coverage=(
            conformal_empirical_coverage
            if conformal_empirical_coverage is not None
            else (
                float(metric_context["conformal_empirical_coverage"])
                if metric_context.get("conformal_empirical_coverage") is not None
                else None
            )
        ),
        conformal_avg_set_size=(
            conformal_avg_set_size
            if conformal_avg_set_size is not None
            else (
                float(metric_context["conformal_avg_set_size"])
                if metric_context.get("conformal_avg_set_size") is not None
                else None
            )
        ),
    )
    metric_gate = write_plan_metric_artifact(
        output_path=metric_gate_json,
        metrics=metrics,
        targets=gate_targets,
        require_ood=require_ood,
        context=metric_context,
    )
    _copy_to_telemetry(telemetry, metric_gate_json, "validation/metric_gate.json")

    return {
        "report_text": report_text,
        "report_dict": report_dict,
        "cm": cm,
        "cm_norm": cm_norm,
        "metric_gate": metric_gate,
        "paths": {
            "report_txt": report_txt,
            "report_json": report_json,
            "per_class_csv": per_class_csv,
            "cm_csv": confusion_csv,
            "cm_png": confusion_png,
            "cm_norm_png": confusion_norm_png,
            "metric_gate_json": metric_gate_json,
        },
    }
