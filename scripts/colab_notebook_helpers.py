#!/usr/bin/env python3
"""Notebook 2 helper functions only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def _artifact_dir(root: Path, *parts: str) -> Path:
    target = root / "outputs" / "colab_notebook_training" / "artifacts"
    for part in parts:
        target /= part
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_history_snapshot(
    *,
    state_history: Optional[Dict[str, Any]],
    train_loss_curve: List[float],
    val_loss_curve: List[float],
    val_acc_curve: List[float],
    macro_f1_curve: List[float],
    weighted_f1_curve: List[float],
    balanced_acc_curve: List[float],
    gap_curve: List[float],
) -> Dict[str, Any]:
    baseline = state_history or {}
    return {
        "train_loss": list(train_loss_curve),
        "val_loss": list(val_loss_curve),
        "val_accuracy": list(val_acc_curve),
        "macro_f1": list(macro_f1_curve),
        "weighted_f1": list(weighted_f1_curve),
        "balanced_accuracy": list(balanced_acc_curve),
        "generalization_gap": list(gap_curve),
        "per_class_accuracy": list(baseline.get("per_class_accuracy", [])),
        "worst_classes": list(baseline.get("worst_classes", [])),
    }


def persist_training_history_artifacts(*, root: Path, history_snapshot: Dict[str, Any], telemetry: Any = None) -> Dict[str, Path]:
    train_dir = _artifact_dir(root, "training")
    history_json = train_dir / "history.json"
    history_csv = train_dir / "history.csv"

    history_json.write_text(json.dumps(history_snapshot, indent=2), encoding="utf-8")

    keys = [
        "train_loss",
        "val_loss",
        "val_accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "generalization_gap",
    ]
    max_len = max((len(history_snapshot.get(key, [])) for key in keys), default=0)
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for index in range(max_len):
            row = [index + 1]
            for key in keys:
                values = history_snapshot.get(key, [])
                row.append(values[index] if index < len(values) else "")
            writer.writerow(row)

    if telemetry is not None:
        telemetry.copy_artifact_file(history_json, "training/history.json")
        telemetry.copy_artifact_file(history_csv, "training/history.csv")
    return {"history_json": history_json, "history_csv": history_csv}


def persist_training_curve_figure(*, root: Path, epoch_done: int, telemetry: Any = None) -> Dict[str, Path]:
    train_dir = _artifact_dir(root, "training")
    latest_curve = train_dir / "training_curves_latest.png"
    epoch_curve = train_dir / f"training_curves_epoch_{int(epoch_done):03d}.png"
    plt.savefig(latest_curve, dpi=150)
    plt.savefig(epoch_curve, dpi=150)
    if telemetry is not None:
        telemetry.copy_artifact_file(latest_curve, "training/training_curves_latest.png")
        telemetry.copy_artifact_file(epoch_curve, f"training/training_curves_epoch_{int(epoch_done):03d}.png")
    return {"latest_curve": latest_curve, "epoch_curve": epoch_curve}


def save_notebook_checkpoint(
    *,
    checkpoint_manager: Any,
    adapter: Any,
    reason: str,
    event: Dict[str, Any],
    history_snapshot: Dict[str, Any],
    run_id: str,
    telemetry: Any = None,
    mark_best: bool = False,
    val_loss: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if checkpoint_manager is None:
        return None
    progress = {
        "epoch": int(event.get("epoch_done", event.get("epoch", 0))),
        "batch": int(event.get("batch", 0)),
        "global_step": int(event.get("global_step", 0)),
        "elapsed_sec": float(event.get("elapsed_sec", 0.0)),
    }
    record = checkpoint_manager.save_checkpoint(
        adapter=adapter,
        progress_state=progress,
        history=history_snapshot,
        reason=reason,
        run_id=run_id,
        mark_best=bool(mark_best),
        val_loss=(float(val_loss) if val_loss is not None else None),
    )
    if telemetry is not None:
        telemetry.emit_event("checkpoint_saved", dict(record), phase="checkpoint")
    return record


def persist_validation_artifacts(
    *,
    root: Path,
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    telemetry: Any = None,
) -> Dict[str, Any]:
    report_text = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1.0, None)

    output_dir = _artifact_dir(root, "validation")
    report_txt = output_dir / "classification_report.txt"
    report_json = output_dir / "classification_report.json"
    per_class_csv = output_dir / "per_class_metrics.csv"
    confusion_csv = output_dir / "confusion_matrix.csv"
    confusion_png = output_dir / "confusion_matrix.png"
    confusion_norm_png = output_dir / "confusion_matrix_normalized.png"

    report_txt.write_text(report_text, encoding="utf-8")
    report_json.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    np.savetxt(confusion_csv, cm, delimiter=",", fmt="%d")

    with per_class_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for class_name in classes:
            row = report_dict.get(class_name, {})
            writer.writerow([class_name, row.get("precision", 0.0), row.get("recall", 0.0), row.get("f1-score", 0.0), row.get("support", 0)])

    for matrix, path, title, normalize in (
        (cm, confusion_png, "Validation Confusion Matrix", False),
        (cm_norm, confusion_norm_png, "Validation Confusion Matrix (Normalized)", True),
    ):
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, cmap="Blues", vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
        plt.yticks(range(len(classes)), classes)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                label = f"{matrix[row_idx, col_idx]:.2f}" if normalize else str(int(matrix[row_idx, col_idx]))
                plt.text(col_idx, row_idx, label, ha="center", va="center")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    if telemetry is not None:
        telemetry.copy_artifact_file(report_txt, "validation/classification_report.txt")
        telemetry.copy_artifact_file(report_json, "validation/classification_report.json")
        telemetry.copy_artifact_file(per_class_csv, "validation/per_class_metrics.csv")
        telemetry.copy_artifact_file(confusion_csv, "validation/confusion_matrix.csv")
        telemetry.copy_artifact_file(confusion_png, "validation/confusion_matrix.png")
        telemetry.copy_artifact_file(confusion_norm_png, "validation/confusion_matrix_normalized.png")

    return {
        "report_text": report_text,
        "report_dict": report_dict,
        "cm": cm,
        "cm_norm": cm_norm,
        "paths": {
            "report_txt": report_txt,
            "report_json": report_json,
            "per_class_csv": per_class_csv,
            "cm_csv": confusion_csv,
            "cm_png": confusion_png,
            "cm_norm_png": confusion_norm_png,
        },
    }
