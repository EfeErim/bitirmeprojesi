#!/usr/bin/env python3
"""Reusable helpers for Colab notebook telemetry, artifacts, and checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def _artifact_dir(root: Path, *parts: str) -> Path:
    target = root / "outputs" / "colab_notebook_training" / "artifacts"
    for part in parts:
        target = target / part
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


def persist_training_history_artifacts(
    *,
    root: Path,
    history_snapshot: Dict[str, Any],
    telemetry: Any = None,
) -> Dict[str, Path]:
    train_dir = _artifact_dir(root, "training")

    history_json = train_dir / "history.json"
    history_json.write_text(json.dumps(history_snapshot, indent=2), encoding="utf-8")

    train_loss_curve = history_snapshot.get("train_loss", [])
    val_loss_curve = history_snapshot.get("val_loss", [])
    val_acc_curve = history_snapshot.get("val_accuracy", [])
    macro_f1_curve = history_snapshot.get("macro_f1", [])
    weighted_f1_curve = history_snapshot.get("weighted_f1", [])
    balanced_acc_curve = history_snapshot.get("balanced_accuracy", [])
    gap_curve = history_snapshot.get("generalization_gap", [])

    max_len = max(
        len(train_loss_curve),
        len(val_loss_curve),
        len(val_acc_curve),
        len(macro_f1_curve),
        len(weighted_f1_curve),
        len(balanced_acc_curve),
        len(gap_curve),
        0,
    )
    history_csv = train_dir / "history.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
                "generalization_gap",
            ]
        )
        for i in range(max_len):
            writer.writerow(
                [
                    i + 1,
                    train_loss_curve[i] if i < len(train_loss_curve) else "",
                    val_loss_curve[i] if i < len(val_loss_curve) else "",
                    val_acc_curve[i] if i < len(val_acc_curve) else "",
                    macro_f1_curve[i] if i < len(macro_f1_curve) else "",
                    weighted_f1_curve[i] if i < len(weighted_f1_curve) else "",
                    balanced_acc_curve[i] if i < len(balanced_acc_curve) else "",
                    gap_curve[i] if i < len(gap_curve) else "",
                ]
            )

    if telemetry is not None:
        telemetry.copy_artifact_file(history_json, "training/history.json")
        telemetry.copy_artifact_file(history_csv, "training/history.csv")

    return {"history_json": history_json, "history_csv": history_csv}


def persist_training_curve_figure(
    *,
    root: Path,
    epoch_done: int,
    telemetry: Any = None,
) -> Dict[str, Path]:
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


def persist_ood_artifacts(*, root: Path, calibration: Dict[str, Any], thresholds: Dict[str, float], telemetry: Any = None) -> Dict[str, Path]:
    ood_dir = _artifact_dir(root, "ood")
    calibration_path = ood_dir / "calibration.json"
    threshold_path = ood_dir / "thresholds.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    threshold_path.write_text(json.dumps({"thresholds": thresholds}, indent=2), encoding="utf-8")
    if telemetry is not None:
        telemetry.copy_artifact_file(calibration_path, "ood/calibration.json")
        telemetry.copy_artifact_file(threshold_path, "ood/thresholds.json")
    return {"calibration": calibration_path, "thresholds": threshold_path}


def persist_adapter_artifacts(*, root: Path, asset_dir: Path, telemetry: Any = None) -> List[str]:
    files: List[str] = []
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        files.append(rel)
        if telemetry is not None:
            telemetry.copy_artifact_file(path, f"adapter/{path.name}")
    return files


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

    val_dir = _artifact_dir(root, "validation")
    report_txt_path = val_dir / "classification_report.txt"
    report_json_path = val_dir / "classification_report.json"
    per_class_csv_path = val_dir / "per_class_metrics.csv"
    cm_npy_path = val_dir / "confusion_matrix.npy"
    cm_csv_path = val_dir / "confusion_matrix.csv"
    cm_png_path = val_dir / "confusion_matrix.png"
    cm_norm_png_path = val_dir / "confusion_matrix_normalized.png"

    report_txt_path.write_text(report_text, encoding="utf-8")
    report_json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    np.save(cm_npy_path, cm)
    np.savetxt(cm_csv_path, cm, delimiter=",", fmt="%d")

    with per_class_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for class_name in classes:
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

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=150)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("Validation Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_norm_png_path, dpi=150)
    plt.show()

    if telemetry is not None:
        telemetry.copy_artifact_file(report_txt_path, "validation/classification_report.txt")
        telemetry.copy_artifact_file(report_json_path, "validation/classification_report.json")
        telemetry.copy_artifact_file(per_class_csv_path, "validation/per_class_metrics.csv")
        telemetry.copy_artifact_file(cm_npy_path, "validation/confusion_matrix.npy")
        telemetry.copy_artifact_file(cm_csv_path, "validation/confusion_matrix.csv")
        telemetry.copy_artifact_file(cm_png_path, "validation/confusion_matrix.png")
        telemetry.copy_artifact_file(cm_norm_png_path, "validation/confusion_matrix_normalized.png")

    return {
        "report_text": report_text,
        "report_dict": report_dict,
        "cm": cm,
        "cm_norm": cm_norm,
        "paths": {
            "report_txt": report_txt_path,
            "report_json": report_json_path,
            "per_class_csv": per_class_csv_path,
            "cm_npy": cm_npy_path,
            "cm_csv": cm_csv_path,
            "cm_png": cm_png_path,
            "cm_norm_png": cm_norm_png_path,
        },
    }


def persist_router_analysis_artifacts(
    *,
    root: Path,
    analysis: Dict[str, Any],
    rows: List[Dict[str, Any]],
    telemetry: Any = None,
) -> Dict[str, Any]:
    router_dir = _artifact_dir(root, "router")
    detections_json = router_dir / "detections_latest.json"
    detections_csv = router_dir / "detections_table_latest.csv"
    confidence_hist_png = router_dir / "confidence_histogram.png"
    crop_dist_png = router_dir / "crop_distribution.png"
    part_dist_png = router_dir / "part_distribution.png"

    detections_json.write_text(json.dumps({"analysis": analysis, "rows": rows}, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(detections_csv, index=False)

    if rows:
        frame = pd.DataFrame(rows)
        conf_values = [float(item["Confidence"]) for item in rows]
        plt.figure(figsize=(5, 3))
        plt.hist(conf_values, bins=min(10, len(conf_values)), color="teal", alpha=0.8)
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.title("Detection Confidence Histogram")
        plt.tight_layout()
        plt.savefig(confidence_hist_png, dpi=150)
        plt.close()

        crop_counts = frame.groupby("Crop").size().sort_values(ascending=False)
        plt.figure(figsize=(6, 3))
        crop_counts.plot(kind="bar", color="steelblue")
        plt.ylabel("Count")
        plt.title("Crop Distribution")
        plt.tight_layout()
        plt.savefig(crop_dist_png, dpi=150)
        plt.close()

        part_counts = frame.groupby("Part").size().sort_values(ascending=False)
        plt.figure(figsize=(6, 3))
        part_counts.plot(kind="bar", color="darkorange")
        plt.ylabel("Count")
        plt.title("Part Distribution")
        plt.tight_layout()
        plt.savefig(part_dist_png, dpi=150)
        plt.close()

    if telemetry is not None:
        telemetry.copy_artifact_file(detections_json, "router/detections_latest.json")
        telemetry.copy_artifact_file(detections_csv, "router/detections_table_latest.csv")
        if confidence_hist_png.exists():
            telemetry.copy_artifact_file(confidence_hist_png, "router/confidence_histogram.png")
        if crop_dist_png.exists():
            telemetry.copy_artifact_file(crop_dist_png, "router/crop_distribution.png")
        if part_dist_png.exists():
            telemetry.copy_artifact_file(part_dist_png, "router/part_distribution.png")

    return {
        "detection_count": len(rows),
        "paths": {
            "detections_json": detections_json,
            "detections_csv": detections_csv,
            "confidence_hist_png": confidence_hist_png,
            "crop_dist_png": crop_dist_png,
            "part_dist_png": part_dist_png,
        },
    }


def persist_pipeline_artifacts(*, root: Path, full_result: Dict[str, Any], telemetry: Any = None) -> Dict[str, Path]:
    pipeline_dir = _artifact_dir(root, "pipeline")
    full_result_json = pipeline_dir / "full_result_latest.json"
    full_result_json.write_text(json.dumps(full_result, indent=2, default=str), encoding="utf-8")
    if telemetry is not None:
        telemetry.copy_artifact_file(full_result_json, "pipeline/full_result_latest.json")
    return {"full_result_json": full_result_json}

