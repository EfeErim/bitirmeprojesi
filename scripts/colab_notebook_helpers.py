#!/usr/bin/env python3
"""Notebook 2 helper functions only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.training.services.reporting import (
    persist_training_history_artifacts as persist_training_history_artifacts_core,
)
from src.training.services.reporting import (
    persist_validation_artifacts as persist_validation_artifacts_core,
)


def _artifact_dir(root: Path, *parts: str) -> Path:
    target = root / "outputs" / "colab_notebook_training" / "artifacts"
    for part in parts:
        target /= part
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_history_snapshot(
    *,
    state_history: Optional[Dict[str, Any]] = None,
    session_history: Optional[Dict[str, Any]] = None,
    train_loss_curve: List[float],
    val_loss_curve: List[float],
    val_acc_curve: List[float],
    macro_f1_curve: List[float],
    weighted_f1_curve: List[float],
    balanced_acc_curve: List[float],
    gap_curve: List[float],
) -> Dict[str, Any]:
    if session_history:
        merged = dict(session_history)
        merged.setdefault("per_class_accuracy", list((state_history or {}).get("per_class_accuracy", [])))
        merged.setdefault("worst_classes", list((state_history or {}).get("worst_classes", [])))
        return merged

    baseline = state_history or {}
    return {
        "train_loss": list(train_loss_curve),
        "val_loss": list(val_loss_curve),
        "val_accuracy": list(val_acc_curve),
        "macro_precision": list(baseline.get("macro_precision", [])),
        "macro_recall": list(baseline.get("macro_recall", [])),
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
    return persist_training_history_artifacts_core(
        artifact_root=_artifact_dir(root),
        history_snapshot=history_snapshot,
        telemetry=telemetry,
    )


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
    session: Any,
    reason: str,
    run_id: str,
    telemetry: Any = None,
    mark_best: bool = False,
    val_loss: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if checkpoint_manager is None:
        return None
    record = checkpoint_manager.save_checkpoint(
        adapter=adapter,
        session=session,
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
    gate_targets: Optional[Dict[str, float]] = None,
    require_ood: bool = False,
    ood_labels: Optional[List[int]] = None,
    ood_scores: Optional[List[float]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return persist_validation_artifacts_core(
        artifact_root=_artifact_dir(root),
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        telemetry=telemetry,
        gate_targets=gate_targets,
        require_ood=require_ood,
        ood_labels=ood_labels,
        ood_scores=ood_scores,
        context=context,
    )
