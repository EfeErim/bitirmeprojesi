#!/usr/bin/env python3
"""Notebook 2 helper functions only."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


def ensure_notebook_checkpoint_manager(
    checkpoint_manager: Any = None,
    *,
    run_id: Optional[str] = None,
    drive_root: Optional[str | Path] = None,
    retention: int = 3,
) -> Any:
    if checkpoint_manager is not None:
        return checkpoint_manager

    from scripts.colab_checkpointing import TrainingCheckpointManager

    resolved_run_id = str(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    resolved_drive_root = Path(
        drive_root or os.environ.get("AADS_DRIVE_LOG_ROOT", "/content/drive/MyDrive/aads_ulora")
    )
    return TrainingCheckpointManager(resolved_drive_root / "telemetry" / resolved_run_id, retention=retention)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0.0))))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


class NotebookTrainingStatusPrinter:
    """Emit low-frequency, notebook-friendly training status lines."""

    def __init__(
        self,
        *,
        total_epochs: int,
        batch_interval: int = 50,
        min_interval_sec: float = 15.0,
        print_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.total_epochs = int(max(1, total_epochs))
        self.batch_interval = int(max(0, batch_interval))
        self.min_interval_sec = float(max(1.0, min_interval_sec))
        self.print_fn = print if print_fn is None else print_fn
        self._last_batch_emit_elapsed = -1.0

    def _emit(self, message: str) -> None:
        self.print_fn(str(message))

    def _metric_fragment(self, payload: Dict[str, Any], key: str, label: str) -> Optional[str]:
        value = payload.get(key)
        if value is None:
            return None
        return f"{label}={float(value):.4f}"

    def handle(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event_name = str(event_type or "")
        event = dict(payload or {})
        if event_name == "batch_end":
            self._handle_batch_end(event)
            return
        if event_name == "validation_end":
            self._handle_validation_end(event)
            return
        if event_name == "best_metric_updated":
            self._handle_best_metric(event)
            return
        if event_name == "stop_requested":
            self._handle_stop_requested(event)

    def _handle_batch_end(self, payload: Dict[str, Any]) -> None:
        batch = int(payload.get("batch", 0))
        if batch <= 0:
            return
        total_batches = int(payload.get("total_batches", 0))
        elapsed_sec = float(payload.get("elapsed_sec", 0.0))
        emit_due_to_interval = self.batch_interval > 0 and (batch % self.batch_interval == 0)
        emit_due_to_time = (
            self._last_batch_emit_elapsed < 0
            or (elapsed_sec - self._last_batch_emit_elapsed) >= self.min_interval_sec
        )
        emit_due_to_terminal_batch = total_batches > 0 and batch >= total_batches
        if not (batch == 1 or emit_due_to_interval or emit_due_to_time or emit_due_to_terminal_batch):
            return

        self._last_batch_emit_elapsed = elapsed_sec
        epoch = int(payload.get("epoch", 0))
        parts = [
            f"[LIVE] {epoch}/{self.total_epochs}",
            f"batch={batch}/{total_batches or '?'}",
            f"loss={float(payload.get('loss', 0.0)):.4f}",
            f"lr={float(payload.get('lr', 0.0)):.6f}",
            f"throughput={float(payload.get('samples_per_sec', 0.0)):.1f}/s",
            f"elapsed={_format_duration(elapsed_sec)}",
            f"eta={_format_duration(float(payload.get('eta_sec', 0.0)))}",
        ]
        advisory = str(payload.get("advisory", "")).strip()
        severity = str(payload.get("severity", "")).strip().lower()
        if advisory and severity in {"warning", "critical"}:
            parts.append(f"{severity}={advisory}")
        self._emit(" ".join(parts))

    def _handle_validation_end(self, payload: Dict[str, Any]) -> None:
        epoch_done = int(payload.get("epoch_done", 0))
        parts = [f"[VALID] {epoch_done}/{self.total_epochs}"]
        for key, label in (
            ("val_loss", "val_loss"),
            ("val_accuracy", "val_acc"),
            ("macro_f1", "macro_f1"),
            ("balanced_accuracy", "bal_acc"),
            ("generalization_gap", "gap"),
        ):
            metric = self._metric_fragment(payload, key, label)
            if metric is not None:
                parts.append(metric)
        advisory = str(payload.get("epoch_advisory", "")).strip()
        severity = str(payload.get("epoch_severity", "")).strip().lower()
        if advisory and severity in {"warning", "critical"}:
            parts.append(f"{severity}={advisory}")
        self._emit(" ".join(parts))

    def _handle_best_metric(self, payload: Dict[str, Any]) -> None:
        metric_name = str(payload.get("best_metric_name", "metric"))
        metric_value = payload.get("best_metric_value")
        if metric_value is None:
            return
        epoch_done = int(payload.get("epoch_done", 0))
        self._emit(f"[BEST] {epoch_done}/{self.total_epochs} {metric_name}={float(metric_value):.4f}")

    def _handle_stop_requested(self, payload: Dict[str, Any]) -> None:
        reason = str(payload.get("reason", "requested"))
        epoch = int(payload.get("epoch", 0))
        step = int(payload.get("global_step", 0))
        self._emit(f"[STOP] epoch={epoch} step={step} reason={reason}")


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
