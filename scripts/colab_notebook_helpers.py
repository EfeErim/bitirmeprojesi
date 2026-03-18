#!/usr/bin/env python3
"""Notebook 2 helper functions only."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib

from src.training.services.reporting import (
    persist_production_readiness_artifact as persist_production_readiness_artifact_core,
)
from src.training.services.reporting import (
    persist_training_history_artifacts as persist_training_history_artifacts_core,
)
from src.training.services.reporting import (
    persist_validation_artifacts as persist_validation_artifacts_core,
)

matplotlib.use("Agg")

_EXPECTED_REPO_EXPORTS = ("outputs", "telemetry", "checkpoint_state")


def _artifact_dir(root: Path, *parts: str) -> Path:
    target = root / "outputs" / "colab_notebook_training" / "artifacts"
    for part in parts:
        target /= part
    target.mkdir(parents=True, exist_ok=True)
    return target


def notebook_artifact_root(root: Path) -> Path:
    return _artifact_dir(root)


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

    resolved_run_id = str(run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
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


def _path_exists(path_like: Optional[str | Path]) -> bool:
    return bool(path_like and Path(path_like).expanduser().exists())


def _call_if_present(target: Any, method_name: str, *args, **kwargs) -> None:
    method = getattr(target, method_name, None)
    if not callable(method):
        return
    try:
        method(*args, **kwargs)
    except Exception:
        pass


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

    @staticmethod
    def _append_advisory(
        parts: List[str],
        payload: Dict[str, Any],
        *,
        message_key: str,
        severity_key: str,
    ) -> None:
        advisory = str(payload.get(message_key, "")).strip()
        severity = str(payload.get(severity_key, "")).strip().lower()
        if advisory and severity in {"warning", "critical"}:
            parts.append(f"{severity}={advisory}")

    def _metric_fragment(self, payload: Dict[str, Any], key: str, label: str) -> Optional[str]:
        value = payload.get(key)
        if value is None:
            return None
        return f"{label}={float(value):.4f}"

    def handle(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event_name = str(event_type or "")
        event = dict(payload or {})
        handler = {
            "batch_end": self._handle_batch_end,
            "validation_end": self._handle_validation_end,
            "best_metric_updated": self._handle_best_metric,
            "stop_requested": self._handle_stop_requested,
        }.get(event_name)
        if handler is not None:
            handler(event)

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
        self._append_advisory(parts, payload, message_key="advisory", severity_key="severity")
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
        self._append_advisory(parts, payload, message_key="epoch_advisory", severity_key="epoch_severity")
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
    import matplotlib.pyplot as plt

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
    artifact_subdir: str = "validation",
    telemetry_subdir: Optional[str] = None,
    gate_targets: Optional[Dict[str, float]] = None,
    require_ood: bool = False,
    ood_labels: Optional[List[int]] = None,
    ood_scores: Optional[List[float]] = None,
    sure_ds_f1: Optional[float] = None,
    conformal_empirical_coverage: Optional[float] = None,
    conformal_avg_set_size: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return persist_validation_artifacts_core(
        artifact_root=_artifact_dir(root),
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        telemetry=telemetry,
        artifact_subdir=artifact_subdir,
        telemetry_subdir=telemetry_subdir,
        gate_targets=gate_targets,
        require_ood=require_ood,
        ood_labels=ood_labels,
        ood_scores=ood_scores,
        sure_ds_f1=sure_ds_f1,
        conformal_empirical_coverage=conformal_empirical_coverage,
        conformal_avg_set_size=conformal_avg_set_size,
        context=context,
    )


def persist_production_readiness_artifact(
    *,
    root: Path,
    classification_metric_gate: Dict[str, Any] | None,
    classification_split: str,
    ood_evidence_source: str | None,
    ood_metrics: Dict[str, Any] | None,
    targets: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
) -> Dict[str, Any]:
    return persist_production_readiness_artifact_core(
        artifact_root=_artifact_dir(root),
        classification_metric_gate=classification_metric_gate,
        classification_split=classification_split,
        ood_evidence_source=ood_evidence_source,
        ood_metrics=ood_metrics,
        targets=targets,
        context=context,
        telemetry=telemetry,
    )


def _resolve_colab_runtime_api() -> Any:
    try:
        from google.colab import runtime
    except Exception:
        return None
    return runtime


def build_notebook_completion_report(
    *,
    state: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
    repo_run_exports: Optional[Dict[str, str]] = None,
    notebook_export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    resolved_state = dict(state or {})
    resolved_exports = dict(repo_run_exports or {})

    evaluation_artifacts = resolved_state.get("evaluation_artifacts")
    evaluation_splits = sorted(evaluation_artifacts.keys()) if isinstance(evaluation_artifacts, dict) else []

    summary_path = getattr(telemetry, "local_summary_path", None)
    repo_export_checks = {
        name: _path_exists(resolved_exports.get(name))
        for name in _EXPECTED_REPO_EXPORTS
    }
    repo_exports_complete = bool(resolved_exports) and all(
        repo_export_checks.get(name, False) for name in _EXPECTED_REPO_EXPORTS
    )

    checks = {
        "evaluation_artifacts": bool(evaluation_splits),
        "production_readiness": isinstance(resolved_state.get("production_readiness"), dict)
        and bool(resolved_state.get("production_readiness")),
        "telemetry_summary": _path_exists(summary_path),
        "repo_exports": repo_exports_complete,
        "executed_notebook_export": _path_exists(notebook_export_path),
    }
    blocking_check_names = (
        "evaluation_artifacts",
        "production_readiness",
        "telemetry_summary",
        "repo_exports",
    )
    soft_check_names = ("executed_notebook_export",)
    missing = [name for name in blocking_check_names if not checks.get(name, False)]
    soft_missing = [name for name in soft_check_names if not checks.get(name, False)]
    readiness = resolved_state.get("production_readiness") or {}
    return {
        "ready": not missing,
        "checks": checks,
        "missing": missing,
        "soft_missing": soft_missing,
        "blocking_checks": {name: checks.get(name, False) for name in blocking_check_names},
        "soft_checks": {name: checks.get(name, False) for name in soft_check_names},
        "evaluation_splits": evaluation_splits,
        "repo_exports": repo_export_checks,
        "production_readiness_status": str(readiness.get("status", "")),
        "ood_evidence_source": readiness.get("ood_evidence_source"),
    }


def maybe_auto_disconnect_colab_runtime(
    *,
    enabled: bool,
    grace_period_sec: float = 20.0,
    state: Optional[Dict[str, Any]] = None,
    telemetry: Any = None,
    repo_run_exports: Optional[Dict[str, str]] = None,
    notebook_export_path: Optional[str | Path] = None,
    completion_report: Optional[Dict[str, Any]] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    emit = print if print_fn is None else print_fn
    report = (
        completion_report
        if completion_report is not None
        else build_notebook_completion_report(
            state=state,
            telemetry=telemetry,
            repo_run_exports=repo_run_exports,
            notebook_export_path=notebook_export_path,
        )
    )
    report["auto_disconnect_enabled"] = bool(enabled)
    report.setdefault("disconnect_requested", False)
    report.setdefault("missing", [])
    report.setdefault("soft_missing", [])

    def _publish_status(phase: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "phase": str(phase),
            "auto_disconnect": bool(enabled),
            "disconnect_requested": bool(report.get("disconnect_requested", False)),
            "completion_checks": dict(report.get("checks", {})),
            "completion_missing": list(report.get("missing", [])),
            "completion_soft_missing": list(report.get("soft_missing", [])),
        }
        payload.update(extra)
        _call_if_present(telemetry, "update_latest", payload)
        _call_if_present(telemetry, "sync_pending")

    if not enabled:
        emit("[COLAB] Auto-disconnect disabled.")
        _publish_status("auto_disconnect_disabled")
        return report

    if not bool(report.get("ready")):
        missing = ", ".join(str(item) for item in report.get("missing", [])) or "unknown"
        emit(f"[COLAB] Auto-disconnect skipped. Incomplete required checks: {missing}")
        soft_missing = ", ".join(str(item) for item in report.get("soft_missing", []))
        if soft_missing:
            emit(f"[COLAB] Soft-missing checks: {soft_missing}")
        _publish_status("auto_disconnect_skipped")
        return report

    runtime_api = _resolve_colab_runtime_api()
    if runtime_api is None or not hasattr(runtime_api, "unassign"):
        emit("[COLAB] Auto-disconnect skipped. google.colab.runtime.unassign is unavailable.")
        _publish_status("auto_disconnect_unavailable")
        return report

    soft_missing = ", ".join(str(item) for item in report.get("soft_missing", []))
    if soft_missing:
        emit(f"[COLAB] Proceeding despite soft-missing checks: {soft_missing}")

    delay = max(0.0, float(grace_period_sec or 0.0))
    report["disconnect_requested"] = True
    report["grace_period_sec"] = delay

    _publish_status(
        "auto_disconnect_pending",
        auto_disconnect=True,
        grace_period_sec=delay,
    )

    if delay > 0:
        emit(f"[COLAB] Work complete. Disconnecting runtime in {delay:.0f}s to avoid idle credit use.")
        time.sleep(delay)
    else:
        emit("[COLAB] Work complete. Disconnecting runtime now to avoid idle credit use.")

    try:
        runtime_api.unassign()
    except Exception as exc:
        report["disconnect_requested"] = False
        report["disconnect_error"] = f"{exc.__class__.__name__}: {exc}"
        emit(f"[COLAB] Auto-disconnect failed: {report['disconnect_error']}")
        _publish_status("auto_disconnect_failed", disconnect_error=report["disconnect_error"])
    return report
