"""Session orchestration for continual adapter training."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from src.training.types import TrainingHistory, TrainingProgressState, ValidationReport
from src.training.validation import evaluate_model


Observer = Callable[[Dict[str, Any]], None]
StopPolicy = Callable[[], bool]

_MIN_MODE_METRICS = {"val_loss", "generalization_gap", "epoch_loss", "train_loss"}


class ContinualTrainingSession:
    """Own epoch/batch orchestration around a trainer batch API."""

    def __init__(
        self,
        trainer: Any,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        num_epochs: int,
        *,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        observers: Optional[List[Observer]] = None,
        stop_policy: Optional[StopPolicy] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        run_id: str = "",
        checkpoint_every_n_steps: int = 0,
        checkpoint_on_exception: bool = False,
    ) -> None:
        self.trainer = trainer
        self.train_loader = train_loader
        self.num_epochs = int(max(1, num_epochs))
        self.val_loader = val_loader
        self.observers = list(observers or [])
        self.stop_policy = stop_policy
        self.run_id = str(run_id)
        self.checkpoint_every_n_steps = int(max(0, checkpoint_every_n_steps))
        self.checkpoint_on_exception = bool(checkpoint_on_exception)

        resume_payload = dict(resume_state or {})
        history_payload = resume_payload.get("history", resume_payload.get("history_snapshot", {}))
        progress_payload = resume_payload.get("progress_state", {})

        self.history = TrainingHistory.from_dict(history_payload if isinstance(history_payload, dict) else {})
        self.progress_state = TrainingProgressState.from_dict(progress_payload if isinstance(progress_payload, dict) else {})
        self.best_metric_state = dict(resume_payload.get("best_metric_state", {}))
        self.history.resume_start_epoch = int(self.progress_state.resume_start_epoch)
        self._elapsed_before_resume = float(self.progress_state.elapsed_sec)

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "event_type": str(event_type),
            "run_id": self.run_id,
            "payload": dict(payload),
        }
        for observer in self.observers:
            observer(event)

    @staticmethod
    def _batch_advisory(loss: float, grad_norm: float, loss_window: List[float]) -> Dict[str, str]:
        advisory = ""
        severity = "info"
        if not torch.isfinite(torch.tensor(loss)).item():
            advisory = "Loss became non-finite (NaN/Inf). Consider stopping this run."
            severity = "critical"
        elif grad_norm > 1000.0:
            advisory = f"Gradient norm is very high ({grad_norm:.2f}). Training may be unstable."
            severity = "warning"
        elif len(loss_window) >= 5:
            baseline = sum(loss_window[:-1]) / max(1, len(loss_window) - 1)
            if baseline > 0 and loss > (baseline * 1.6):
                advisory = "Batch loss spiked above recent trend. Monitor closely."
                severity = "warning"
        return {"advisory": advisory, "severity": severity}

    @staticmethod
    def _decorate_validation(report: ValidationReport, epoch_loss: float) -> ValidationReport:
        generalization_gap = float(report.val_loss - epoch_loss)
        advisory = ""
        severity = "info"
        if generalization_gap > 0.5:
            advisory = (
                f"Validation loss is notably above training loss (gap={generalization_gap:.3f}). "
                "Potential overfitting."
            )
            severity = "warning"
        return ValidationReport(
            val_loss=report.val_loss,
            val_accuracy=report.val_accuracy,
            macro_f1=report.macro_f1,
            weighted_f1=report.weighted_f1,
            balanced_accuracy=report.balanced_accuracy,
            per_class_accuracy=dict(report.per_class_accuracy),
            per_class_support=dict(report.per_class_support),
            worst_classes=list(report.worst_classes),
            generalization_gap=generalization_gap,
            epoch_advisory=advisory,
            epoch_severity=severity,
        )

    @staticmethod
    def _metric_mode(metric_name: str) -> str:
        return "min" if str(metric_name) in _MIN_MODE_METRICS else "max"

    @staticmethod
    def _metric_value(metric_name: str, *, epoch_loss: float, report: Optional[ValidationReport]) -> Optional[float]:
        if metric_name in {"epoch_loss", "train_loss"}:
            return float(epoch_loss)
        if report is None:
            return None
        if hasattr(report, metric_name):
            value = getattr(report, metric_name)
            if value is None:
                return None
            return float(value)
        return None

    @staticmethod
    def _metric_improved(
        *,
        current: float,
        best: Optional[float],
        mode: str,
        min_delta: float = 0.0,
    ) -> bool:
        if best is None:
            return True
        if mode == "min":
            return float(current) < float(best) - float(min_delta)
        return float(current) > float(best) + float(min_delta)

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "v6_training_session",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "progress_state": self.progress_state.to_dict(),
            "history": self.history.to_dict(),
            "best_metric_state": dict(self.best_metric_state),
        }

    def _emit_checkpoint_request(self, *, reason: str, payload: Dict[str, Any]) -> None:
        checkpoint_payload = {
            "reason": str(reason),
            "mark_best": bool(payload.get("mark_best", False)),
            "val_loss": payload.get("val_loss"),
            "epoch": int(payload.get("epoch_done", payload.get("epoch", 0))),
            "batch": int(payload.get("batch", 0)),
            "global_step": int(payload.get("global_step", self.progress_state.global_step)),
            "elapsed_sec": float(payload.get("elapsed_sec", self.progress_state.elapsed_sec)),
            "history_snapshot": self.history.to_dict(),
            "best_metric_state": dict(self.best_metric_state),
        }
        self._emit("checkpoint_requested", checkpoint_payload)

    def _maybe_update_best_metric(
        self,
        *,
        epoch_idx: int,
        epoch_loss: float,
        validation_report: Optional[ValidationReport],
    ) -> Dict[str, Any]:
        metric_name = str(getattr(self.trainer.config, "evaluation_best_metric", "val_loss"))
        metric_mode = self._metric_mode(metric_name)
        metric_value = self._metric_value(metric_name, epoch_loss=epoch_loss, report=validation_report)
        previous_best = self.best_metric_state.get("best_metric_value")
        improved = False
        if metric_value is not None:
            improved = self._metric_improved(
                current=float(metric_value),
                best=None if previous_best is None else float(previous_best),
                mode=metric_mode,
                min_delta=0.0,
            )
        if improved and metric_value is not None:
            self.best_metric_state = {
                "best_metric_name": metric_name,
                "best_metric_value": float(metric_value),
                "best_epoch": int(epoch_idx + 1),
                "mode": metric_mode,
                "patience_bad_epochs": 0,
            }
            self.history.best_metric_name = metric_name
            self.history.best_metric_value = float(metric_value)
            self.history.best_epoch = int(epoch_idx + 1)
            if hasattr(self.trainer, "best_metric_state"):
                self.trainer.best_metric_state = dict(self.best_metric_state)
            self._emit(
                "best_metric_updated",
                {
                    "epoch_done": int(epoch_idx + 1),
                    "best_metric_name": metric_name,
                    "best_metric_value": float(metric_value),
                    "mode": metric_mode,
                },
            )
        return {
            "best_metric_name": metric_name,
            "best_metric_value": metric_value,
            "best_metric_mode": metric_mode,
            "mark_best": bool(improved),
        }

    def _should_early_stop(
        self,
        *,
        epoch_idx: int,
        epoch_loss: float,
        validation_report: Optional[ValidationReport],
    ) -> bool:
        if not bool(getattr(self.trainer.config, "early_stopping_enabled", False)):
            return False
        metric_name = str(getattr(self.trainer.config, "early_stopping_metric", "val_loss"))
        metric_mode = str(getattr(self.trainer.config, "early_stopping_mode", self._metric_mode(metric_name)))
        patience = int(getattr(self.trainer.config, "early_stopping_patience", 3))
        min_delta = float(getattr(self.trainer.config, "early_stopping_min_delta", 0.0))

        metric_value = self._metric_value(metric_name, epoch_loss=epoch_loss, report=validation_report)
        if metric_value is None:
            return False

        current_bad_epochs = int(self.best_metric_state.get("patience_bad_epochs", 0))
        best_for_metric = (
            float(self.best_metric_state["best_metric_value"])
            if self.best_metric_state.get("best_metric_name") == metric_name and self.best_metric_state.get("best_metric_value") is not None
            else None
        )
        if self._metric_improved(current=float(metric_value), best=best_for_metric, mode=metric_mode, min_delta=min_delta):
            current_bad_epochs = 0
        else:
            current_bad_epochs += 1

        self.best_metric_state["patience_bad_epochs"] = int(current_bad_epochs)
        self.best_metric_state["early_stopping_metric"] = metric_name
        self.best_metric_state["early_stopping_mode"] = metric_mode
        if hasattr(self.trainer, "best_metric_state"):
            self.trainer.best_metric_state = dict(self.best_metric_state)

        return current_bad_epochs > patience

    def run(self) -> TrainingHistory:
        global_step = int(self.progress_state.global_step)
        start_epoch = int(self.progress_state.resume_start_epoch or self.progress_state.epoch)
        self.history.resume_start_epoch = start_epoch
        batch_loss_window: List[float] = []
        train_started_at = time.perf_counter() - max(0.0, self._elapsed_before_resume)
        total_batches = len(self.train_loader) if hasattr(self.train_loader, "__len__") else 0

        if hasattr(self.trainer, "configure_training_plan"):
            self.trainer.configure_training_plan(total_batches=max(1, total_batches), num_epochs=self.num_epochs)

        if global_step > 0 or start_epoch > 0:
            self._emit(
                "resume_loaded",
                {
                    "resume_epoch": int(start_epoch),
                    "global_step": int(global_step),
                    "optimizer_steps": int(self.progress_state.optimizer_steps),
                },
            )

        stopped_early = False
        try:
            for epoch_idx in range(start_epoch, self.num_epochs):
                self.trainer.set_train_mode()
                losses: List[float] = []
                for batch_idx, batch in enumerate(self.train_loader):
                    stats = self.trainer.train_batch(batch)
                    losses.append(float(stats.loss))
                    batch_loss_window.append(float(stats.loss))
                    if len(batch_loss_window) > 8:
                        batch_loss_window = batch_loss_window[-8:]

                    global_step += 1
                    elapsed = float(time.perf_counter() - train_started_at)
                    total_batches_planned = max(1, self.num_epochs * max(1, total_batches))
                    processed_batches = max(1, (epoch_idx * max(1, total_batches)) + (batch_idx + 1))
                    eta_sec = float((elapsed / processed_batches) * max(0, total_batches_planned - processed_batches))
                    advisory = self._batch_advisory(float(stats.loss), float(stats.grad_norm), batch_loss_window)

                    self.progress_state = TrainingProgressState(
                        epoch=epoch_idx + 1,
                        batch=batch_idx + 1,
                        total_batches=int(total_batches),
                        global_step=int(global_step),
                        optimizer_steps=int(stats.optimizer_steps),
                        elapsed_sec=elapsed,
                        eta_sec=eta_sec,
                        resume_start_epoch=int(start_epoch),
                        stopped_early=False,
                    )
                    self.history.global_step = int(global_step)
                    self.history.optimizer_steps = int(stats.optimizer_steps)
                    payload = {
                        **self.progress_state.to_dict(),
                        **stats.to_dict(),
                        **advisory,
                        "epoch_progress": float((batch_idx + 1) / max(1, total_batches)),
                    }
                    self._emit("batch_end", payload)

                    if self.checkpoint_every_n_steps > 0 and (global_step % self.checkpoint_every_n_steps == 0):
                        self._emit_checkpoint_request(reason="batch_interval", payload=payload)

                    if self.stop_policy is not None and bool(self.stop_policy()):
                        stopped_early = True
                        self.progress_state.stopped_early = True
                        self._emit("stop_requested", self.progress_state.to_dict())
                        break

                if not losses:
                    break

                epoch_loss = float(sum(losses) / max(1, len(losses)))
                self.history.train_loss.append(epoch_loss)
                self.trainer.current_epoch = int(epoch_idx + 1)
                validation_report: Optional[ValidationReport] = None
                epoch_payload: Dict[str, Any] = {
                    "epoch_done": int(epoch_idx + 1),
                    "epoch_loss": epoch_loss,
                    "global_step": int(global_step),
                    "optimizer_steps": int(getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps)),
                    "stopped_early": bool(stopped_early),
                    "elapsed_sec": float(time.perf_counter() - train_started_at),
                }

                if self.val_loader is not None:
                    validation_report = evaluate_model(self.trainer, self.val_loader)
                    if validation_report is not None:
                        decorated = self._decorate_validation(validation_report, epoch_loss)
                        validation_report = decorated
                        self.history.append_validation(decorated)
                        epoch_payload.update(decorated.to_dict())
                        self._emit("validation_end", {"epoch_done": int(epoch_idx + 1), **decorated.to_dict()})

                best_metric_payload = self._maybe_update_best_metric(
                    epoch_idx=epoch_idx,
                    epoch_loss=epoch_loss,
                    validation_report=validation_report,
                )
                epoch_payload.update(best_metric_payload)

                if hasattr(self.trainer, "config") and getattr(self.trainer.config, "scheduler_step_on", "batch") == "epoch":
                    step_scheduler = getattr(self.trainer, "_step_scheduler", None)
                    if callable(step_scheduler):
                        step_scheduler()

                epoch_payload["history_snapshot"] = self.history.to_dict()
                self.trainer.set_train_mode()
                self._emit("epoch_end", epoch_payload)
                self._emit_checkpoint_request(reason="epoch_end", payload=epoch_payload)

                if self._should_early_stop(epoch_idx=epoch_idx, epoch_loss=epoch_loss, validation_report=validation_report):
                    stopped_early = True
                    self.progress_state.stopped_early = True
                    self._emit(
                        "stop_requested",
                        {
                            **self.progress_state.to_dict(),
                            "reason": "early_stopping",
                            "best_metric_state": dict(self.best_metric_state),
                        },
                    )

                if stopped_early:
                    break
        except Exception as exc:
            aborted_payload = {
                **self.progress_state.to_dict(),
                "error": str(exc),
                "history_snapshot": self.history.to_dict(),
            }
            self._emit("training_aborted", aborted_payload)
            if self.checkpoint_on_exception:
                self._emit_checkpoint_request(reason="exception", payload=aborted_payload)
            raise

        self.history.stopped_early = bool(stopped_early)
        self.history.global_step = int(global_step)
        self.history.optimizer_steps = int(getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps))
        self.progress_state.global_step = int(global_step)
        self.progress_state.optimizer_steps = int(getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps))
        self.progress_state.resume_start_epoch = int(start_epoch)
        self.progress_state.stopped_early = bool(stopped_early)
        self._emit(
            "training_completed",
            {
                **self.progress_state.to_dict(),
                "history_snapshot": self.history.to_dict(),
                "best_metric_state": dict(self.best_metric_state),
            },
        )
        return self.history
