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
    ) -> None:
        self.trainer = trainer
        self.train_loader = train_loader
        self.num_epochs = int(max(1, num_epochs))
        self.val_loader = val_loader
        self.observers = list(observers or [])
        self.stop_policy = stop_policy
        self.run_id = str(run_id)

        resume_payload = dict(resume_state or {})
        history_payload = resume_payload.get("history", resume_payload.get("history_snapshot", {}))
        progress_payload = resume_payload.get("progress_state", {})

        self.history = TrainingHistory.from_dict(history_payload if isinstance(history_payload, dict) else {})
        self.progress_state = TrainingProgressState.from_dict(progress_payload if isinstance(progress_payload, dict) else {})
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

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "v6_training_session",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "progress_state": self.progress_state.to_dict(),
            "history": self.history.to_dict(),
        }

    def run(self) -> TrainingHistory:
        global_step = int(self.progress_state.global_step)
        start_epoch = int(self.progress_state.resume_start_epoch or self.progress_state.epoch)
        self.history.resume_start_epoch = start_epoch
        batch_loss_window: List[float] = []
        train_started_at = time.perf_counter() - max(0.0, self._elapsed_before_resume)
        total_batches = len(self.train_loader) if hasattr(self.train_loader, "__len__") else 0

        if global_step > 0 or start_epoch > 0:
            self._emit(
                "resume_loaded",
                {
                    "resume_epoch": int(start_epoch),
                    "global_step": int(global_step),
                },
            )

        stopped_early = False
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
                    elapsed_sec=elapsed,
                    eta_sec=eta_sec,
                    resume_start_epoch=int(start_epoch),
                    stopped_early=False,
                )
                payload = {
                    **self.progress_state.to_dict(),
                    **stats.to_dict(),
                    **advisory,
                    "epoch_progress": float((batch_idx + 1) / max(1, total_batches)),
                }
                self._emit("batch_end", payload)

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
            epoch_payload: Dict[str, Any] = {
                "epoch_done": int(epoch_idx + 1),
                "epoch_loss": epoch_loss,
                "global_step": int(global_step),
                "stopped_early": bool(stopped_early),
            }

            if self.val_loader is not None:
                validation_report = evaluate_model(self.trainer, self.val_loader)
                if validation_report is not None:
                    decorated = self._decorate_validation(validation_report, epoch_loss)
                    self.history.append_validation(decorated)
                    epoch_payload.update(decorated.to_dict())

            self.trainer.set_train_mode()
            self._emit("epoch_end", epoch_payload)

            if stopped_early:
                break

        self.history.stopped_early = bool(stopped_early)
        self.history.global_step = int(global_step)
        self.progress_state.global_step = int(global_step)
        self.progress_state.resume_start_epoch = int(start_epoch)
        self.progress_state.stopped_early = bool(stopped_early)
        return self.history
