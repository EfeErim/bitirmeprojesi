"""Session orchestration for continual adapter training."""

from __future__ import annotations

import time
from collections.abc import Sized
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from src.training.services.session_flow import (
    batch_advisory,
    build_checkpoint_request_payload,
    decorate_validation,
    should_early_stop,
    update_best_metric_state,
)
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

        self.history = TrainingHistory.from_dict(
            history_payload if isinstance(history_payload, dict) else {}
        )
        self.progress_state = TrainingProgressState.from_dict(
            progress_payload if isinstance(progress_payload, dict) else {}
        )
        self.best_metric_state = dict(resume_payload.get("best_metric_state", {}))
        self.history.resume_start_epoch = int(self.progress_state.resume_start_epoch)
        self._elapsed_before_resume = float(self.progress_state.elapsed_sec)
        preferred_loader = self.val_loader if self.val_loader is not None else self.train_loader
        set_loader = getattr(self.trainer, "set_preferred_ood_calibration_loader", None)
        if callable(set_loader):
            set_loader(preferred_loader)

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "event_type": str(event_type),
            "run_id": self.run_id,
            "payload": dict(payload),
        }
        for observer in self.observers:
            observer(event)

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
        checkpoint_payload = build_checkpoint_request_payload(
            reason=reason,
            payload=payload,
            global_step=self.progress_state.global_step,
            elapsed_sec=self.progress_state.elapsed_sec,
            history=self.history,
            best_metric_state=self.best_metric_state,
        )
        self._emit("checkpoint_requested", checkpoint_payload)

    def run(self) -> TrainingHistory:
        global_step = int(self.progress_state.global_step)
        start_epoch = int(self.progress_state.resume_start_epoch or self.progress_state.epoch)
        self.history.resume_start_epoch = start_epoch
        batch_loss_window: List[float] = []
        train_started_at = time.perf_counter() - max(0.0, self._elapsed_before_resume)
        total_batches = len(self.train_loader) if isinstance(self.train_loader, Sized) else 0

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
                    advisory = batch_advisory(float(stats.loss), float(stats.grad_norm), batch_loss_window)

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
                    "optimizer_steps": int(
                        getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps)
                    ),
                    "stopped_early": bool(stopped_early),
                    "elapsed_sec": float(time.perf_counter() - train_started_at),
                }

                if self.val_loader is not None:
                    validation_report = evaluate_model(self.trainer, self.val_loader)
                    if validation_report is not None:
                        decorated = decorate_validation(validation_report, epoch_loss)
                        validation_report = decorated
                        self.history.append_validation(decorated)
                        epoch_payload.update(decorated.to_dict())
                        self._emit("validation_end", {"epoch_done": int(epoch_idx + 1), **decorated.to_dict()})

                self.best_metric_state, best_metric_payload = update_best_metric_state(
                    trainer=self.trainer,
                    history=self.history,
                    best_metric_state=self.best_metric_state,
                    epoch_idx=epoch_idx,
                    epoch_loss=epoch_loss,
                    validation_report=validation_report,
                    emit_fn=self._emit,
                )
                epoch_payload.update(best_metric_payload)

                if (
                    hasattr(self.trainer, "config")
                    and getattr(self.trainer.config, "scheduler_step_on", "batch") == "epoch"
                ):
                    step_scheduler = getattr(self.trainer, "_step_scheduler", None)
                    if callable(step_scheduler):
                        step_scheduler()

                epoch_payload["history_snapshot"] = self.history.to_dict()
                self.trainer.set_train_mode()
                self._emit("epoch_end", epoch_payload)
                self._emit_checkpoint_request(reason="epoch_end", payload=epoch_payload)

                self.best_metric_state, should_stop = should_early_stop(
                    trainer=self.trainer,
                    best_metric_state=self.best_metric_state,
                    epoch_loss=epoch_loss,
                    validation_report=validation_report,
                )
                if should_stop:
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
        self.history.optimizer_steps = int(
            getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps)
        )
        self.progress_state.global_step = int(global_step)
        self.progress_state.optimizer_steps = int(
            getattr(self.trainer, "optimizer_steps", self.progress_state.optimizer_steps)
        )
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
