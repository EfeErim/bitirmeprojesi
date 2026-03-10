"""Canonical training workflow for the supported adapter-training path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.loaders import create_training_loaders
from src.training.services.ood_benchmark import run_leave_one_class_out_benchmark
from src.training.services.reporting import (
    BatchMetricsRecorder,
    load_batch_metrics_history,
    persist_batch_metrics_artifacts,
    persist_production_readiness_artifact,
    persist_training_history_artifacts,
    persist_training_results_figure,
    persist_training_summary_artifact,
    persist_validation_artifacts,
)
from src.training.validation import evaluate_model_with_artifact_metrics
from src.workflows.training_support import (
    build_artifact_payload,
    prepare_training_run,
    select_calibration_source,
    stringify_paths,
)
from src.workflows.training_support import (
    loader_size as workflow_loader_size,
)

Observer = Callable[[Dict[str, Any]], None]


def _loader_size(loader: Any) -> int:
    return workflow_loader_size(loader)


def _stringify_paths(value: Any) -> Any:
    return stringify_paths(value)


def _last_history_value(history_payload: Dict[str, Any], key: str) -> Optional[float]:
    values = history_payload.get(key, [])
    if not isinstance(values, list) or not values:
        return None
    return float(values[-1])


_FINAL_HISTORY_METRICS = (
    "train_loss",
    "val_loss",
    "val_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "generalization_gap",
)


def _collect_final_metrics(history_payload: Dict[str, Any]) -> Dict[str, float]:
    final_metrics = {
        key: _last_history_value(history_payload, key)
        for key in _FINAL_HISTORY_METRICS
    }
    return {
        key: value
        for key, value in final_metrics.items()
        if value is not None
    }


@dataclass
class TrainingWorkflowResult:
    run_id: str
    crop_name: str
    class_names: List[str]
    history: Dict[str, Any]
    loader_sizes: Dict[str, int]
    adapter_dir: Path
    artifact_dir: Optional[Path] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    ood_calibration: Dict[str, Any] = field(default_factory=dict)
    checkpoint_records: List[Dict[str, Any]] = field(default_factory=list)
    ood_evidence_source: str = ""
    ood_benchmark: Dict[str, Any] = field(default_factory=dict)
    production_readiness: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "crop_name": self.crop_name,
            "class_names": list(self.class_names),
            "history": dict(self.history),
            "loader_sizes": {str(k): int(v) for k, v in self.loader_sizes.items()},
            "adapter_dir": str(self.adapter_dir),
            "artifact_dir": ("" if self.artifact_dir is None else str(self.artifact_dir)),
            "artifacts": _stringify_paths(self.artifacts),
            "ood_calibration": dict(self.ood_calibration),
            "checkpoint_records": [dict(item) for item in self.checkpoint_records],
            "ood_evidence_source": str(self.ood_evidence_source),
            "ood_benchmark": _stringify_paths(self.ood_benchmark),
            "production_readiness": dict(self.production_readiness),
        }


class TrainingWorkflow:
    """Stable app-facing entrypoint for continual adapter training."""

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = "colab",
        device: str = "cuda",
    ) -> None:
        self.config = dict(config or get_config(environment=environment))
        self.device = str(device)

    @staticmethod
    def _emit_telemetry(
        telemetry: Any,
        event_type: str,
        payload: Dict[str, Any],
        *,
        phase: str,
        force_sync: Optional[bool] = None,
    ) -> None:
        if telemetry is None:
            return
        if force_sync is None:
            telemetry.emit_event(event_type, payload, phase=phase)
            return
        telemetry.emit_event(event_type, payload, phase=phase, force_sync=force_sync)

    def _build_workflow_observer(
        self,
        *,
        adapter: IndependentCropAdapter,
        run_id: str,
        telemetry: Any,
        checkpoint_manager: Any,
        batch_recorder: Optional[BatchMetricsRecorder],
        checkpoint_records: List[Dict[str, Any]],
        session_holder: Dict[str, Any],
    ) -> Observer:
        def _workflow_observer(event: Dict[str, Any]) -> None:
            event_type = str(event.get("event_type", "training_event"))
            payload = dict(event.get("payload", {}))
            if event_type == "batch_end" and batch_recorder is not None:
                batch_recorder.append(payload)

            self._emit_telemetry(
                telemetry,
                event_type,
                payload,
                phase="training",
                force_sync=False,
            )
            if telemetry is not None and hasattr(telemetry, "update_latest") and event_type in {
                "batch_end",
                "validation_end",
                "epoch_end",
                "training_completed",
            }:
                telemetry.update_latest({"event_type": event_type, **payload})

            if checkpoint_manager is None or event_type != "checkpoint_requested":
                return
            session = session_holder.get("session")
            if session is None:
                return
            record = checkpoint_manager.save_checkpoint(
                adapter=adapter,
                session=session,
                reason=str(payload.get("reason", "event")),
                run_id=run_id,
                mark_best=bool(payload.get("mark_best", False)),
                val_loss=payload.get("val_loss"),
            )
            checkpoint_records.append(dict(record))

        return _workflow_observer

    def _persist_training_artifacts(
        self,
        *,
        artifact_dir: Path,
        history_payload: Dict[str, Any],
        batch_metrics_csv: Path,
        telemetry: Any,
    ) -> Dict[str, Any]:
        batch_history = load_batch_metrics_history(batch_metrics_csv)
        return {
            **persist_training_history_artifacts(
                artifact_root=artifact_dir,
                history_snapshot=history_payload,
                telemetry=telemetry,
            ),
            **persist_batch_metrics_artifacts(
                artifact_root=artifact_dir,
                batch_metrics_csv=batch_metrics_csv,
                telemetry=telemetry,
            ),
            **persist_training_results_figure(
                artifact_root=artifact_dir,
                history_snapshot=history_payload,
                batch_history=batch_history,
                telemetry=telemetry,
            ),
        }

    def _persist_evaluation_artifacts(
        self,
        *,
        artifact_dir: Path,
        trainer: Any,
        loader: Any,
        ood_loader: Any,
        detected_classes: List[str],
        telemetry: Any,
        run_id: str,
        crop_name: str,
        loader_sizes: Dict[str, int],
        split_name: str,
        artifact_subdir: str,
        telemetry_subdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if trainer is None or _loader_size(loader) <= 0:
            return {}

        evaluation_result = evaluate_model_with_artifact_metrics(trainer, loader, ood_loader=ood_loader)
        if evaluation_result is None:
            return {}

        require_ood = bool(
            getattr(getattr(trainer, "config", None), "evaluation_require_ood_for_gate", False)
        )
        emit_metric_gate = bool(
            getattr(getattr(trainer, "config", None), "evaluation_emit_ood_gate", True)
        )
        metric_context = {
            "run_id": run_id,
            "crop_name": crop_name,
            "split_name": split_name,
            "num_classes": len(detected_classes),
            "loader_sizes": loader_sizes,
            **dict(evaluation_result.context),
        }
        return persist_validation_artifacts(
            artifact_root=artifact_dir,
            y_true=evaluation_result.y_true,
            y_pred=evaluation_result.y_pred,
            classes=detected_classes,
            telemetry=telemetry,
            artifact_subdir=artifact_subdir,
            telemetry_subdir=telemetry_subdir,
            require_ood=require_ood,
            emit_metric_gate=emit_metric_gate,
            ood_labels=evaluation_result.ood_labels,
            ood_scores=evaluation_result.ood_scores,
            sure_ds_f1=evaluation_result.sure_ds_f1,
            conformal_empirical_coverage=evaluation_result.conformal_empirical_coverage,
            conformal_avg_set_size=evaluation_result.conformal_avg_set_size,
            context=metric_context,
        )

    def _persist_split_artifacts(
        self,
        *,
        artifact_dir: Path,
        trainer: Any,
        loaders: Dict[str, Any],
        detected_classes: List[str],
        telemetry: Any,
        run_id: str,
        crop_name: str,
        loader_sizes: Dict[str, int],
    ) -> Dict[str, Dict[str, Any]]:
        split_specs = (
            ("val", loaders.get("val"), "validation"),
            ("test", loaders.get("test"), "test"),
        )
        return {
            split_name: self._persist_evaluation_artifacts(
                artifact_dir=artifact_dir,
                trainer=trainer,
                loader=loader,
                ood_loader=loaders.get("ood"),
                detected_classes=detected_classes,
                telemetry=telemetry,
                run_id=run_id,
                crop_name=crop_name,
                loader_sizes=loader_sizes,
                split_name=split_name,
                artifact_subdir=artifact_subdir,
            )
            for split_name, loader, artifact_subdir in split_specs
        }

    def _build_summary_payload(
        self,
        *,
        run_id: str,
        crop_name: str,
        detected_classes: List[str],
        loader_sizes: Dict[str, int],
        loader_batch_counts: Dict[str, int],
        split_class_counts: Dict[str, Dict[str, int]],
        adapter_dir: Path,
        artifact_dir: Path,
        checkpoint_records: List[Dict[str, Any]],
        ood_calibration: Dict[str, Any],
        history_payload: Dict[str, Any],
        calibration_split_name: str,
        ood_evidence_source: str,
        ood_benchmark: Dict[str, Any],
        production_readiness: Dict[str, Any],
        best_state_restored: bool,
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "crop_name": crop_name,
            "class_names": list(detected_classes),
            "class_count": len(detected_classes),
            "loader_sizes": loader_sizes,
            "loader_batch_counts": loader_batch_counts,
            "split_class_counts": split_class_counts,
            "adapter_dir": str(adapter_dir),
            "artifact_dir": str(artifact_dir),
            "checkpoint_count": len(checkpoint_records),
            "ood_calibration": dict(ood_calibration),
            "calibration_split_name": str(calibration_split_name),
            "stopped_early": bool(history_payload.get("stopped_early", False)),
            "global_step": int(history_payload.get("global_step", 0)),
            "optimizer_steps": int(history_payload.get("optimizer_steps", 0)),
            "best_metric_name": str(history_payload.get("best_metric_name", "")),
            "best_metric_value": history_payload.get("best_metric_value"),
            "best_epoch": int(history_payload.get("best_epoch", 0)),
            "best_state_restored": bool(best_state_restored),
            "ood_evidence_source": str(ood_evidence_source or ""),
            "ood_benchmark": dict(ood_benchmark),
            "production_readiness": dict(production_readiness),
            "final_metrics": _collect_final_metrics(history_payload),
        }

    @staticmethod
    def _select_authoritative_evaluation(
        validation_artifacts: Dict[str, Any],
        test_artifacts: Dict[str, Any],
        *,
        calibration_split_name: str,
    ) -> tuple[str, Dict[str, Any]]:
        if isinstance(test_artifacts, dict) and isinstance(test_artifacts.get("metric_gate"), dict):
            return "test", test_artifacts
        if calibration_split_name == "val":
            return "", {}
        if isinstance(validation_artifacts, dict) and isinstance(validation_artifacts.get("metric_gate"), dict):
            return "val", validation_artifacts
        return "", {}

    def _resolve_ood_evidence(
        self,
        *,
        crop_name: str,
        detected_classes: List[str],
        loaders: Dict[str, Any],
        evaluation_cfg: Dict[str, Any],
        authoritative_artifacts: Dict[str, Any],
        artifact_dir: Path,
        run_id: str,
        num_epochs: Optional[int],
        telemetry: Any,
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        real_ood_present = _loader_size(loaders.get("ood")) > 0
        if real_ood_present:
            ood_metrics = (
                dict(authoritative_artifacts.get("metric_gate", {}).get("metrics", {}))
                if authoritative_artifacts
                else {}
            )
            return "real_ood_split", ood_metrics, {}

        should_run_benchmark = (
            str(evaluation_cfg.get("ood_fallback_strategy", "held_out_benchmark")) == "held_out_benchmark"
            and bool(evaluation_cfg.get("ood_benchmark_auto_run", True))
        )
        if not should_run_benchmark:
            return "unavailable", {}, {}

        self._emit_telemetry(
            telemetry,
            "ood_benchmark_planned",
            {
                "run_id": run_id,
                "crop_name": crop_name,
                "estimated_fold_trainings": int(len(detected_classes)),
                "class_count": int(len(detected_classes)),
            },
            phase="evaluation",
        )

        ood_benchmark = run_leave_one_class_out_benchmark(
            crop_name=crop_name,
            class_names=detected_classes,
            loaders=loaders,
            config=self.config,
            device=self.device,
            artifact_root=artifact_dir,
            adapter_factory=IndependentCropAdapter,
            run_id=run_id,
            num_epochs=num_epochs,
            telemetry=telemetry,
            emit_event=lambda event_type, payload: self._emit_telemetry(
                telemetry,
                event_type,
                payload,
                phase="evaluation",
            ),
            min_classes=int(evaluation_cfg.get("ood_benchmark_min_classes", 3)),
        )
        return "held_out_benchmark", dict(ood_benchmark.get("metrics", {})), ood_benchmark

    def run(
        self,
        *,
        crop_name: str,
        data_dir: str | Path,
        output_dir: str | Path,
        class_names: Optional[Sequence[str]] = None,
        num_epochs: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        use_cache: bool = True,
        checkpoint_manager: Any = None,
        telemetry: Any = None,
        observers: Optional[Iterable[Observer]] = None,
        stop_policy: Optional[Callable[[], bool]] = None,
        checkpoint_every_n_steps: Optional[int] = None,
        checkpoint_on_exception: Optional[bool] = None,
        validation_every_n_epochs: Optional[int] = None,
        sampler: Optional[str] = None,
        error_policy: Optional[str] = None,
        run_id: str = "",
    ) -> TrainingWorkflowResult:
        crop_name = str(crop_name).strip().lower()
        if not crop_name:
            raise ValueError("crop_name must not be empty")

        run_setup = prepare_training_run(
            config=self.config,
            device=self.device,
            crop_name=crop_name,
            data_dir=data_dir,
            class_names=class_names,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_cache=use_cache,
            sampler=sampler,
            error_policy=error_policy,
            run_id=run_id,
            loader_factory=create_training_loaders,
            adapter_factory=IndependentCropAdapter,
        )
        training_cfg = run_setup.training_cfg
        colab_cfg = run_setup.colab_cfg
        run_id = run_setup.run_id
        loaders = run_setup.loaders
        loader_sizes = dict(run_setup.loader_sizes)
        loader_batch_counts = dict(run_setup.loader_batch_counts)
        detected_classes = list(run_setup.detected_classes)
        split_class_counts = {
            str(split_name): {str(class_name): int(count) for class_name, count in counts.items()}
            for split_name, counts in dict(run_setup.split_class_counts).items()
        }
        adapter = run_setup.adapter
        artifact_dir = Path(output_dir) / "training_metrics"
        batch_recorder = BatchMetricsRecorder(artifact_root=artifact_dir)

        checkpoint_records: List[Dict[str, Any]] = []
        session_observers: List[Observer] = list(observers or [])
        session_holder: Dict[str, Any] = {}
        session_observers.append(
            self._build_workflow_observer(
                adapter=adapter,
                run_id=run_id,
                telemetry=telemetry,
                checkpoint_manager=checkpoint_manager,
                batch_recorder=batch_recorder,
                checkpoint_records=checkpoint_records,
                session_holder=session_holder,
            )
        )

        session = adapter.build_training_session(
            train_loader=loaders["train"],
            num_epochs=num_epochs,
            val_loader=loaders.get("val"),
            observers=session_observers,
            stop_policy=stop_policy,
            run_id=run_id,
            checkpoint_every_n_steps=int(
                checkpoint_every_n_steps
                if checkpoint_every_n_steps is not None
                else colab_cfg.get("checkpoint_every_n_steps", 0)
            ),
            checkpoint_on_exception=bool(
                colab_cfg.get("checkpoint_on_exception", True)
                if checkpoint_on_exception is None
                else checkpoint_on_exception
            ),
            validation_every_n_epochs=int(
                validation_every_n_epochs
                if validation_every_n_epochs is not None
                else colab_cfg.get("validation_every_n_epochs", 1)
            ),
        )
        session_holder["session"] = session

        self._emit_telemetry(
            telemetry,
            "training_workflow_started",
            {
                "run_id": run_id,
                "crop_name": crop_name,
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
            },
            phase="training",
        )

        history = session.run()
        history_payload = history.to_dict()
        if int(history_payload.get("optimizer_steps", 0)) <= 0:
            raise RuntimeError(
                "Training produced zero optimizer steps. Check the continual split size, batch_size, and "
                "grad_accumulation_steps before launching a full experiment."
            )
        restore_best_state = getattr(session, "restore_best_model_state", None)
        best_state_restored = bool(restore_best_state()) if callable(restore_best_state) else False
        calibration_split_name, calibration_loader = select_calibration_source(loaders, loader_sizes)
        ood_calibration = {}
        if _loader_size(calibration_loader) > 0:
            ood_calibration = adapter.calibrate_ood(calibration_loader)
        adapter_dir = adapter.save_adapter(str(output_dir))

        training_artifacts = self._persist_training_artifacts(
            artifact_dir=artifact_dir,
            history_payload=history_payload,
            batch_metrics_csv=batch_recorder.output_path,
            telemetry=telemetry,
        )
        trainer_for_artifacts = getattr(session, "trainer", None)
        split_artifacts = self._persist_split_artifacts(
            artifact_dir=artifact_dir,
            trainer=trainer_for_artifacts,
            loaders=loaders,
            detected_classes=detected_classes,
            telemetry=telemetry,
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
        )
        validation_artifacts = split_artifacts["val"]
        test_artifacts = split_artifacts["test"]

        evaluation_cfg = dict(training_cfg.get("evaluation", {}))
        authoritative_split, authoritative_artifacts = self._select_authoritative_evaluation(
            validation_artifacts,
            test_artifacts,
            calibration_split_name=calibration_split_name,
        )
        ood_evidence_source, ood_evidence_metrics, ood_benchmark = self._resolve_ood_evidence(
            crop_name=crop_name,
            detected_classes=detected_classes,
            loaders=loaders,
            evaluation_cfg=evaluation_cfg,
            authoritative_artifacts=authoritative_artifacts,
            artifact_dir=artifact_dir,
            run_id=run_id,
            num_epochs=num_epochs,
            telemetry=telemetry,
        )

        readiness_artifacts = persist_production_readiness_artifact(
            artifact_root=artifact_dir,
            classification_metric_gate=(
                dict(authoritative_artifacts.get("metric_gate", {}))
                if isinstance(authoritative_artifacts, dict)
                else None
            ),
            classification_split=authoritative_split,
            ood_evidence_source=ood_evidence_source,
            ood_metrics=ood_evidence_metrics,
            context={
                "run_id": run_id,
                "crop_name": crop_name,
                "loader_sizes": loader_sizes,
                "loader_batch_counts": loader_batch_counts,
                "split_class_counts": split_class_counts,
                "calibration_split_name": calibration_split_name,
                "best_state_restored": bool(best_state_restored),
                "classification_split": authoritative_split,
                "ood_benchmark_status": ood_benchmark.get("status"),
                "ood_benchmark_passed": ood_benchmark.get("passed"),
            },
            require_ood=bool(evaluation_cfg.get("require_ood_for_gate", True)),
            telemetry=telemetry,
        )
        production_readiness = dict(readiness_artifacts.get("payload", {}))
        self._emit_telemetry(
            telemetry,
            "production_readiness_ready",
            {
                "run_id": run_id,
                "crop_name": crop_name,
                "status": production_readiness.get("status"),
                "passed": production_readiness.get("passed"),
                "ood_evidence_source": production_readiness.get("ood_evidence_source"),
            },
            phase="artifact",
        )

        summary_payload = self._build_summary_payload(
            run_id=run_id,
            crop_name=crop_name,
            detected_classes=detected_classes,
            loader_sizes=loader_sizes,
            loader_batch_counts=loader_batch_counts,
            split_class_counts=split_class_counts,
            adapter_dir=adapter_dir,
            artifact_dir=artifact_dir,
            checkpoint_records=checkpoint_records,
            ood_calibration=ood_calibration,
            history_payload=history_payload,
            calibration_split_name=calibration_split_name,
            ood_evidence_source=ood_evidence_source,
            ood_benchmark=ood_benchmark,
            production_readiness=production_readiness,
            best_state_restored=best_state_restored,
        )
        summary_artifacts = persist_training_summary_artifact(
            artifact_root=artifact_dir,
            summary_payload=summary_payload,
            telemetry=telemetry,
        )
        artifact_payload = build_artifact_payload(
            training_artifacts=training_artifacts,
            validation_artifacts=validation_artifacts,
            test_artifacts=test_artifacts,
            ood_benchmark=ood_benchmark,
            readiness_artifacts=readiness_artifacts,
            summary_artifacts=summary_artifacts,
        )

        result = TrainingWorkflowResult(
            run_id=run_id,
            crop_name=crop_name,
            class_names=detected_classes,
            history=history_payload,
            loader_sizes=loader_sizes,
            adapter_dir=adapter_dir,
            artifact_dir=artifact_dir,
            artifacts=artifact_payload,
            ood_calibration=ood_calibration,
            checkpoint_records=checkpoint_records,
            ood_evidence_source=ood_evidence_source,
            ood_benchmark=ood_benchmark,
            production_readiness=production_readiness,
        )

        self._emit_telemetry(
            telemetry,
            "training_artifacts_ready",
            {"artifact_dir": str(artifact_dir), "artifacts": artifact_payload},
            phase="artifact",
        )
        self._emit_telemetry(
            telemetry,
            "training_workflow_completed",
            result.to_dict(),
            phase="training",
        )

        return result
