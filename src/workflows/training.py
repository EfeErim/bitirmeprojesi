"""Canonical training workflow for the supported adapter-training path."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.data.loaders import create_training_loaders
from src.training.services.reporting import (
    persist_batch_metrics_artifacts,
    persist_training_history_artifacts,
    persist_training_results_figure,
    persist_training_summary_artifact,
    persist_validation_artifacts,
)
from src.training.validation import evaluate_model_with_predictions

Observer = Callable[[Dict[str, Any]], None]


def _loader_size(loader: Any) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except Exception:
            return 0
    try:
        return int(len(loader))
    except Exception:
        return 0


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value


def _last_history_value(history_payload: Dict[str, Any], key: str) -> Optional[float]:
    values = history_payload.get(key, [])
    if not isinstance(values, list) or not values:
        return None
    return float(values[-1])


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
        batch_history: List[Dict[str, Any]],
        checkpoint_records: List[Dict[str, Any]],
        session_holder: Dict[str, Any],
    ) -> Observer:
        def _workflow_observer(event: Dict[str, Any]) -> None:
            event_type = str(event.get("event_type", "training_event"))
            payload = dict(event.get("payload", {}))
            if event_type == "batch_end":
                batch_history.append(payload)

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
        batch_history: List[Dict[str, Any]],
        telemetry: Any,
    ) -> Dict[str, Any]:
        return {
            **persist_training_history_artifacts(
                artifact_root=artifact_dir,
                history_snapshot=history_payload,
                telemetry=telemetry,
            ),
            **persist_batch_metrics_artifacts(
                artifact_root=artifact_dir,
                batch_history=batch_history,
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

        validation_result = evaluate_model_with_predictions(trainer, loader)
        if validation_result is None:
            return {}

        _, y_true, y_pred = validation_result
        require_ood = bool(
            getattr(getattr(trainer, "config", None), "evaluation_require_ood_for_gate", False)
        )
        return persist_validation_artifacts(
            artifact_root=artifact_dir,
            y_true=y_true,
            y_pred=y_pred,
            classes=detected_classes,
            telemetry=telemetry,
            artifact_subdir=artifact_subdir,
            telemetry_subdir=telemetry_subdir,
            require_ood=require_ood,
            context={
                "run_id": run_id,
                "crop_name": crop_name,
                "split_name": split_name,
                "num_classes": len(detected_classes),
                "loader_sizes": loader_sizes,
            },
        )

    def _build_summary_payload(
        self,
        *,
        run_id: str,
        crop_name: str,
        detected_classes: List[str],
        loader_sizes: Dict[str, int],
        adapter_dir: Path,
        artifact_dir: Path,
        checkpoint_records: List[Dict[str, Any]],
        ood_calibration: Dict[str, Any],
        history_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "crop_name": crop_name,
            "class_names": list(detected_classes),
            "class_count": len(detected_classes),
            "loader_sizes": loader_sizes,
            "adapter_dir": str(adapter_dir),
            "artifact_dir": str(artifact_dir),
            "checkpoint_count": len(checkpoint_records),
            "ood_calibration": dict(ood_calibration),
            "stopped_early": bool(history_payload.get("stopped_early", False)),
            "global_step": int(history_payload.get("global_step", 0)),
            "optimizer_steps": int(history_payload.get("optimizer_steps", 0)),
            "best_metric_name": str(history_payload.get("best_metric_name", "")),
            "best_metric_value": history_payload.get("best_metric_value"),
            "best_epoch": int(history_payload.get("best_epoch", 0)),
            "final_metrics": {
                key: value
                for key, value in {
                    "train_loss": _last_history_value(history_payload, "train_loss"),
                    "val_loss": _last_history_value(history_payload, "val_loss"),
                    "val_accuracy": _last_history_value(history_payload, "val_accuracy"),
                    "macro_precision": _last_history_value(history_payload, "macro_precision"),
                    "macro_recall": _last_history_value(history_payload, "macro_recall"),
                    "macro_f1": _last_history_value(history_payload, "macro_f1"),
                    "weighted_f1": _last_history_value(history_payload, "weighted_f1"),
                    "balanced_accuracy": _last_history_value(history_payload, "balanced_accuracy"),
                    "generalization_gap": _last_history_value(history_payload, "generalization_gap"),
                }.items()
                if value is not None
            },
        }

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
        sampler: Optional[str] = None,
        error_policy: Optional[str] = None,
        run_id: str = "",
    ) -> TrainingWorkflowResult:
        crop_name = str(crop_name).strip().lower()
        if not crop_name:
            raise ValueError("crop_name must not be empty")

        training_cfg = dict(self.config.get("training", {}).get("continual", {}))
        data_cfg = dict(training_cfg.get("data", {}))
        colab_cfg = dict(self.config.get("colab", {}).get("training", {}))
        run_id = str(run_id or f"{crop_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

        loaders = create_training_loaders(
            data_dir=str(data_dir),
            crop=crop_name,
            batch_size=int(training_cfg.get("batch_size", 8)),
            num_workers=int(num_workers if num_workers is not None else colab_cfg.get("num_workers", 2)),
            use_cache=bool(use_cache),
            cache_size=int(data_cfg.get("cache_size", 1000)),
            target_size=int(data_cfg.get("target_size", 224)),
            error_policy=str(error_policy or data_cfg.get("loader_error_policy", "tolerant")),
            sampler=str(sampler or data_cfg.get("sampler", "shuffle")),
            seed=int(training_cfg.get("seed", 42)),
            validate_images_on_init=bool(data_cfg.get("validate_images_on_init", True)),
            pin_memory=bool(colab_cfg.get("pin_memory", True) if pin_memory is None else pin_memory),
        )

        detected_classes = list(class_names or getattr(loaders["train"].dataset, "classes", []))
        if not detected_classes:
            raise ValueError(f"No classes found for crop '{crop_name}' in {data_dir}")

        adapter = IndependentCropAdapter(
            crop_name=crop_name,
            model_name=str(
                training_cfg.get("backbone", {}).get(
                    "model_name",
                    "facebook/dinov3-vitl16-pretrain-lvd1689m",
                )
            ),
            device=self.device,
        )
        adapter.initialize_engine(class_names=detected_classes, config=self.config)

        checkpoint_records: List[Dict[str, Any]] = []
        batch_history: List[Dict[str, Any]] = []
        session_observers: List[Observer] = list(observers or [])
        session_holder: Dict[str, Any] = {}
        session_observers.append(
            self._build_workflow_observer(
                adapter=adapter,
                run_id=run_id,
                telemetry=telemetry,
                checkpoint_manager=checkpoint_manager,
                batch_history=batch_history,
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
        val_loader = loaders.get("val")
        val_loader_size = _loader_size(val_loader)
        calibration_loader = loaders["train"]
        if val_loader is not None and val_loader_size > 0:
            calibration_loader = val_loader
        ood_calibration = {}
        if _loader_size(calibration_loader) > 0:
            ood_calibration = adapter.calibrate_ood(calibration_loader)
        adapter_dir = adapter.save_adapter(str(output_dir))

        artifact_dir = Path(output_dir) / "training_metrics"
        loader_sizes = {name: _loader_size(loader) for name, loader in loaders.items()}
        training_artifacts = self._persist_training_artifacts(
            artifact_dir=artifact_dir,
            history_payload=history_payload,
            batch_history=batch_history,
            telemetry=telemetry,
        )
        trainer_for_artifacts = getattr(session, "trainer", None)
        validation_artifacts = self._persist_evaluation_artifacts(
            artifact_dir=artifact_dir,
            trainer=trainer_for_artifacts,
            loader=val_loader,
            detected_classes=detected_classes,
            telemetry=telemetry,
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
            split_name="val",
            artifact_subdir="validation",
        )
        test_artifacts = self._persist_evaluation_artifacts(
            artifact_dir=artifact_dir,
            trainer=trainer_for_artifacts,
            loader=loaders.get("test"),
            detected_classes=detected_classes,
            telemetry=telemetry,
            run_id=run_id,
            crop_name=crop_name,
            loader_sizes=loader_sizes,
            split_name="test",
            artifact_subdir="test",
        )

        summary_payload = self._build_summary_payload(
            run_id=run_id,
            crop_name=crop_name,
            detected_classes=detected_classes,
            loader_sizes=loader_sizes,
            adapter_dir=adapter_dir,
            artifact_dir=artifact_dir,
            checkpoint_records=checkpoint_records,
            ood_calibration=ood_calibration,
            history_payload=history_payload,
        )
        summary_artifacts = persist_training_summary_artifact(
            artifact_root=artifact_dir,
            summary_payload=summary_payload,
            telemetry=telemetry,
        )
        artifact_payload = _stringify_paths(
            {
                "training": training_artifacts,
                "validation": validation_artifacts.get("paths", {}),
                "test": test_artifacts.get("paths", {}),
                "summary": summary_artifacts,
            }
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
