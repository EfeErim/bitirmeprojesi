"""Canonical training workflow for the supported adapter-training path."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import get_config
from src.utils.data_loader import create_training_loaders

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


@dataclass
class TrainingWorkflowResult:
    run_id: str
    crop_name: str
    class_names: List[str]
    history: Dict[str, Any]
    loader_sizes: Dict[str, int]
    adapter_dir: Path
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
        session_observers: List[Observer] = list(observers or [])
        session_holder: Dict[str, Any] = {}

        def _workflow_observer(event: Dict[str, Any]) -> None:
            if telemetry is not None:
                telemetry.emit_event(
                    str(event.get("event_type", "training_event")),
                    dict(event.get("payload", {})),
                    phase="training",
                    force_sync=False,
                )
            if checkpoint_manager is None or event.get("event_type") != "checkpoint_requested":
                return
            session = session_holder.get("session")
            if session is None:
                return
            payload = dict(event.get("payload", {}))
            record = checkpoint_manager.save_checkpoint(
                adapter=adapter,
                session=session,
                reason=str(payload.get("reason", "event")),
                run_id=run_id,
                mark_best=bool(payload.get("mark_best", False)),
                val_loss=payload.get("val_loss"),
            )
            checkpoint_records.append(dict(record))

        session_observers.append(_workflow_observer)

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

        if telemetry is not None:
            telemetry.emit_event(
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
        calibration_loader = loaders["val"] if _loader_size(loaders["val"]) > 0 else loaders["train"]
        ood_calibration = {}
        if _loader_size(calibration_loader) > 0:
            ood_calibration = adapter.calibrate_ood(calibration_loader)
        adapter_dir = adapter.save_adapter(str(output_dir))

        result = TrainingWorkflowResult(
            run_id=run_id,
            crop_name=crop_name,
            class_names=detected_classes,
            history=history.to_dict(),
            loader_sizes={name: _loader_size(loader) for name, loader in loaders.items()},
            adapter_dir=adapter_dir,
            ood_calibration=ood_calibration,
            checkpoint_records=checkpoint_records,
        )

        if telemetry is not None:
            telemetry.emit_event(
                "training_workflow_completed",
                result.to_dict(),
                phase="training",
            )

        return result
