"""Persistence helpers for trainer checkpoints, metadata, OOD state, and adapter assets."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from src.ood.continual_ood import ClassCalibration, ContinualOODDetector
from src.shared.contracts import AdapterMetadata
from src.shared.json_utils import read_json_dict, write_json
from src.training.services.serialization import serialize_ood_state as build_serialized_ood_state
from src.training.types import TrainingCheckpointPayload


def _restore_class_index_state(trainer: Any, class_to_idx: Dict[str, int], target_modules_resolved: list[str]) -> None:
    trainer.class_to_idx = {str(k): int(v) for k, v in class_to_idx.items()}
    trainer.target_modules_resolved = [str(v) for v in target_modules_resolved]
    if hasattr(trainer, "_refresh_class_index_cache"):
        trainer._refresh_class_index_cache()


def _load_module_state(module: Any, state: Any, *, strict: bool = True) -> None:
    if module is None or state is None:
        return
    module.load_state_dict(state, strict=strict)


def _restore_checkpoint_component_states(trainer: Any, checkpoint: TrainingCheckpointPayload) -> None:
    model_state = checkpoint.model_state
    _load_module_state(trainer.adapter_model, model_state.get("adapter_model"), strict=False)
    _load_module_state(trainer.classifier, model_state.get("classifier"))
    _load_module_state(trainer.fusion, model_state.get("fusion"))


def _resolve_adapter_root(adapter_dir: str | Path) -> Path:
    root = Path(adapter_dir)
    if root.is_dir() and (root / "continual_sd_lora_adapter").exists():
        return root / "continual_sd_lora_adapter"
    return root


def _restore_exported_adapter_model(trainer: Any, root: Path, *, peft_model_cls: Any = None) -> None:
    adapter_config_path = root / "adapter_config.json"
    if adapter_config_path.exists() and peft_model_cls is not None and trainer.backbone is not None:
        loaded_adapter = peft_model_cls.from_pretrained(trainer.backbone, str(root), is_trainable=False)
        trainer.adapter_model = trainer._prepare_module_for_device(loaded_adapter, module_name="adapter_model")
        trainer._adapter_wrapped = True
        return
    adapter_model_path = root / "adapter_model.pt"
    if adapter_model_path.exists():
        _load_module_state(
            trainer.adapter_model,
            torch.load(adapter_model_path, map_location=trainer.device),
            strict=False,
        )


def _restore_exported_head_state(trainer: Any, root: Path) -> None:
    classifier_path = root / "classifier.pth"
    fusion_path = root / "fusion.pth"
    if classifier_path.exists():
        _load_module_state(trainer.classifier, torch.load(classifier_path, map_location=trainer.device))
    if fusion_path.exists():
        _load_module_state(trainer.fusion, torch.load(fusion_path, map_location=trainer.device))


def _restore_exported_ood_state(trainer: Any, meta: Dict[str, Any]) -> None:
    ood_state = meta.get("ood_state", {})
    if isinstance(ood_state, dict) and ood_state:
        trainer.ood_detector = restore_ood_state(
            ood_state,
            default_threshold_factor=trainer.config.ood_threshold_factor,
            device=trainer.device,
        )
    else:
        trainer.ood_detector.calibration_version = int(meta.get("ood_calibration", {}).get("version", 0))


def compute_config_hash(contract: Dict[str, Any]) -> str:
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assert_exportable_ood_state(ood_detector: ContinualOODDetector) -> None:
    """Fail fast when an adapter export would silently disable OOD."""
    ood_detector.assert_calibrated(operation="saving adapter assets")


def serialize_ood_state(ood_detector: ContinualOODDetector) -> Dict[str, Any]:
    return build_serialized_ood_state(ood_detector, strict=True)


def restore_ood_state(
    payload: Dict[str, Any],
    *,
    default_threshold_factor: float,
    device: torch.device | str | None = None,
) -> ContinualOODDetector:
    threshold_factor = float(payload.get("threshold_factor", default_threshold_factor))
    target_device = torch.device(device) if device is not None else torch.device("cpu")

    # Restore extended flags with backward-compatible defaults
    radial_beta_range_raw = payload.get("radial_beta_range", [0.5, 2.0])
    detector = ContinualOODDetector(
        threshold_factor=threshold_factor,
        radial_l2_enabled=bool(payload.get("radial_l2_enabled", False)),
        radial_beta_range=(float(radial_beta_range_raw[0]), float(radial_beta_range_raw[1])),
        radial_beta_steps=int(payload.get("radial_beta_steps", 16)),
        sure_enabled=bool(payload.get("sure_enabled", False)),
        sure_semantic_percentile=float(payload.get("sure_semantic_percentile", 95.0)),
        sure_confidence_percentile=float(payload.get("sure_confidence_percentile", 90.0)),
        conformal_enabled=bool(payload.get("conformal_enabled", False)),
        conformal_alpha=float(payload.get("conformal_alpha", 0.05)),
    )
    detector.calibration_version = int(payload.get("calibration_version", 0))

    # Restore radial beta and conformal qhat
    radial_beta = payload.get("radial_beta")
    detector.radial_beta = float(radial_beta) if radial_beta is not None else None
    conformal_qhat = payload.get("conformal_qhat")
    detector.conformal_qhat = float(conformal_qhat) if conformal_qhat is not None else None

    class_stats = payload.get("class_stats", {})
    if not isinstance(class_stats, dict):
        return detector

    for class_id_raw, stats in class_stats.items():
        if not isinstance(stats, dict):
            continue
        detector.class_stats[int(class_id_raw)] = ClassCalibration(
            mean=torch.tensor(stats.get("mean", []), dtype=torch.float32, device=target_device),
            var=torch.tensor(stats.get("var", []), dtype=torch.float32, device=target_device),
            mahalanobis_mu=float(stats.get("mahalanobis_mu", 0.0)),
            mahalanobis_sigma=float(stats.get("mahalanobis_sigma", 1.0)),
            energy_mu=float(stats.get("energy_mu", 0.0)),
            energy_sigma=float(stats.get("energy_sigma", 1.0)),
            threshold=float(stats.get("threshold", 0.0)),
            sure_semantic_threshold=float(stats.get("sure_semantic_threshold", 0.0)),
            sure_confidence_threshold=float(stats.get("sure_confidence_threshold", 0.0)),
        )
    return detector


def capture_rng_state(*, np_module: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    if np_module is not None:
        payload["numpy"] = np_module.random.get_state()
    return payload


def restore_rng_state(payload: Dict[str, Any], *, np_module: Any = None) -> None:
    if not isinstance(payload, dict):
        return
    try:
        torch_state = payload.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
    except Exception:
        pass
    try:
        cuda_state = payload.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception:
        pass
    try:
        python_state = payload.get("python")
        if python_state is not None:
            random.setstate(python_state)
    except Exception:
        pass
    try:
        numpy_state = payload.get("numpy")
        if numpy_state is not None and np_module is not None:
            np_module.random.set_state(numpy_state)
    except Exception:
        pass


def build_adapter_metadata(
    *,
    schema_version: str,
    engine: str,
    trainer_config: Dict[str, Any],
    config_hash: str,
    backbone_model_name: str,
    fusion_layers: list[int],
    fusion_output_dim: int,
    fusion_dropout: float,
    fusion_gating: str,
    class_to_idx: Dict[str, int],
    target_modules_resolved: list[str],
    ood_detector: ContinualOODDetector,
    peft_available: bool,
    adapter_wrapped: bool,
) -> AdapterMetadata:
    return AdapterMetadata(
        schema_version=schema_version,
        engine=engine,
        trainer_config=dict(trainer_config),
        config_hash=str(config_hash),
        backbone={"model_name": str(backbone_model_name), "frozen": True},
        fusion={
            "layers": list(fusion_layers),
            "output_dim": int(fusion_output_dim),
            "dropout": float(fusion_dropout),
            "gating": str(fusion_gating),
        },
        class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
        ood_calibration={"version": int(ood_detector.calibration_version)},
        ood_state=serialize_ood_state(ood_detector),
        target_modules_resolved=[str(item) for item in target_modules_resolved],
        adapter_runtime={
            "peft_available": bool(peft_available),
            "adapter_wrapped": bool(adapter_wrapped),
            "degraded_without_peft": bool(not adapter_wrapped),
        },
    )


def build_trainer_metadata_payload(trainer: Any) -> Dict[str, Any]:
    return build_adapter_metadata(
        schema_version="v6",
        engine="continual_sd_lora",
        trainer_config=trainer.config.as_contract_dict(),
        config_hash=trainer._config_hash,
        backbone_model_name=trainer.config.backbone_model_name,
        fusion_layers=list(trainer.config.fusion_layers),
        fusion_output_dim=trainer.config.fusion_output_dim,
        fusion_dropout=trainer.config.fusion_dropout,
        fusion_gating=trainer.config.fusion_gating,
        class_to_idx=dict(trainer.class_to_idx),
        target_modules_resolved=list(trainer.target_modules_resolved),
        ood_detector=trainer.ood_detector,
        peft_available=bool(trainer._peft_available),
        adapter_wrapped=bool(trainer._adapter_wrapped),
    ).to_dict()


def snapshot_training_state(trainer: Any, *, np_module: Any = None) -> TrainingCheckpointPayload:
    if trainer.adapter_model is None or trainer.classifier is None or trainer.fusion is None:
        raise RuntimeError("Cannot snapshot training state before initialization.")
    return TrainingCheckpointPayload(
        schema_version="v6_training_checkpoint",
        created_at=datetime.utcnow().isoformat() + "Z",
        trainer_config=trainer.config.as_contract_dict(),
        config_hash=trainer._config_hash,
        class_to_idx=dict(trainer.class_to_idx),
        target_modules_resolved=list(trainer.target_modules_resolved),
        model_state={
            "adapter_model": trainer.adapter_model.state_dict(),
            "classifier": trainer.classifier.state_dict(),
            "fusion": trainer.fusion.state_dict(),
        },
        optimizer_state=trainer.optimizer.state_dict() if trainer.optimizer is not None else None,
        scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        scaler_state=trainer.scaler.state_dict() if trainer.scaler.is_enabled() else None,
        ood_state=serialize_ood_state(trainer.ood_detector),
        rng_state=capture_rng_state(np_module=np_module),
        best_metric_state=dict(trainer.best_metric_state),
        current_epoch=int(trainer.current_epoch),
        optimizer_steps=int(trainer.optimizer_steps),
    )


def restore_training_state(
    trainer: Any,
    payload: TrainingCheckpointPayload | Dict[str, Any],
    *,
    np_module: Any = None,
) -> TrainingCheckpointPayload:
    checkpoint = (
        payload
        if isinstance(payload, TrainingCheckpointPayload)
        else TrainingCheckpointPayload.from_dict(payload)
    )
    _restore_class_index_state(trainer, checkpoint.class_to_idx, checkpoint.target_modules_resolved)

    needs_initialize = (
        trainer.adapter_model is None
        or trainer.classifier is None
        or trainer.fusion is None
    )
    if needs_initialize:
        trainer.initialize_engine(class_to_idx=trainer.class_to_idx)
    else:
        trainer._is_initialized = True

    _restore_checkpoint_component_states(trainer, checkpoint)

    if trainer.optimizer is None:
        trainer.setup_optimizer()
    if checkpoint.optimizer_state is not None and trainer.optimizer is not None:
        trainer.optimizer.load_state_dict(checkpoint.optimizer_state)
    if checkpoint.scheduler_state is not None:
        trainer._ensure_scheduler()
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(checkpoint.scheduler_state)
    if checkpoint.scaler_state is not None and trainer.scaler.is_enabled():
        trainer.scaler.load_state_dict(checkpoint.scaler_state)

    if isinstance(checkpoint.ood_state, dict):
        trainer.ood_detector = restore_ood_state(
            checkpoint.ood_state,
            default_threshold_factor=trainer.config.ood_threshold_factor,
            device=trainer.device,
        )
    restore_rng_state(checkpoint.rng_state, np_module=np_module)
    trainer.current_epoch = int(checkpoint.current_epoch)
    trainer.optimizer_steps = int(checkpoint.optimizer_steps)
    trainer.best_metric_state = dict(checkpoint.best_metric_state)
    return checkpoint


def save_trainer_adapter(trainer: Any, output_dir: str) -> Path:
    out = Path(output_dir)
    if trainer.adapter_model is None or trainer.classifier is None or trainer.fusion is None:
        raise RuntimeError("Cannot save adapter before initialization.")
    assert_exportable_ood_state(trainer.ood_detector)
    adapter_dir = out / "continual_sd_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(trainer.adapter_model, "save_pretrained"):
        trainer.adapter_model.save_pretrained(adapter_dir)
    else:
        torch.save(trainer.adapter_model.state_dict(), adapter_dir / "adapter_model.pt")

    torch.save(trainer.classifier.state_dict(), adapter_dir / "classifier.pth")
    torch.save(trainer.fusion.state_dict(), adapter_dir / "fusion.pth")

    write_json(adapter_dir / "adapter_meta.json", build_trainer_metadata_payload(trainer))
    return adapter_dir


def load_trainer_adapter(trainer: Any, adapter_dir: str, *, peft_model_cls: Any = None) -> Dict[str, Any]:
    root = _resolve_adapter_root(adapter_dir)
    meta_path = root / "adapter_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"adapter_meta.json not found in {root}")
    meta = read_json_dict(meta_path)
    _restore_class_index_state(
        trainer,
        dict(meta.get("class_to_idx", {})),
        list(meta.get("target_modules_resolved", [])),
    )

    trainer.initialize_engine(class_to_idx=trainer.class_to_idx)
    _restore_exported_adapter_model(trainer, root, peft_model_cls=peft_model_cls)
    _restore_exported_head_state(trainer, root)
    _restore_exported_ood_state(trainer, meta)
    return meta
