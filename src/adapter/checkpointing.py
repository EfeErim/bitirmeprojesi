"""Checkpoint and metadata helpers for the independent crop adapter surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from src.shared.contracts import AdapterMetadata
from src.shared.json_utils import read_json_dict, write_json

TrainerFactory = Callable[[Dict[str, Any]], Any]
CheckpointPayloadFactory = Callable[[], type[Any]]


def resolve_training_checkpoint_root(checkpoint_dir: str | Path) -> Path:
    root = Path(checkpoint_dir)
    if root.is_dir() and (root / "training_checkpoint").exists():
        return root / "training_checkpoint"
    return root


def normalize_trainer_config(
    trainer_config: Optional[Dict[str, Any]],
    *,
    model_name: str,
    device: Any,
    backbone: Optional[Dict[str, Any]] = None,
    fusion: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = dict(trainer_config or {})
    if not normalized:
        normalized = {
            "backbone": dict(backbone or {"model_name": model_name}),
            "adapter": {
                "target_modules_strategy": "all_linear_transformer",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            },
            "fusion": dict(fusion or {"layers": [2, 5, 8, 11]}),
            "ood": {"threshold_factor": 2.0},
            "device": str(device),
        }
    normalized["device"] = str(normalized.get("device", str(device)))
    return normalized


def save_training_checkpoint(
    *,
    trainer: Any,
    checkpoint_dir: str,
    session_state: Optional[Dict[str, Any]] = None,
    run_id: str = "",
) -> Path:
    """Persist trainer and session state for fault-tolerant notebook runs."""
    root = Path(checkpoint_dir)
    checkpoint_dir_path = root / "training_checkpoint"
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    trainer_payload = trainer.snapshot_training_state()
    torch.save(trainer_payload.to_dict(), checkpoint_dir_path / "training_checkpoint.pt")

    session_payload = dict(session_state or {})
    meta = {
        "schema_version": trainer_payload.schema_version,
        "created_at": trainer_payload.created_at,
        "run_id": str(run_id),
        "current_epoch": int(trainer_payload.current_epoch),
        "global_step": int(dict(session_payload.get("progress_state", {})).get("global_step", 0)),
        "epoch": int(dict(session_payload.get("progress_state", {})).get("epoch", 0)),
    }
    write_json(checkpoint_dir_path / "session_state.json", session_payload)
    write_json(checkpoint_dir_path / "checkpoint_meta.json", meta)
    return checkpoint_dir_path


def load_training_checkpoint(
    *,
    checkpoint_dir: str,
    device: Any,
    trainer: Any,
    trainer_factory: TrainerFactory,
    checkpoint_payload_factory: CheckpointPayloadFactory,
    model_name: str,
) -> tuple[Any, Any, Dict[str, Any]]:
    """Load trainer and session payloads from a saved checkpoint directory."""
    root = resolve_training_checkpoint_root(checkpoint_dir)
    trainer_payload_raw = torch.load(root / "training_checkpoint.pt", map_location=device, weights_only=False)
    trainer_payload = checkpoint_payload_factory().from_dict(trainer_payload_raw)
    if trainer is None:
        normalized = normalize_trainer_config(
            trainer_payload.trainer_config,
            model_name=model_name,
            device=device,
        )
        trainer = trainer_factory(normalized)
    trainer_payload = trainer.restore_training_state(trainer_payload)

    session_payload: Dict[str, Any] = {}
    session_path = root / "session_state.json"
    if session_path.exists():
        session_payload = read_json_dict(session_path)
    return trainer, trainer_payload, session_payload


def build_checkpoint_load_result(*, trainer_payload: Any, session_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trainer_state": trainer_payload.to_dict(),
        "session_state": session_payload,
        "run_id": str(dict(session_payload).get("run_id", "")),
        "progress_state": dict(session_payload.get("progress_state", {})) if isinstance(session_payload, dict) else {},
        "history": dict(session_payload.get("history", {})) if isinstance(session_payload, dict) else {},
        "best_metric_state": (
            dict(session_payload.get("best_metric_state", {})) if isinstance(session_payload, dict) else {}
        ),
    }


def build_runtime_adapter_metadata(
    *,
    trainer: Any,
    class_to_idx: Dict[str, int],
    schema_version: str,
    engine: str,
    model_name: str,
    ood_calibration_version: int,
    target_modules_resolved: list[str],
    serialize_ood_state_fn: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    trainer_config = (
        trainer.config.as_contract_dict()
        if hasattr(trainer.config, "as_contract_dict")
        else {"backbone": {"model_name": getattr(trainer.config, "backbone_model_name", model_name)}}
    )
    ood_detector = getattr(trainer, "ood_detector", None)
    ood_state = (
        serialize_ood_state_fn(ood_detector)
        if hasattr(ood_detector, "class_stats")
        else {
            "threshold_factor": 2.0,
            "calibration_version": ood_calibration_version,
            "class_stats": {},
        }
    )
    return AdapterMetadata(
        schema_version=schema_version,
        engine=engine,
        trainer_config=trainer_config,
        backbone={
            "model_name": getattr(trainer.config, "backbone_model_name", model_name),
            "frozen": True,
        },
        fusion={
            "layers": list(getattr(trainer.config, "fusion_layers", [2, 5, 8, 11])),
            "output_dim": int(getattr(trainer.config, "fusion_output_dim", 768)),
            "dropout": float(getattr(trainer.config, "fusion_dropout", 0.1)),
            "gating": str(getattr(trainer.config, "fusion_gating", "softmax")),
        },
        class_to_idx=dict(class_to_idx),
        ood_calibration={"version": int(ood_calibration_version)},
        ood_state=ood_state,
        target_modules_resolved=list(target_modules_resolved),
    ).to_dict()
