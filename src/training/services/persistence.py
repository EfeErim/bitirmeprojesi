"""Persistence helpers for trainer checkpoints, metadata, and OOD state."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict

import torch

from src.ood.continual_ood import ClassCalibration, ContinualOODDetector
from src.shared.contracts import AdapterMetadata


def compute_config_hash(contract: Dict[str, Any]) -> str:
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def serialize_ood_state(ood_detector: ContinualOODDetector) -> Dict[str, Any]:
    class_stats: Dict[str, Any] = {}
    for class_id, stats in ood_detector.class_stats.items():
        class_stats[str(class_id)] = {
            "mean": stats.mean.detach().cpu().tolist(),
            "var": stats.var.detach().cpu().tolist(),
            "mahalanobis_mu": float(stats.mahalanobis_mu),
            "mahalanobis_sigma": float(stats.mahalanobis_sigma),
            "energy_mu": float(stats.energy_mu),
            "energy_sigma": float(stats.energy_sigma),
            "threshold": float(stats.threshold),
        }
    return {
        "threshold_factor": float(ood_detector.threshold_factor),
        "calibration_version": int(ood_detector.calibration_version),
        "class_stats": class_stats,
    }


def restore_ood_state(
    payload: Dict[str, Any],
    *,
    default_threshold_factor: float,
) -> ContinualOODDetector:
    threshold_factor = float(payload.get("threshold_factor", default_threshold_factor))
    detector = ContinualOODDetector(threshold_factor=threshold_factor)
    detector.calibration_version = int(payload.get("calibration_version", 0))

    class_stats = payload.get("class_stats", {})
    if not isinstance(class_stats, dict):
        return detector

    for class_id_raw, stats in class_stats.items():
        if not isinstance(stats, dict):
            continue
        detector.class_stats[int(class_id_raw)] = ClassCalibration(
            mean=torch.tensor(stats.get("mean", []), dtype=torch.float32),
            var=torch.tensor(stats.get("var", []), dtype=torch.float32),
            mahalanobis_mu=float(stats.get("mahalanobis_mu", 0.0)),
            mahalanobis_sigma=float(stats.get("mahalanobis_sigma", 1.0)),
            energy_mu=float(stats.get("energy_mu", 0.0)),
            energy_sigma=float(stats.get("energy_sigma", 1.0)),
            threshold=float(stats.get("threshold", 0.0)),
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
