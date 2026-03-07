"""Typed contracts shared across training, inference, and Colab helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OODAnalysis:
    ensemble_score: float = 0.0
    class_threshold: float = 0.0
    is_ood: bool = False
    calibration_version: int = 0
    mahalanobis_z: Optional[float] = None
    energy_z: Optional[float] = None
    # --- Radially Scaled L2 Normalization ---
    radial_beta: Optional[float] = None
    # --- SURE+ Double Scoring ---
    sure_semantic_score: Optional[float] = None
    sure_confidence_score: Optional[float] = None
    sure_semantic_ood: Optional[bool] = None
    sure_confidence_reject: Optional[bool] = None
    # --- Conformal Prediction ---
    conformal_set: Optional[List[str]] = None
    conformal_coverage: Optional[float] = None
    conformal_set_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ensemble_score": float(self.ensemble_score),
            "class_threshold": float(self.class_threshold),
            "is_ood": bool(self.is_ood),
            "calibration_version": int(self.calibration_version),
        }
        if self.mahalanobis_z is not None:
            payload["mahalanobis_z"] = float(self.mahalanobis_z)
        if self.energy_z is not None:
            payload["energy_z"] = float(self.energy_z)
        if self.radial_beta is not None:
            payload["radial_beta"] = float(self.radial_beta)
        if self.sure_semantic_score is not None:
            payload["sure_semantic_score"] = float(self.sure_semantic_score)
        if self.sure_confidence_score is not None:
            payload["sure_confidence_score"] = float(self.sure_confidence_score)
        if self.sure_semantic_ood is not None:
            payload["sure_semantic_ood"] = bool(self.sure_semantic_ood)
        if self.sure_confidence_reject is not None:
            payload["sure_confidence_reject"] = bool(self.sure_confidence_reject)
        if self.conformal_set is not None:
            payload["conformal_set"] = list(self.conformal_set)
        if self.conformal_coverage is not None:
            payload["conformal_coverage"] = float(self.conformal_coverage)
        if self.conformal_set_size is not None:
            payload["conformal_set_size"] = int(self.conformal_set_size)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "OODAnalysis":
        data = dict(payload or {})
        conformal_set_raw = data.get("conformal_set")
        return cls(
            ensemble_score=float(data.get("ensemble_score", 0.0)),
            class_threshold=float(data.get("class_threshold", 0.0)),
            is_ood=bool(data.get("is_ood", False)),
            calibration_version=int(data.get("calibration_version", 0)),
            mahalanobis_z=(
                None if data.get("mahalanobis_z") is None else float(data.get("mahalanobis_z", 0.0))
            ),
            energy_z=None if data.get("energy_z") is None else float(data.get("energy_z", 0.0)),
            radial_beta=(
                None if data.get("radial_beta") is None else float(data.get("radial_beta", 0.0))
            ),
            sure_semantic_score=(
                None if data.get("sure_semantic_score") is None
                else float(data.get("sure_semantic_score", 0.0))
            ),
            sure_confidence_score=(
                None if data.get("sure_confidence_score") is None
                else float(data.get("sure_confidence_score", 0.0))
            ),
            sure_semantic_ood=(
                None if data.get("sure_semantic_ood") is None
                else bool(data.get("sure_semantic_ood", False))
            ),
            sure_confidence_reject=(
                None if data.get("sure_confidence_reject") is None
                else bool(data.get("sure_confidence_reject", False))
            ),
            conformal_set=(
                None if conformal_set_raw is None else [str(c) for c in conformal_set_raw]
            ),
            conformal_coverage=(
                None if data.get("conformal_coverage") is None
                else float(data.get("conformal_coverage", 0.0))
            ),
            conformal_set_size=(
                None if data.get("conformal_set_size") is None
                else int(data.get("conformal_set_size", 0))
            ),
        )


@dataclass
class InferenceResult:
    status: str
    crop: Optional[str] = None
    part: Optional[str] = None
    router_confidence: float = 0.0
    diagnosis: Optional[str] = None
    diagnosis_index: Optional[int] = None
    confidence: float = 0.0
    message: str = ""
    ood_analysis: Optional[OODAnalysis] = None
    conformal_set: Optional[List[str]] = None

    def to_dict(self, *, include_ood: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": str(self.status),
            "crop": self.crop,
            "part": self.part,
            "router_confidence": float(self.router_confidence),
            "diagnosis": self.diagnosis,
            "confidence": float(self.confidence),
        }
        if self.diagnosis_index is not None:
            payload["diagnosis_index"] = int(self.diagnosis_index)
        if self.message:
            payload["message"] = str(self.message)
        if include_ood and self.ood_analysis is not None:
            payload["ood_analysis"] = self.ood_analysis.to_dict()
        if self.conformal_set is not None:
            payload["conformal_set"] = list(self.conformal_set)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "InferenceResult":
        data = dict(payload or {})
        diagnosis_index = data.get("diagnosis_index")
        conformal_set_raw = data.get("conformal_set")
        return cls(
            status=str(data.get("status", "unknown")),
            crop=data.get("crop"),
            part=data.get("part"),
            router_confidence=float(data.get("router_confidence", 0.0)),
            diagnosis=data.get("diagnosis"),
            diagnosis_index=None if diagnosis_index is None else int(diagnosis_index),
            confidence=float(data.get("confidence", 0.0)),
            message=str(data.get("message", "")),
            ood_analysis=(
                OODAnalysis.from_dict(data.get("ood_analysis"))
                if data.get("ood_analysis") is not None
                else None
            ),
            conformal_set=(
                None if conformal_set_raw is None else [str(c) for c in conformal_set_raw]
            ),
        )


@dataclass
class AdapterMetadata:
    schema_version: str = "v6"
    engine: str = "continual_sd_lora"
    trainer_config: Dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    backbone: Dict[str, Any] = field(default_factory=dict)
    fusion: Dict[str, Any] = field(default_factory=dict)
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    ood_calibration: Dict[str, Any] = field(default_factory=dict)
    ood_state: Dict[str, Any] = field(default_factory=dict)
    target_modules_resolved: List[str] = field(default_factory=list)
    adapter_runtime: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "engine": str(self.engine),
            "trainer_config": dict(self.trainer_config),
            "config_hash": str(self.config_hash),
            "backbone": dict(self.backbone),
            "fusion": dict(self.fusion),
            "class_to_idx": {str(k): int(v) for k, v in self.class_to_idx.items()},
            "ood_calibration": dict(self.ood_calibration),
            "ood_state": dict(self.ood_state),
            "target_modules_resolved": [str(item) for item in self.target_modules_resolved],
            "adapter_runtime": dict(self.adapter_runtime),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "AdapterMetadata":
        data = dict(payload or {})
        return cls(
            schema_version=str(data.get("schema_version", "v6")),
            engine=str(data.get("engine", "continual_sd_lora")),
            trainer_config=dict(data.get("trainer_config", {})),
            config_hash=str(data.get("config_hash", "")),
            backbone=dict(data.get("backbone", {})),
            fusion=dict(data.get("fusion", {})),
            class_to_idx={str(k): int(v) for k, v in dict(data.get("class_to_idx", {})).items()},
            ood_calibration=dict(data.get("ood_calibration", {})),
            ood_state=dict(data.get("ood_state", {})),
            target_modules_resolved=[str(item) for item in list(data.get("target_modules_resolved", []))],
            adapter_runtime=dict(data.get("adapter_runtime", {})),
        )


@dataclass
class CheckpointRecord:
    name: str
    path: Path
    created_at: str
    global_step: int
    epoch: int
    reason: str
    is_best: bool = False
    val_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            "created_at": self.created_at,
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "reason": str(self.reason),
            "is_best": bool(self.is_best),
        }
        if self.val_loss is not None:
            payload["val_loss"] = float(self.val_loss)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "CheckpointRecord":
        data = dict(payload or {})
        val_loss = data.get("val_loss")
        return cls(
            name=str(data.get("name", "")),
            path=Path(str(data.get("path", ""))),
            created_at=str(data.get("created_at", "")),
            global_step=int(data.get("global_step", 0)),
            epoch=int(data.get("epoch", 0)),
            reason=str(data.get("reason", "")),
            is_best=bool(data.get("is_best", False)),
            val_loss=None if val_loss is None else float(val_loss),
        )
