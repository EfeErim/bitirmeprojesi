"""Composable helpers for the continual training surface."""

from .metrics import compute_plan_metrics, load_plan_targets, validate_plan_metrics, write_plan_metric_artifact
from .persistence import (
    build_adapter_metadata,
    capture_rng_state,
    compute_config_hash,
    restore_ood_state,
    restore_rng_state,
    serialize_ood_state,
)
from .runtime import (
    amp_scaler_enabled,
    autocast_context,
    build_grad_scaler,
    configure_runtime_reproducibility,
    resolve_amp_dtype,
    resolve_session_num_epochs,
)

__all__ = [
    "amp_scaler_enabled",
    "autocast_context",
    "build_adapter_metadata",
    "build_grad_scaler",
    "capture_rng_state",
    "compute_config_hash",
    "compute_plan_metrics",
    "configure_runtime_reproducibility",
    "load_plan_targets",
    "resolve_amp_dtype",
    "resolve_session_num_epochs",
    "restore_ood_state",
    "restore_rng_state",
    "serialize_ood_state",
    "validate_plan_metrics",
    "write_plan_metric_artifact",
]
