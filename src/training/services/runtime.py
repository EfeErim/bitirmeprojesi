"""Runtime helpers for AMP, seeds, and session defaults."""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any, Optional

import torch


def configure_runtime_reproducibility(config: Any, *, np_module: Any = None) -> None:
    seed = int(getattr(config, "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if np_module is not None:
        np_module.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if bool(getattr(config, "deterministic", False)):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_amp_dtype(device: torch.device, mixed_precision: str) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    mode = str(mixed_precision).lower()
    if mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, mixed_precision: str) -> Any:
    dtype = resolve_amp_dtype(device, mixed_precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def amp_scaler_enabled(device: torch.device, mixed_precision: str) -> bool:
    return resolve_amp_dtype(device, mixed_precision) == torch.float16 and device.type == "cuda"


def build_grad_scaler(device: torch.device, mixed_precision: str):
    enabled = amp_scaler_enabled(device, mixed_precision)
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_namespace, "GradScaler", None)
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls(device.type, enabled=enabled)
        except TypeError:
            try:
                return grad_scaler_cls(enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_session_num_epochs(config: Any, explicit_num_epochs: Optional[int], *, default: int = 1) -> int:
    if explicit_num_epochs is not None:
        return int(max(1, explicit_num_epochs))
    configured = getattr(config, "num_epochs", default)
    try:
        return int(max(1, int(configured)))
    except Exception:
        return int(max(1, default))
