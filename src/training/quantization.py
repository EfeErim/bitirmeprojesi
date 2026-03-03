#!/usr/bin/env python3
"""Quantization utilities for v6 continual SD-LoRA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import importlib.util
import logging

logger = logging.getLogger(__name__)


PROHIBITED_4BIT_KEYS = {
    "load_in_" + "4bit",
    "bnb_" + "4bit_compute_dtype",
    "bnb_" + "4bit_quant_type",
    "bnb_" + "4bit_use_double_quant",
}

PROHIBITED_4BIT_TOKENS = (
    "4bit",
    "4-bit",
    "n" + "f4",
    "q" + "lora",
    "q-lora",
)


def _iter_items(value: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield str(idx), child


def find_prohibited_4bit_flags(config: Any, prefix: str = "") -> list[str]:
    """Return all paths that contain prohibited 4-bit quantization settings."""
    hits: list[str] = []
    if isinstance(config, dict):
        for key, value in config.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_lower = str(key).lower()
            if key_lower in PROHIBITED_4BIT_KEYS or any(token in key_lower for token in PROHIBITED_4BIT_TOKENS):
                hits.append(path)
            if isinstance(value, str):
                value_lower = value.lower()
                if any(token in value_lower for token in PROHIBITED_4BIT_TOKENS):
                    hits.append(path)
            elif isinstance(value, (dict, list)):
                hits.extend(find_prohibited_4bit_flags(value, path))
    elif isinstance(config, list):
        for idx, item in enumerate(config):
            path = f"{prefix}[{idx}]"
            hits.extend(find_prohibited_4bit_flags(item, path))
    elif isinstance(config, str):
        value_lower = config.lower()
        if any(token in value_lower for token in PROHIBITED_4BIT_TOKENS):
            hits.append(prefix or "<root>")
    return sorted(set(hits))


def assert_no_prohibited_4bit_flags(config: Any) -> None:
    hits = find_prohibited_4bit_flags(config)
    if hits:
        raise ValueError(
            "v6 forbids low-bit adapter settings outside hybrid int8 mode. Remove keys/values at: "
            + ", ".join(hits)
        )


@dataclass
class HybridINT8Config:
    """Configuration for outlier-aware hybrid INT8 loading."""

    mode: str = "int8_hybrid"
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = True
    strict_backend: bool = True
    allow_cpu_fallback: bool = False
    device_map: str = "auto"

    def validate(self) -> None:
        if str(self.mode).lower() != "int8_hybrid":
            raise ValueError("Only 'int8_hybrid' mode is supported in v6.")
        if self.llm_int8_threshold <= 0:
            raise ValueError("llm_int8_threshold must be positive.")


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _build_bnb_int8_config(cfg: HybridINT8Config):
    from transformers import BitsAndBytesConfig  # lazy import

    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=cfg.llm_int8_threshold,
        llm_int8_has_fp16_weight=cfg.llm_int8_has_fp16_weight,
    )


def load_hybrid_int8_backbone(
    model_name: str,
    *,
    auto_model_cls: Optional[Any] = None,
    cfg: Optional[HybridINT8Config] = None,
    strict_model_loading: bool = False,
) -> Any:
    """
    Load a backbone with hybrid INT8 quantization.

    No silent 4-bit fallback is allowed. If INT8 backend is unavailable and
    `allow_cpu_fallback` is false, this raises with an actionable message.
    """
    from transformers import AutoModel  # lazy import

    cfg = cfg or HybridINT8Config()
    cfg.validate()
    model_cls = auto_model_cls or AutoModel

    import torch

    has_bnb = _has_module("bitsandbytes")
    has_transformers_bnb = hasattr(__import__("transformers"), "BitsAndBytesConfig")

    if not torch.cuda.is_available():
        message = (
            "Hybrid INT8 backend requires a CUDA-enabled runtime, but torch.cuda.is_available() is False. "
            "Use a GPU runtime (e.g., Colab A100/T4/L4), or explicitly enable `allow_cpu_fallback` "
            "for non-quantized test/dev runs."
        )
        if cfg.strict_backend and not cfg.allow_cpu_fallback:
            raise RuntimeError(message)
        logger.warning("%s Falling back to non-quantized load due to explicit config.", message)
        return model_cls.from_pretrained(model_name)

    if has_bnb and has_transformers_bnb:
        quant_config = _build_bnb_int8_config(cfg)
        logger.info("Loading %s with hybrid INT8 (bitsandbytes enabled).", model_name)
        return model_cls.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=cfg.device_map,
        )

    message = (
        "Hybrid INT8 backend is unavailable (bitsandbytes/transformers integration missing). "
        "Install compatible bitsandbytes + transformers, or explicitly enable "
        "`allow_cpu_fallback` for non-quantized test/dev runs."
    )
    if cfg.strict_backend and not cfg.allow_cpu_fallback:
        raise RuntimeError(message)

    logger.warning("%s Falling back to non-quantized load due to explicit config.", message)
    return model_cls.from_pretrained(model_name)
