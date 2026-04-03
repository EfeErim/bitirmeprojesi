"""Canonical router pipeline surface for SAM3 + BioCLIP-2.5 inference."""

from __future__ import annotations

from src.router.vlm_pipeline import VLMPipeline as _LegacyVLMPipeline


class RouterPipeline(_LegacyVLMPipeline):
    """Canonical router pipeline surface.

    The implementation still lives in the legacy compatibility module so older
    imports keep working, but maintained code should depend on this type.
    """


VLMPipeline = RouterPipeline

__all__ = ["RouterPipeline", "VLMPipeline"]
