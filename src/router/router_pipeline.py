"""Canonical router pipeline surface for SAM3 + BioCLIP-2.5 inference."""

from __future__ import annotations

from src.router.vlm_pipeline import VLMPipeline

# Preserve the old VLM name as a compatibility alias while maintained surfaces
# migrate to the router-specific name.
RouterPipeline = VLMPipeline

__all__ = ["RouterPipeline", "VLMPipeline"]
