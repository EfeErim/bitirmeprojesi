"""Router module for AADS-ULoRA v5.5 VLM pipeline."""

from importlib import import_module

from src.router.vlm_pipeline import DiagnosticScoutingAnalyzer, VLMPipeline

__all__ = ["VLMPipeline", "DiagnosticScoutingAnalyzer", "SimpleCropRouter", "simple_crop_router"]


def __getattr__(name):
    """Lazily import optional router surfaces to keep package import lightweight."""
    if name == "simple_crop_router":
        return import_module("src.router.simple_crop_router")
    if name == "SimpleCropRouter":
        return import_module("src.router.simple_crop_router").SimpleCropRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
