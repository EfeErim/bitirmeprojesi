"""Lazy router exports to avoid importing heavy optional dependencies at package import time."""

from __future__ import annotations

from typing import Any

__all__ = ["RouterPipeline", "VLMPipeline", "DiagnosticScoutingAnalyzer"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from src.router.diagnostic_scouting import DiagnosticScoutingAnalyzer
    from src.router.router_pipeline import RouterPipeline
    from src.router.vlm_pipeline import VLMPipeline

    exports = {
        "RouterPipeline": RouterPipeline,
        "VLMPipeline": VLMPipeline,
        "DiagnosticScoutingAnalyzer": DiagnosticScoutingAnalyzer,
    }
    return exports[name]
