#!/usr/bin/env python3
"""Import smoke tests for active v6 module paths."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name,symbol_name",
    [
        ("src.utils.data_loader", "CropDataset"),
        ("src.router.vlm_pipeline", "VLMPipeline"),
        ("src.pipeline.independent_multi_crop_pipeline", "IndependentMultiCropPipeline"),
        ("src.ood.prototypes", "PrototypeComputer"),
    ],
)
def test_v6_symbol_imports(module_name: str, symbol_name: str) -> None:
    """Validate core v6 symbols are importable from active package paths."""
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol_name), f"{symbol_name} not found in {module_name}"
