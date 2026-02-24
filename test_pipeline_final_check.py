#!/usr/bin/env python3
"""Compatibility wrapper for final pipeline sanity check.

Canonical script location: scripts/test_pipeline_final_check.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "test_pipeline_final_check.py"
    runpy.run_path(str(target), run_name="__main__")
