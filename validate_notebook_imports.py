#!/usr/bin/env python3
"""Compatibility wrapper for notebook import validation.

Canonical script location: scripts/validate_notebook_imports.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "validate_notebook_imports.py"
    runpy.run_path(str(target), run_name="__main__")
