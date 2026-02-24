#!/usr/bin/env python3
"""Compatibility wrapper for dynamic taxonomy sanity check.

Canonical script location: scripts/test_dynamic_taxonomy.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "test_dynamic_taxonomy.py"
    runpy.run_path(str(target), run_name="__main__")
