#!/usr/bin/env python3
"""Compatibility wrapper for Colab upload helper test.

Canonical script location: scripts/colab_test_upload.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "colab_test_upload.py"
    runpy.run_path(str(target), run_name="__main__")
