#!/usr/bin/env python3
"""Shared helpers for script bundle runners."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_step(command: list[str], cwd: Path, title: str) -> int:
    print(f"\n[STEP] {title}")
    print("[CMD]", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        print(f"[FAIL] {title} (exit={completed.returncode})")
        return completed.returncode
    print(f"[OK] {title}")
    return 0
