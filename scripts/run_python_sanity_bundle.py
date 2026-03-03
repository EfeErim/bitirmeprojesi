#!/usr/bin/env python3
"""Run core Python sanity checks used during local development.

Bundle contents:
1) Notebook import/environment validation
2) Dynamic taxonomy regression check
3) Final pipeline sanity check
"""

from __future__ import annotations

import sys
from pathlib import Path

from _shared_runner import run_step


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python_exe = str(Path(sys.executable))

    steps = [
        ([python_exe, "scripts/validate_notebook_imports.py"], "Notebook import validation"),
        ([python_exe, "scripts/test_dynamic_taxonomy.py"], "Dynamic taxonomy sanity check"),
        ([python_exe, "scripts/test_pipeline_final_check.py"], "Pipeline final sanity check"),
    ]

    for command, title in steps:
        code = run_step(command=command, cwd=root, title=title)
        if code != 0:
            return code

    print("\n[SUCCESS] Python sanity bundle completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
