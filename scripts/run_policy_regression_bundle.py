#!/usr/bin/env python3
"""Run Phase-4 policy validation bundle.

Bundle contents:
1) Profile policy sanity report
2) Targeted regression tests for policy graph + schema + router stability
"""

from __future__ import annotations

import subprocess
import sys
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable)

    steps = [
        (
            [str(python_exe), "scripts/profile_policy_sanity.py"],
            "Profile policy sanity check",
        ),
        (
            [
                str(python_exe),
                "-m",
                "pytest",
                "tests/unit/validation/test_schemas.py",
                "tests/unit/router/test_vlm_policy_stage_order.py",
                "tests/unit/router/test_vlm_strict_loading.py",
                "test_dynamic_taxonomy.py",
                "test_pipeline_final_check.py",
                "-q",
            ],
            "Targeted policy regression tests",
        ),
    ]

    for command, title in steps:
        code = run_step(command=command, cwd=root, title=title)
        if code != 0:
            return code

    print("\n[SUCCESS] Policy regression bundle completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
