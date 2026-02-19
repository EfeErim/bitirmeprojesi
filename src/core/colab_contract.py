"""Shared Colab runtime contract and step gate utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


COLAB_WORKSPACE_PATH = Path("/content/aads_ulora")
COLAB_DRIVE_MOUNT_PATH = Path("/content/drive/MyDrive/aads_ulora")

REQUIRED_WORKSPACE_DIRS = (
    "data",
    "models",
    "checkpoints",
    "logs",
    "outputs",
    "cache",
    "config",
    "scripts",
    "src",
    "tests",
    "docs",
    "colab_notebooks",
)

PLACEHOLDER_DRIVE_IDS = {
    "YOUR_FILE_ID_HERE",
    "REPLACE_WITH_FILE_ID",
    "TODO",
}


@dataclass(frozen=True)
class StepGate:
    step_id: str
    check_name: str
    passed: bool
    expected: str
    actual: str

    def as_log_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{self.step_id}] {status} :: {self.check_name} | "
            f"expected={self.expected} | actual={self.actual}"
        )


def is_placeholder_drive_id(value: str) -> bool:
    normalized = (value or "").strip()
    if not normalized:
        return True
    return normalized in PLACEHOLDER_DRIVE_IDS or normalized.startswith("YOUR_")


def required_workspace_paths(root: Path) -> Iterable[Path]:
    for directory in REQUIRED_WORKSPACE_DIRS:
        yield root / directory
