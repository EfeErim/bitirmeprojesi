#!/usr/bin/env python3
"""Shared repository bootstrap helpers for Colab notebooks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def is_repo_root(path: Path) -> bool:
    return (path / "src").is_dir() and (path / "config").is_dir() and (path / "scripts").is_dir()


def maybe_clone_repo() -> Optional[Path]:
    if os.environ.get("AADS_DISABLE_AUTO_CLONE") == "1":
        return None

    repo_url = os.environ.get("AADS_REPO_URL", "https://github.com/EfeErim/bitirmeprojesi.git")
    clone_target = Path(os.environ.get("AADS_REPO_CLONE_TARGET", "/content/bitirmeprojesi")).expanduser()

    if is_repo_root(clone_target):
        return clone_target

    if clone_target.exists() and any(clone_target.iterdir()):
        for child in clone_target.iterdir():
            if child.is_dir() and is_repo_root(child):
                return child
        return None

    clone_target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Repository not found locally. Auto-cloning from: {repo_url}")
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(clone_target)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)

    if completed.returncode == 0 and is_repo_root(clone_target):
        return clone_target
    return None


def resolve_repo_root() -> Path:
    env_candidates = [os.environ.get("AADS_REPO_ROOT"), os.environ.get("REPO_ROOT")]
    for raw in env_candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if is_repo_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if is_repo_root(candidate):
            return candidate

    common_candidates = [
        Path("/content/bitirme projesi"),
        Path("/content/bitirmeprojesi"),
        Path("/content/aads_ulora"),
        Path("/content/drive/MyDrive/bitirme projesi"),
        Path("/content/drive/MyDrive/bitirmeprojesi"),
    ]
    for candidate in common_candidates:
        if is_repo_root(candidate):
            return candidate

    auto_cloned = maybe_clone_repo()
    if auto_cloned is not None:
        return auto_cloned

    raise FileNotFoundError(
        "Repository root not found and auto-clone failed. "
        "Set AADS_REPO_ROOT, or set AADS_REPO_URL/AADS_REPO_CLONE_TARGET."
    )


def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except Exception:
        return False
    return True


def mount_drive_if_available(force_remount: bool = False) -> None:
    if not running_in_colab():
        return
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=force_remount)
    except Exception as exc:
        print(f"Drive mount skipped: {exc}")
