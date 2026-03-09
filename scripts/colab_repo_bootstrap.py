#!/usr/bin/env python3
"""Shared repository bootstrap helpers for Colab notebooks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")


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


def install_colab_requirements(req_path: Path, in_colab: bool) -> None:
    """Install notebook requirements with Colab-safe torch pin handling."""
    req = Path(req_path)
    if not req.exists():
        return

    if not in_colab:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)
        return

    lines = req.read_text(encoding="utf-8").splitlines()
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("torch") or stripped.startswith("torchvision") or stripped.startswith("torchaudio"):
            continue
        filtered.append(line)

    tmp_req = Path("/tmp/aads_colab_requirements_no_torch.txt")
    tmp_req.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(tmp_req)], check=False)


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


def resolve_hf_token() -> Optional[str]:
    """Resolve a Hugging Face token from env vars first, then Colab secrets."""
    for env_name in HF_TOKEN_NAMES:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            return token

    if not running_in_colab():
        return None

    try:
        from google.colab import userdata
    except Exception:
        return None

    for secret_name in HF_TOKEN_NAMES:
        try:
            token = str(userdata.get(secret_name) or "").strip()
        except Exception:
            token = ""
        if token:
            os.environ["HF_TOKEN"] = token
            return token

    return None


def login_and_check_hf_token(*, print_fn: Optional[Callable[[str], None]] = None) -> bool:
    """Authenticate once and validate the token with a lightweight identity lookup."""
    emit = print if print_fn is None else print_fn
    token = resolve_hf_token()
    if not token:
        emit("[HF] No token found. Set a Colab secret or env var named HF_TOKEN before running inference.")
        return False

    try:
        from huggingface_hub import HfApi, login
    except Exception as exc:
        emit(f"[HF] Could not import huggingface_hub: {exc}")
        return False

    try:
        login(token=token, add_to_git_credential=False)
        profile = dict(HfApi(token=token).whoami() or {})
        username = str(
            profile.get("name")
            or profile.get("fullname")
            or profile.get("email")
            or profile.get("user")
            or "authenticated user"
        )
        emit(f"[HF] Authenticated as {username}")
        return True
    except Exception as exc:
        emit(f"[HF] Authentication check failed: {exc}")
        return False
