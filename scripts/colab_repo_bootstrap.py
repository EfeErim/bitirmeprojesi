#!/usr/bin/env python3
"""Shared repository bootstrap helpers for Colab notebooks."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
GITHUB_TOKEN_NAMES = ("GH_TOKEN", "GITHUB_TOKEN")
TORCH_REQUIREMENT_PREFIXES = ("torch", "torchvision", "torchaudio")
SENSITIVE_TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml", ".csv", ".ini"}
MAX_PUSH_FILE_SIZE_BYTES = 95 * 1024 * 1024
EXCLUDED_PUSH_SUFFIXES = {".pt"}
EXCLUDED_PUSH_PATH_PARTS = ("checkpoint_state/checkpoints",)


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
    clone_url = _build_repo_access_url(repo_url, resolve_github_token())
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(clone_target)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)

    if completed.returncode == 0 and is_repo_root(clone_target):
        return clone_target

    if completed.returncode != 0 and "github.com" in str(repo_url):
        print(
            "Auto-clone failed. If this repository is private, set GH_TOKEN or GITHUB_TOKEN "
            "as an env var or Colab secret, or point AADS_REPO_ROOT to an existing repo checkout."
        )
    return None


def install_colab_requirements(req_path: Path, in_colab: bool) -> None:
    """Install notebook requirements with Colab-safe torch pin handling."""
    req = Path(req_path)
    if not req.exists():
        return

    if not in_colab:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)
        return

    filtered = _flatten_colab_safe_requirements(req)
    tmp_req = Path(tempfile.gettempdir()) / "aads_colab_requirements_no_torch.txt"
    tmp_req.parent.mkdir(parents=True, exist_ok=True)
    tmp_req.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(tmp_req)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        output = str(completed.stdout or "").strip()
        if output:
            print(output)
        raise RuntimeError(
            "Colab dependency installation failed for the filtered requirements set. "
            "See pip output above for details."
        )


def _requirement_name(requirement_line: str) -> str:
    requirement = str(requirement_line or "").strip().split(";", 1)[0].strip()
    if not requirement:
        return ""
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    return "" if match is None else str(match.group(1)).lower()


def _flatten_colab_safe_requirements(req_path: Path, _seen: Optional[set[Path]] = None) -> list[str]:
    resolved_path = Path(req_path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if resolved_path in seen:
        return []
    seen.add(resolved_path)

    filtered: list[str] = []
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if not stripped or stripped.startswith("#"):
            continue
        if lowered.startswith(("-r ", "--requirement ")):
            _, include_path = stripped.split(maxsplit=1)
            nested_path = (resolved_path.parent / include_path.strip()).resolve()
            filtered.extend(_flatten_colab_safe_requirements(nested_path, seen))
            continue
        if _requirement_name(stripped) in TORCH_REQUIREMENT_PREFIXES:
            continue
        filtered.append(stripped)
    return filtered


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
        "Set AADS_REPO_ROOT, or set AADS_REPO_URL/AADS_REPO_CLONE_TARGET. "
        "Private GitHub repos also require GH_TOKEN or GITHUB_TOKEN for auto-clone."
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


def export_current_colab_notebook(
    destination_path: str | Path,
    *,
    attempts: int = 3,
    retry_delay_sec: float = 1.0,
) -> Optional[Path]:
    """Write the current Colab notebook JSON, including cell outputs, to disk."""
    if not running_in_colab():
        return None

    try:
        from google.colab import _message
    except Exception:
        return None

    payload = None
    max_attempts = max(1, int(attempts))
    delay = max(0.0, float(retry_delay_sec))
    for attempt_index in range(max_attempts):
        try:
            response = _message.blocking_request("get_ipynb", timeout_sec=30)
        except Exception:
            response = None
        candidate = response.get("ipynb") if isinstance(response, dict) else None
        if isinstance(candidate, dict) and candidate:
            payload = candidate
            break
        # Colab occasionally returns an empty payload near runtime teardown.
        # Retry a few times before treating this as a soft failure so finalization can continue.
        if attempt_index + 1 < max_attempts and delay > 0.0:
            time.sleep(delay)

    if not isinstance(payload, dict) or not payload:
        return None

    destination = Path(destination_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def _read_json_dict(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=check,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        text=True,
    )


def _chunked(items: list[str], size: int = 200) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def _git_current_branch(repo_root: Path) -> str:
    completed = _run_git(["branch", "--show-current"], cwd=repo_root, capture_output=True)
    return str(completed.stdout or "").strip()


def _git_remote_url(repo_root: Path, remote_name: str) -> str:
    completed = _run_git(["remote", "get-url", remote_name], cwd=repo_root, capture_output=True)
    return str(completed.stdout or "").strip()


def _build_authenticated_remote_url(repo_url: str, token: str) -> str:
    parsed = urlsplit(str(repo_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(
            "GitHub auto-push currently supports only HTTPS remotes. "
            "Set origin to an https:// URL or disable auto-push."
        )
    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit((parsed.scheme, f"x-access-token:{token}@{netloc}", parsed.path, parsed.query, parsed.fragment))


def _clean_https_remote_url(repo_url: str) -> str:
    parsed = urlsplit(str(repo_url or "").strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(
            "GitHub auto-push currently supports only HTTPS remotes. "
            "Set origin to an https:// URL or disable auto-push."
        )
    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _redact_secret(value: str, secret: str) -> str:
    redacted = str(value or "")
    if secret:
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _redact_possible_tokens(value: str, secret: Optional[str] = None) -> str:
    redacted = _redact_secret(str(value or ""), str(secret or ""))
    # Common GitHub token formats that should never be written to telemetry.
    redacted = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{10,}\b", "<redacted>", redacted)
    redacted = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{10,}\b", "<redacted>", redacted)
    redacted = re.sub(r"AUTHORIZATION\s*:\s*basic\s+[A-Za-z0-9+/=]+", "AUTHORIZATION: <redacted>", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"AUTHORIZATION\s*:\s*bearer\s+[A-Za-z0-9_\-.=+/]+", "AUTHORIZATION: <redacted>", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"x-access-token:[^\s@/]+", "x-access-token:<redacted>", redacted, flags=re.IGNORECASE)
    redacted = re.sub(r"https://[^\s:@]+:[^\s@]+@github\.com", "https://<redacted>@github.com", redacted, flags=re.IGNORECASE)
    return redacted


def _redact_possible_tokens_with_secrets(value: str, secrets: Sequence[str]) -> str:
    redacted = str(value or "")
    for secret in [str(item).strip() for item in list(secrets or []) if str(item).strip()]:
        redacted = _redact_possible_tokens(redacted, secret)
    return _redact_possible_tokens(redacted, None)


def _contains_possible_tokens(value: str) -> bool:
    text = str(value or "")
    patterns = (
        r"\bgh[pousr]_[A-Za-z0-9_]{10,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{10,}\b",
        r"AUTHORIZATION\s*:\s*(?:basic|bearer)\s+[A-Za-z0-9_\-.=+/]+",
        r"x-access-token:[^\s@/]+",
        r"https://[^\s:@]+:[^\s@]+@github\.com",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _redact_nested_obj(value: Any, explicit_secrets: Sequence[str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_nested_obj(item, explicit_secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_nested_obj(item, explicit_secrets) for item in value]
    if isinstance(value, str):
        return _redact_possible_tokens_with_secrets(value, explicit_secrets)
    return value


def _sanitize_run_text_artifacts(run_dir: Path, *, explicit_secrets: Sequence[str]) -> dict[str, Any]:
    scanned = 0
    redacted_files = 0
    leaks: list[str] = []

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SENSITIVE_TEXT_SUFFIXES:
            continue

        scanned += 1
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        redacted_text = raw_text
        suffix = path.suffix.lower()

        if suffix == ".json":
            try:
                payload = json.loads(raw_text)
                redacted_payload = _redact_nested_obj(payload, explicit_secrets)
                redacted_text = json.dumps(redacted_payload, ensure_ascii=False, indent=2) + "\n"
            except Exception:
                redacted_text = _redact_possible_tokens_with_secrets(raw_text, explicit_secrets)
        elif suffix == ".jsonl":
            redacted_lines: list[str] = []
            for raw_line in raw_text.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    redacted_lines.append(raw_line)
                    continue
                try:
                    parsed_line = json.loads(stripped)
                    redacted_line_payload = _redact_nested_obj(parsed_line, explicit_secrets)
                    redacted_lines.append(json.dumps(redacted_line_payload, ensure_ascii=False))
                except Exception:
                    redacted_lines.append(_redact_possible_tokens_with_secrets(raw_line, explicit_secrets))
            redacted_text = "\n".join(redacted_lines)
            if raw_text.endswith("\n"):
                redacted_text += "\n"
        else:
            redacted_text = _redact_possible_tokens_with_secrets(raw_text, explicit_secrets)

        if redacted_text != raw_text:
            path.write_text(redacted_text, encoding="utf-8")
            redacted_files += 1

        if _contains_possible_tokens(redacted_text):
            leaks.append(path.as_posix())

    return {
        "scanned": scanned,
        "redacted_files": redacted_files,
        "leaks": leaks,
    }


def _should_exclude_run_file_from_push(path: Path, *, run_dir: Path) -> tuple[bool, str]:
    relative_to_run = path.relative_to(run_dir).as_posix().lower()
    if path.suffix.lower() in EXCLUDED_PUSH_SUFFIXES:
        return True, "excluded_suffix"
    if any(part in relative_to_run for part in EXCLUDED_PUSH_PATH_PARTS):
        return True, "excluded_checkpoint_path"
    try:
        if path.stat().st_size > MAX_PUSH_FILE_SIZE_BYTES:
            return True, "excluded_large_file"
    except OSError:
        return True, "excluded_unreadable"
    return False, "tracked"


def _realign_local_branch_to_remote(repo: Path, *, remote_name: str, branch: str) -> bool:
    remote_probe = _run_capture(["git", "ls-remote", "--heads", remote_name, f"refs/heads/{branch}"], cwd=repo)
    if remote_probe.returncode != 0 or not str(remote_probe.stdout or "").strip():
        return False

    fetch_completed = _run_git(["fetch", remote_name, branch], cwd=repo, check=False, capture_output=True)
    if fetch_completed.returncode != 0:
        output = str(fetch_completed.stdout or "").strip() or "fetch failed"
        raise RuntimeError(f"Git fetch failed before secure push: {output}")

    soft_reset = _run_git(["reset", "--soft", f"{remote_name}/{branch}"], cwd=repo, check=False, capture_output=True)
    if soft_reset.returncode != 0:
        output = str(soft_reset.stdout or "").strip() or "soft reset failed"
        raise RuntimeError(f"Git reset --soft failed before secure push: {output}")

    unstage_reset = _run_git(["reset"], cwd=repo, check=False, capture_output=True)
    if unstage_reset.returncode != 0:
        output = str(unstage_reset.stdout or "").strip() or "unstage reset failed"
        raise RuntimeError(f"Git reset failed before secure push: {output}")

    return True


def _run_git_ls_remote_with_token(*, remote_url: str, token: str, ref: str = "HEAD") -> subprocess.CompletedProcess[str]:
    clean_remote_url = _clean_https_remote_url(remote_url)
    auth_value = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    env = os.environ.copy()
    try:
        config_count = int(str(env.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        config_count = 0
    env["GIT_CONFIG_COUNT"] = str(config_count + 1)
    env[f"GIT_CONFIG_KEY_{config_count}"] = "http.https://github.com/.extraheader"
    env[f"GIT_CONFIG_VALUE_{config_count}"] = f"AUTHORIZATION: basic {auth_value}"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return subprocess.run(
        ["git", "ls-remote", clean_remote_url, ref],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def _run_git_push_with_token(*, repo: Path, remote_url: str, token: str, branch: str) -> None:
    clean_remote_url = _clean_https_remote_url(remote_url)
    auth_value = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    env = os.environ.copy()
    try:
        config_count = int(str(env.get("GIT_CONFIG_COUNT", "0") or "0"))
    except ValueError:
        config_count = 0
    env["GIT_CONFIG_COUNT"] = str(config_count + 1)
    env[f"GIT_CONFIG_KEY_{config_count}"] = "http.https://github.com/.extraheader"
    env[f"GIT_CONFIG_VALUE_{config_count}"] = f"AUTHORIZATION: basic {auth_value}"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    completed = subprocess.run(
        ["git", "push", clean_remote_url, f"HEAD:{branch}"],
        cwd=str(repo),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        output = _redact_secret(str(completed.stdout or ""), token)
        output_lower = output.lower()
        permission_hint = ""
        if (
            "permission to " in output_lower
            or "denied to" in output_lower
            or "http 403" in output_lower
            or "returned error: 403" in output_lower
        ):
            permission_hint = (
                " The token appears to lack write access to this repository. "
                "Use a GitHub token from an account or PAT that can push to the target repo, "
                "or set AUTO_PUSH_TO_GITHUB=False to keep the run local."
            )
        raise RuntimeError(
            f"GitHub push failed with exit code {completed.returncode}. "
            f"Output:\n{output.strip() or '<no output>'}{permission_hint}"
        )


def _build_repo_access_url(repo_url: str, token: Optional[str]) -> str:
    cleaned_url = str(repo_url or "").strip()
    if not cleaned_url or not token:
        return cleaned_url
    try:
        return _build_authenticated_remote_url(cleaned_url, token)
    except RuntimeError:
        return cleaned_url


def mirror_path_to_repo(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    exclude_dir_names: tuple[str, ...] = ("checkpoints",),
) -> Optional[Path]:
    """Copy a file or directory tree into the repo, optionally skipping directories by name."""
    source = Path(source_path).expanduser()
    if not source.exists():
        return None

    destination = Path(destination_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        shutil.copy2(source, destination)
        return destination

    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)

    excluded = set(str(name) for name in exclude_dir_names)

    def _ignore(current_dir: str, names: list[str]) -> set[str]:
        current = Path(current_dir)
        ignored: set[str] = set()
        for name in names:
            if name in excluded and (current / name).is_dir():
                ignored.add(name)
        return ignored

    shutil.copytree(source, destination, ignore=_ignore)
    return destination


def mirror_checkpoint_state_to_repo(
    source_root: str | Path,
    destination_root: str | Path,
) -> Optional[Path]:
    """Copy checkpoint metadata plus the mirrored best checkpoint only."""
    source = Path(source_root).expanduser()
    if not source.exists():
        return None

    destination = Path(destination_root).expanduser()
    mirrored_root = mirror_path_to_repo(source, destination, exclude_dir_names=("checkpoints",))
    if mirrored_root is None:
        return None

    source_checkpoints_dir = source / "checkpoints"
    if not source_checkpoints_dir.exists():
        return mirrored_root

    best_manifest = _read_json_dict(source / "best_checkpoint.json")
    best_name = str(best_manifest.get("name") or "").strip()
    source_best_path = source / "checkpoints" / "best"

    manifest_path = str(best_manifest.get("path") or "").strip()
    if manifest_path:
        candidate = Path(manifest_path).expanduser()
        if candidate.exists():
            source_best_path = candidate
        else:
            raise RuntimeError(f"Best checkpoint path from manifest was not found: {candidate}")
    elif best_name:
        named_candidate = source_checkpoints_dir / best_name
        if named_candidate.exists():
            source_best_path = named_candidate

    if not source_best_path.exists():
        if any(source_checkpoints_dir.iterdir()):
            raise RuntimeError(
                "Best checkpoint could not be resolved from checkpoint_state metadata. "
                "Check best_checkpoint.json and the checkpoint directory contents."
            )
        return mirrored_root

    destination_checkpoints_dir = destination / "checkpoints"
    destination_best_name = best_name or source_best_path.name or "best"
    destination_best_path = destination_checkpoints_dir / destination_best_name
    mirror_path_to_repo(source_best_path, destination_best_path, exclude_dir_names=())

    if best_manifest:
        best_manifest["path"] = str(destination_best_path)
        (destination / "best_checkpoint.json").write_text(
            json.dumps(best_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "latest_checkpoint.json").write_text(
            json.dumps(best_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "checkpoint_index.json").write_text(
            json.dumps([best_manifest], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return mirrored_root


def resolve_github_token() -> Optional[str]:
    """Resolve a GitHub token from env vars first, then Colab secrets."""
    for env_name in GITHUB_TOKEN_NAMES:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("GH_TOKEN", token)
            return token

    if not running_in_colab():
        return None

    for secret_name in GITHUB_TOKEN_NAMES:
        token = _resolve_colab_secret(secret_name)
        if token:
            os.environ["GH_TOKEN"] = token
            return token

    return None


def push_repo_run_to_github(
    repo_root: str | Path,
    run_id: str,
    *,
    remote_name: str = "origin",
    branch: Optional[str] = None,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Commit and push one mirrored runs/<RUN_ID> tree, excluding checkpoint blobs and oversized files."""
    emit = print if print_fn is None else print_fn
    repo = Path(repo_root).expanduser().resolve()
    run_dir = repo / "runs" / str(run_id)
    if not is_repo_root(repo):
        raise FileNotFoundError(f"Repository root not found: {repo}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run export directory not found: {run_dir}")

    resolved_token = str(token or resolve_github_token() or "").strip()
    if not resolved_token:
        raise RuntimeError("GitHub auto-push requires GH_TOKEN or GITHUB_TOKEN in env vars or Colab secrets.")

    resolved_branch = str(branch or os.environ.get("AADS_REPO_PUSH_BRANCH") or _git_current_branch(repo) or "").strip()
    if not resolved_branch:
        raise RuntimeError("Could not determine the target git branch for auto-push.")

    remote_url = _git_remote_url(repo, remote_name)
    relative_run_dir = run_dir.relative_to(repo).as_posix()

    realigned = _realign_local_branch_to_remote(repo, remote_name=remote_name, branch=resolved_branch)
    if realigned:
        emit(
            f"[GIT] Local branch realigned to {remote_name}/{resolved_branch} before secure run push."
        )

    sanitize_report = _sanitize_run_text_artifacts(run_dir, explicit_secrets=[resolved_token, str(resolve_hf_token() or "")])
    emit(
        f"[SECURITY] scanned={sanitize_report['scanned']} redacted_files={sanitize_report['redacted_files']}"
    )
    leaks = list(sanitize_report.get("leaks", []))
    if leaks:
        preview = leaks[:10]
        raise RuntimeError(
            "Push blocked: secret-like patterns are still present after sanitization in "
            f"{len(leaks)} file(s). First matches: {preview}"
        )

    tracked_files: list[str] = []
    skipped_files: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_to_repo = path.relative_to(repo).as_posix()
        excluded, _reason = _should_exclude_run_file_from_push(path, run_dir=run_dir)
        if excluded:
            skipped_files.append(relative_to_repo)
            continue
        tracked_files.append(relative_to_repo)

    _run_git(["config", "user.name", os.environ.get("AADS_GIT_USER_NAME", "AADS Colab")], cwd=repo)
    _run_git(["config", "user.email", os.environ.get("AADS_GIT_USER_EMAIL", "aads-colab@local")], cwd=repo)

    if tracked_files:
        for chunk in _chunked(tracked_files):
            _run_git(["add", "-f", "--", *chunk], cwd=repo)
    if skipped_files:
        for chunk in _chunked(skipped_files):
            _run_git(["rm", "--cached", "-r", "--ignore-unmatch", "--", *chunk], cwd=repo, check=False)

    staged = _run_git(["diff", "--cached", "--name-only", "--", relative_run_dir], cwd=repo, capture_output=True)
    staged_files = [line.strip() for line in str(staged.stdout or "").splitlines() if line.strip()]
    emit(
        f"[GIT] Stage prepared for runs/{run_id}: staged={len(staged_files)} skipped={len(skipped_files)}"
    )
    if not staged_files:
        emit(f"[GIT] No eligible repo mirror changes to push for runs/{run_id}.")
        return {
            "enabled": True,
            "pushed": False,
            "branch": resolved_branch,
            "remote_name": remote_name,
            "run_dir": str(run_dir),
            "staged_files": [],
            "skipped_files": skipped_files,
        }

    message = str(commit_message or f"Add notebook 2 outputs for run {run_id}")
    _run_git(["commit", "-m", message, "--", relative_run_dir], cwd=repo)
    _run_git_push_with_token(repo=repo, remote_url=remote_url, token=resolved_token, branch=resolved_branch)
    emit(f"[GIT] Pushed {len(staged_files)} file(s) from runs/{run_id} to {remote_name}/{resolved_branch}.")
    return {
        "enabled": True,
        "pushed": True,
        "branch": resolved_branch,
        "remote_name": remote_name,
        "run_dir": str(run_dir),
        "staged_files": staged_files,
        "skipped_files": skipped_files,
    }


def push_repo_paths_to_github(
    repo_root: str | Path,
    relative_paths: Sequence[str | Path],
    *,
    remote_name: str = "origin",
    branch: Optional[str] = None,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
    print_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Force-add selected repo-relative paths, commit them, and push to GitHub."""
    emit = print if print_fn is None else print_fn
    repo = Path(repo_root).expanduser().resolve()
    if not is_repo_root(repo):
        raise FileNotFoundError(f"Repository root not found: {repo}")

    normalized_paths: List[str] = []
    for raw_path in relative_paths:
        raw_text = str(raw_path).strip().replace("\\", "/")
        if raw_text.startswith("/") or ":" in raw_text.split("/", 1)[0]:
            raise ValueError(f"Path must be repo-relative and stay inside the repo: {raw_path}")
        relative = Path(raw_text).as_posix().strip().strip("/")
        if not relative or relative.startswith("../") or "/../" in relative:
            raise ValueError(f"Path must be repo-relative and stay inside the repo: {raw_path}")
        normalized_paths.append(relative)
    if not normalized_paths:
        raise ValueError("At least one repo-relative path is required.")

    resolved_token = str(token or resolve_github_token() or "").strip()
    if not resolved_token:
        raise RuntimeError("GitHub auto-push requires GH_TOKEN or GITHUB_TOKEN in env vars or Colab secrets.")

    resolved_branch = str(branch or os.environ.get("AADS_REPO_PUSH_BRANCH") or _git_current_branch(repo) or "").strip()
    if not resolved_branch:
        raise RuntimeError("Could not determine the target git branch for auto-push.")

    remote_url = _git_remote_url(repo, remote_name)

    _run_git(["config", "user.name", os.environ.get("AADS_GIT_USER_NAME", "AADS Colab")], cwd=repo)
    _run_git(["config", "user.email", os.environ.get("AADS_GIT_USER_EMAIL", "aads-colab@local")], cwd=repo)
    _run_git(["add", "-A", "-f", "--", *normalized_paths], cwd=repo)

    staged = _run_git(["diff", "--cached", "--name-only", "--", *normalized_paths], cwd=repo, capture_output=True)
    staged_files = [line.strip() for line in str(staged.stdout or "").splitlines() if line.strip()]
    if not staged_files:
        emit(f"[GIT] No eligible changes to push for: {', '.join(normalized_paths)}.")
        return {
            "enabled": True,
            "pushed": False,
            "branch": resolved_branch,
            "remote_name": remote_name,
            "paths": normalized_paths,
            "staged_files": [],
        }

    message = str(commit_message or f"Add generated repo assets: {', '.join(normalized_paths)}")
    _run_git(["commit", "-m", message, "--", *normalized_paths], cwd=repo)
    _run_git_push_with_token(repo=repo, remote_url=remote_url, token=resolved_token, branch=resolved_branch)
    emit(
        f"[GIT] Pushed {len(staged_files)} file(s) from "
        f"{', '.join(normalized_paths)} to {remote_name}/{resolved_branch}."
    )
    return {
        "enabled": True,
        "pushed": True,
        "branch": resolved_branch,
        "remote_name": remote_name,
        "paths": normalized_paths,
        "staged_files": staged_files,
    }


def resolve_hf_token() -> Optional[str]:
    """Resolve a Hugging Face token from env vars first, then Colab secrets."""
    for env_name in HF_TOKEN_NAMES:
        token = str(os.environ.get(env_name, "")).strip()
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            return token

    if not running_in_colab():
        return None

    for secret_name in HF_TOKEN_NAMES:
        token = _resolve_colab_secret(secret_name)
        if token:
            os.environ["HF_TOKEN"] = token
            return token

    return None


def _resolve_colab_secret(secret_name: str) -> str:
    if not running_in_colab():
        return ""

    try:
        from google.colab import userdata
    except Exception:
        return ""

    try:
        return str(userdata.get(secret_name) or "").strip()
    except Exception:
        return ""


def _run_capture(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout_sec: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1.0, float(timeout_sec)),
    )


def probe_repo_update_status(
    repo_root: str | Path,
    *,
    remote_name: str = "origin",
    branch: Optional[str] = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    if not is_repo_root(repo):
        return {"status": "unavailable", "message": f"Repository root not found: {repo}"}

    resolved_branch = str(branch or _git_current_branch(repo) or "").strip()
    if not resolved_branch:
        return {"status": "unavailable", "message": "Current git branch could not be determined."}

    local_head_completed = _run_git(["rev-parse", "HEAD"], cwd=repo, check=False, capture_output=True)
    local_head = str(local_head_completed.stdout or "").strip()
    if local_head_completed.returncode != 0 or not local_head:
        return {"status": "unavailable", "branch": resolved_branch, "message": "Local HEAD could not be resolved."}

    remote_completed = _run_capture(["git", "ls-remote", remote_name, f"refs/heads/{resolved_branch}"], cwd=repo)
    remote_stdout = str(remote_completed.stdout or "").strip()
    if remote_completed.returncode != 0 or not remote_stdout:
        return {
            "status": "unavailable",
            "branch": resolved_branch,
            "local_head": local_head,
            "message": "Remote branch information could not be read.",
            "detail": remote_stdout,
        }

    remote_head = remote_stdout.split()[0].strip()
    update_available = bool(remote_head and remote_head != local_head)
    return {
        "status": "ok",
        "branch": resolved_branch,
        "local_head": local_head,
        "remote_head": remote_head,
        "update_available": update_available,
        "relation": "update_available" if update_available else "up_to_date",
    }


def probe_github_repo_access(
    *,
    repo_url: Optional[str] = None,
    repo_root: Optional[str | Path] = None,
    token: Optional[str] = None,
) -> dict[str, Any]:
    resolved_repo_url = str(repo_url or "").strip()
    if not resolved_repo_url and repo_root is not None:
        repo = Path(repo_root).expanduser().resolve()
        if is_repo_root(repo):
            try:
                resolved_repo_url = _git_remote_url(repo, "origin")
            except Exception:
                resolved_repo_url = ""
    if not resolved_repo_url:
        resolved_repo_url = str(os.environ.get("AADS_REPO_URL", "")).strip()

    if not resolved_repo_url:
        return {"status": "unavailable", "message": "Repository URL could not be determined."}

    resolved_token = str(token or resolve_github_token() or "").strip()
    anonymous_probe = _run_capture(["git", "ls-remote", resolved_repo_url, "HEAD"])
    anonymous_ok = anonymous_probe.returncode == 0 and bool(str(anonymous_probe.stdout or "").strip())

    token_ok = anonymous_ok
    token_detail = _redact_possible_tokens(str(anonymous_probe.stdout or "").strip(), resolved_token)
    if not anonymous_ok and resolved_token:
        token_probe = _run_git_ls_remote_with_token(remote_url=resolved_repo_url, token=resolved_token, ref="HEAD")
        token_ok = token_probe.returncode == 0 and bool(str(token_probe.stdout or "").strip())
        token_detail = _redact_possible_tokens(str(token_probe.stdout or "").strip(), resolved_token)

    if anonymous_ok:
        read_access_mode = "public"
    elif resolved_token and token_ok:
        read_access_mode = "token_required"
    else:
        read_access_mode = "unavailable"

    parsed = urlsplit(resolved_repo_url)
    has_embedded_auth = "@" in str(parsed.netloc or "")
    push_ready = bool(resolved_token or has_embedded_auth or parsed.scheme == "ssh")
    return {
        "status": "ok" if read_access_mode != "unavailable" else "unavailable",
        "repo_url": resolved_repo_url,
        "token_present": bool(resolved_token),
        "read_access_mode": read_access_mode,
        "anonymous_read_access": bool(anonymous_ok),
        "token_read_access": bool(token_ok),
        "push_requires_auth": True,
        "push_ready": bool(push_ready),
        "detail": token_detail if token_detail else _redact_possible_tokens(str(anonymous_probe.stdout or "").strip(), resolved_token),
    }


def probe_hf_model_access(
    model_ids: Sequence[str],
    *,
    token: Optional[str] = None,
) -> dict[str, Any]:
    resolved_model_ids = [str(model_id).strip() for model_id in list(model_ids or []) if str(model_id).strip()]
    if not resolved_model_ids:
        return {"status": "skipped", "model_ids": [], "access_mode": "not_checked"}

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {
            "status": "unavailable",
            "model_ids": resolved_model_ids,
            "access_mode": "unavailable",
            "message": f"huggingface_hub import failed: {exc}",
        }

    resolved_token = str(token or resolve_hf_token() or "").strip()
    api_anon = HfApi()
    api_auth = HfApi(token=resolved_token) if resolved_token else None
    per_model: list[dict[str, Any]] = []
    for model_id in resolved_model_ids:
        anonymous_ok = False
        token_ok = False
        anonymous_detail = ""
        token_detail = ""
        try:
            api_anon.model_info(model_id)
            anonymous_ok = True
        except Exception as exc:
            anonymous_detail = f"{exc.__class__.__name__}: {exc}"
        if anonymous_ok:
            token_ok = True
        elif api_auth is not None:
            try:
                api_auth.model_info(model_id)
                token_ok = True
            except Exception as exc:
                token_detail = f"{exc.__class__.__name__}: {exc}"
        access_mode = "public" if anonymous_ok else "token_required" if token_ok else "unavailable"
        per_model.append(
            {
                "model_id": model_id,
                "access_mode": access_mode,
                "anonymous_ok": anonymous_ok,
                "token_ok": token_ok,
                "detail": token_detail or anonymous_detail,
            }
        )

    overall_mode = "public"
    if any(item["access_mode"] == "unavailable" for item in per_model):
        overall_mode = "unavailable"
    elif any(item["access_mode"] == "token_required" for item in per_model):
        overall_mode = "token_required"

    return {
        "status": "ok" if overall_mode != "unavailable" else "unavailable",
        "model_ids": resolved_model_ids,
        "token_present": bool(resolved_token),
        "access_mode": overall_mode,
        "requires_token_for_any": any(item["access_mode"] == "token_required" for item in per_model),
        "per_model": per_model,
    }


def collect_notebook_access_report(
    *,
    repo_root: Optional[str | Path] = None,
    repo_url: Optional[str] = None,
    hf_model_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved_repo_root = Path(repo_root).expanduser().resolve() if repo_root is not None else None
    github = probe_github_repo_access(repo_url=repo_url, repo_root=resolved_repo_root)
    updates = (
        probe_repo_update_status(resolved_repo_root)
        if resolved_repo_root is not None and is_repo_root(resolved_repo_root)
        else {"status": "unavailable", "message": "Repository root is not available yet."}
    )
    huggingface = probe_hf_model_access(list(hf_model_ids or []))
    return {
        "github": github,
        "repo_updates": updates,
        "huggingface": huggingface,
    }


def print_notebook_access_report(
    report: dict[str, Any],
    *,
    print_fn: Optional[Callable[[str], None]] = None,
) -> None:
    emit = print if print_fn is None else print_fn
    github = dict(report.get("github", {}))
    updates = dict(report.get("repo_updates", {}))
    huggingface = dict(report.get("huggingface", {}))

    relation = str(updates.get("relation", "unknown"))
    if relation == "up_to_date":
        emit("[KONTROL] Repo guncel gorunuyor.")
    elif relation == "update_available":
        emit(f"[KONTROL] Repo icin guncelleme var. Branch={updates.get('branch', '')}")
    else:
        emit(f"[KONTROL] Repo guncelleme durumu okunamadi: {updates.get('message', 'bilgi yok')}")

    read_access_mode = str(github.get("read_access_mode", "unavailable"))
    if read_access_mode == "public":
        emit("[KONTROL] GitHub okuma erisimi public; clone/pull icin ekstra token gerekmiyor.")
    elif read_access_mode == "token_required":
        emit("[KONTROL] GitHub okuma erisimi token istiyor; private repo icin GH_TOKEN gerekli.")
    else:
        emit("[KONTROL] GitHub okuma erisimi dogrulanamadi.")

    if bool(github.get("push_ready")):
        emit("[KONTROL] GitHub push icin kimlik bilgisi hazir.")
    else:
        emit("[KONTROL] GitHub push icin ek auth gerekli.")

    hf_mode = str(huggingface.get("access_mode", "not_checked"))
    if hf_mode == "public":
        emit("[KONTROL] Gerekli Hugging Face modelleri anonim erisimle aciliyor.")
    elif hf_mode == "token_required":
        emit("[KONTROL] En az bir Hugging Face modeli token istiyor; Colab secret kullanin.")
    elif hf_mode == "not_checked":
        emit("[KONTROL] Hugging Face model erisimi bu notebook icin ayrica kontrol edilmedi.")
    else:
        emit("[KONTROL] Hugging Face model erisimi dogrulanamadi.")


def login_and_check_hf_token(*, print_fn: Optional[Callable[[str], None]] = None) -> bool:
    """Authenticate once and validate the token with a lightweight identity lookup."""
    emit = print if print_fn is None else print_fn
    token = resolve_hf_token()
    if not token:
        emit(
            "[HF] Token bulunamadi. Inference veya egitimden once "
            "HF_TOKEN adli Colab secret ya da env var tanimlayin."
        )
        return False

    try:
        from huggingface_hub import HfApi, login
    except Exception as exc:
        emit(f"[HF] huggingface_hub import edilemedi: {exc}")
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
        emit(f"[HF] Kimlik dogrulandi: {username}")
        return True
    except Exception as exc:
        emit(f"[HF] Kimlik dogrulama kontrolu basarisiz: {exc}")
        return False
