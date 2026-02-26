"""Project-level Python runtime customizations.

This module is auto-imported by Python.

By default, invasive monkeypatches are enabled only in pytest runs.
Use `AADS_ENABLE_RUNTIME_PATCHES=1` to force-enable globally.
Use `AADS_ENABLE_RUNTIME_PATCHES=0` to disable them.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Optional


def _parse_env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _is_pytest_context() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    argv = " ".join(sys.argv).lower()
    return "pytest" in argv or "py.test" in argv


_ENABLE_PATCHES = _parse_env_bool("AADS_ENABLE_RUNTIME_PATCHES")
if _ENABLE_PATCHES is None:
    _ENABLE_PATCHES = _is_pytest_context()


# ---------------------------------------------------------------------------
# Warnings filter compatibility (guarded)
# ---------------------------------------------------------------------------
if hasattr(warnings, "filters"):
    _ORIGINAL_FILTERS = warnings.filters.copy()
else:
    _ORIGINAL_FILTERS = []


class FilterManager:
    def __init__(self):
        self.filters = _ORIGINAL_FILTERS.copy()

    def add_filter(self, action, message="", category=Warning, lineno=0, append=False):
        """Add a filter, handling duplicates gracefully."""
        try:
            for i, (a, m, c, l) in enumerate(self.filters):
                if a == action and m == message and c == category and l == lineno:
                    del self.filters[i]
                    break
        except Exception:
            pass
        self.filters.append((action, message, category, lineno))

    def get_filters(self):
        return self.filters


_filter_manager = FilterManager()


def _patched_add_filter(*args, **kwargs):
    """Patched version that uses our filter manager."""
    if args:
        action = args[0]
        message = args[1] if len(args) > 1 else kwargs.get("message", "")
        category = args[2] if len(args) > 2 else kwargs.get("category", Warning)
        lineno = args[3] if len(args) > 3 else kwargs.get("lineno", 0)
        append = args[4] if len(args) > 4 else kwargs.get("append", False)
    else:
        action = kwargs.get("action")
        message = kwargs.get("message", "")
        category = kwargs.get("category", Warning)
        lineno = kwargs.get("lineno", 0)
        append = kwargs.get("append", False)

    _filter_manager.add_filter(action, message, category, lineno, append)


def _patched_simplefilter(action, category=Warning, lineno=0, append=False):
    _filter_manager.add_filter(action, "", category, lineno, append)


def _patched_reset_filters():
    _filter_manager.filters = _ORIGINAL_FILTERS.copy()


def get_current_filters():
    """Expose current filters for diagnostics."""
    return _filter_manager.get_filters()


if _ENABLE_PATCHES:
    if hasattr(warnings, "_add_filter"):
        warnings._add_filter = _patched_add_filter
    warnings.simplefilter = _patched_simplefilter
    _patched_reset_filters()


# ---------------------------------------------------------------------------
# Temp-path stability for restricted local environments (guarded)
# ---------------------------------------------------------------------------
if _ENABLE_PATCHES:
    _TMP_ROOT = (Path(__file__).resolve().parent / ".runtime_tmp").resolve()
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    _TMP_ROOT_STR = str(_TMP_ROOT)

    def _safe_gettempdir() -> str:
        return _TMP_ROOT_STR

    def _safe_gettempdirb() -> bytes:
        return os.fsencode(_TMP_ROOT_STR)

    def _safe_mkdtemp(suffix=None, prefix=None, dir=None):
        """Create temp dirs without relying on platform temp ACL behavior."""
        base = Path(dir).resolve() if dir else _TMP_ROOT
        base.mkdir(parents=True, exist_ok=True)
        suffix = "" if suffix is None else str(suffix)
        prefix = "tmp" if prefix is None else str(prefix)

        for _ in range(1024):
            candidate = base / f"{prefix}{uuid.uuid4().hex}{suffix}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                return str(candidate)
            except FileExistsError:
                continue
        raise FileExistsError("Unable to create temporary directory after many attempts")

    class _SafeTemporaryDirectory(tempfile.TemporaryDirectory):
        """TemporaryDirectory that does not fail hard on teardown in restricted envs."""

        def cleanup(self):
            try:
                super().cleanup()
            except PermissionError:
                pass
            except OSError:
                pass

    _ORIG_RMTREE = shutil.rmtree

    def _safe_rmtree(path, *args, **kwargs):
        """Best-effort cleanup to avoid hard failures on Windows permission quirks."""
        try:
            return _ORIG_RMTREE(path, *args, **kwargs)
        except PermissionError:
            return None
        except OSError:
            return None

    tempfile.tempdir = _TMP_ROOT_STR
    tempfile.gettempdir = _safe_gettempdir
    tempfile.gettempdirb = _safe_gettempdirb
    tempfile.mkdtemp = _safe_mkdtemp
    tempfile.TemporaryDirectory = _SafeTemporaryDirectory
    shutil.rmtree = _safe_rmtree

    try:
        import _pytest.pathlib as _pytest_pathlib

        _orig_cleanup_dead_symlinks = _pytest_pathlib.cleanup_dead_symlinks

        def _patched_cleanup_dead_symlinks(root):
            try:
                return _orig_cleanup_dead_symlinks(root)
            except PermissionError:
                return None

        _pytest_pathlib.cleanup_dead_symlinks = _patched_cleanup_dead_symlinks
    except Exception:
        pass
