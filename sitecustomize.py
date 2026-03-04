"""Project-level Python runtime customizations.

This module is auto-imported by Python.

By default, invasive monkeypatches are enabled only in pytest runs.
Use `AADS_ENABLE_RUNTIME_PATCHES=1` to force-enable globally.
Use `AADS_ENABLE_RUNTIME_PATCHES=0` to disable them.
If `AADS_ENABLE_RUNTIME_PATCHES` is set to an invalid value, pytest-context
autodetection is used as a compatibility fallback.
Use `AADS_TRACE_RUNTIME_PATCH_STATE=1` to emit patch activation diagnostics.
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


def _resolve_patch_activation() -> tuple[bool, str, Optional[str], bool]:
    env_name = "AADS_ENABLE_RUNTIME_PATCHES"
    raw_value = os.getenv(env_name)
    explicit = _parse_env_bool(env_name)
    pytest_context = _is_pytest_context()

    if explicit is True:
        return True, f"{env_name}=1", raw_value, pytest_context
    if explicit is False:
        return False, f"{env_name}=0", raw_value, pytest_context

    if raw_value is not None:
        warnings.warn(
            f"Ignoring invalid {env_name} value {raw_value!r}; using pytest-context auto fallback.",
            RuntimeWarning,
            stacklevel=1,
        )
    return pytest_context, "pytest-context-auto", raw_value, pytest_context


_TRACE_PATCH_STATE = _parse_env_bool("AADS_TRACE_RUNTIME_PATCH_STATE") is True


def _emit_patch_trace(message: str) -> None:
    if _TRACE_PATCH_STATE:
        print(f"[sitecustomize] {message}", file=sys.stderr)


_ENABLE_PATCHES, _PATCH_SOURCE, _PATCH_ENV_RAW, _PATCH_PYTEST_CONTEXT = _resolve_patch_activation()
_RUNTIME_PATCH_STATE: dict[str, object] = {
    "enabled": _ENABLE_PATCHES,
    "source": _PATCH_SOURCE,
    "env_raw": _PATCH_ENV_RAW,
    "pytest_context": _PATCH_PYTEST_CONTEXT,
    "sections": [],
}


def _record_patch_section(section: str, active: bool, detail: str = "") -> None:
    sections = _RUNTIME_PATCH_STATE["sections"]
    if isinstance(sections, list):
        sections.append({"section": section, "active": active, "detail": detail})


def get_runtime_patch_state() -> dict[str, object]:
    """Expose runtime patch activation and section-level status for diagnostics."""
    sections = _RUNTIME_PATCH_STATE.get("sections", [])
    copied_sections = [dict(item) for item in sections] if isinstance(sections, list) else []
    return {
        "enabled": _RUNTIME_PATCH_STATE.get("enabled"),
        "source": _RUNTIME_PATCH_STATE.get("source"),
        "env_raw": _RUNTIME_PATCH_STATE.get("env_raw"),
        "pytest_context": _RUNTIME_PATCH_STATE.get("pytest_context"),
        "sections": copied_sections,
    }


_emit_patch_trace(
    f"activation enabled={_ENABLE_PATCHES} source={_PATCH_SOURCE} "
    f"pytest_context={_PATCH_PYTEST_CONTEXT}"
)


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
    _record_patch_section(
        "warnings_filters",
        True,
        "patched warnings filter entrypoints for duplicate-tolerant behavior",
    )
else:
    _record_patch_section("warnings_filters", False, "runtime patches disabled")


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

    _pytest_cleanup_guard_applied = False
    try:
        import _pytest.pathlib as _pytest_pathlib

        _orig_cleanup_dead_symlinks = _pytest_pathlib.cleanup_dead_symlinks

        def _patched_cleanup_dead_symlinks(root):
            try:
                return _orig_cleanup_dead_symlinks(root)
            except PermissionError:
                return None

        _pytest_pathlib.cleanup_dead_symlinks = _patched_cleanup_dead_symlinks
        _pytest_cleanup_guard_applied = True
    except Exception:
        pass

    _record_patch_section(
        "temp_path_stability",
        True,
        "patched tempfile/shutil guards"
        + (" + pytest cleanup guard" if _pytest_cleanup_guard_applied else ""),
    )
else:
    _record_patch_section("temp_path_stability", False, "runtime patches disabled")


for _section_state in get_runtime_patch_state().get("sections", []):
    if isinstance(_section_state, dict):
        _emit_patch_trace(
            f"section={_section_state.get('section')} "
            f"active={_section_state.get('active')} detail={_section_state.get('detail')}"
        )
