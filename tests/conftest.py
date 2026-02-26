"""
Pytest configuration and shared fixtures for AADS-ULoRA tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import warnings
import os
import shutil
import tempfile
import uuid

# Fix for Python 3.13 warnings filter issue
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# On this Windows/Python 3.13 environment, mode=0o700 can produce directories
# that are not scannable by the current user. Pytest tmpdir internals create
# temp paths with mode=0o700, so normalize that mode during tests.
_ORIG_PATH_MKDIR = Path.mkdir
_ORIG_TEMPFILE_TEMPDIR = tempfile.tempdir
_ORIG_TEMPFILE_GETTEMPDIR = tempfile.gettempdir
_ORIG_TEMPFILE_GETTEMPDIRB = tempfile.gettempdirb
_ORIG_TEMPFILE_MKDTEMP = tempfile.mkdtemp
_ORIG_SHUTIL_RMTREE = shutil.rmtree

_PYTEST_PATHLIB = None
_PYTEST_TMPDIR = None
_ORIG_PYTEST_PATHLIB_CLEANUP = None
_ORIG_PYTEST_TMPDIR_CLEANUP = None


def _safe_path_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    if os.name == "nt" and mode == 0o700:
        mode = 0o777
    return _ORIG_PATH_MKDIR(self, mode=mode, parents=parents, exist_ok=exist_ok)


Path.mkdir = _safe_path_mkdir


# Keep temporary path operations stable in restricted Windows environments.
_RUNTIME_TMP_ROOT = (Path(__file__).resolve().parent.parent / ".runtime_tmp").resolve()
_RUNTIME_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_gettempdir() -> str:
    return str(_RUNTIME_TMP_ROOT)


def _safe_gettempdirb() -> bytes:
    return os.fsencode(str(_RUNTIME_TMP_ROOT))


def _safe_mkdtemp(suffix=None, prefix=None, dir=None):
    base = Path(dir).resolve() if dir else _RUNTIME_TMP_ROOT
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
    raise FileExistsError("Unable to create temporary directory")


def _safe_rmtree(path, *args, **kwargs):
    try:
        return _ORIG_SHUTIL_RMTREE(path, *args, **kwargs)
    except PermissionError:
        return None
    except OSError:
        return None


tempfile.tempdir = str(_RUNTIME_TMP_ROOT)
tempfile.gettempdir = _safe_gettempdir
tempfile.gettempdirb = _safe_gettempdirb
tempfile.mkdtemp = _safe_mkdtemp
shutil.rmtree = _safe_rmtree

try:
    import _pytest.pathlib as _pytest_pathlib

    _PYTEST_PATHLIB = _pytest_pathlib
    _ORIG_PYTEST_PATHLIB_CLEANUP = _pytest_pathlib.cleanup_dead_symlinks

    def _patched_cleanup_dead_symlinks(root):
        try:
            return _ORIG_PYTEST_PATHLIB_CLEANUP(root)
        except PermissionError:
            return None

    _pytest_pathlib.cleanup_dead_symlinks = _patched_cleanup_dead_symlinks

    import _pytest.tmpdir as _pytest_tmpdir
    _PYTEST_TMPDIR = _pytest_tmpdir
    _ORIG_PYTEST_TMPDIR_CLEANUP = getattr(_pytest_tmpdir, "cleanup_dead_symlinks", None)
    _pytest_tmpdir.cleanup_dead_symlinks = _patched_cleanup_dead_symlinks
except Exception:
    pass


# Import fixtures from test_fixtures module
from tests.fixtures.test_fixtures import (
    mock_router_data,
    mock_pipeline_data,
    mock_ood_data,
    mock_adapter_data,
    mock_validation_data,
    mock_tensor_factory,
    mock_dataset_factory
)


@pytest.fixture(scope='session')
def test_device():
    """Fixture to provide appropriate device for testing."""
    return torch.device('cpu')  # Use CPU for consistent tests


@pytest.fixture(scope='session')
def test_seed():
    """Fixture to set random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def temp_dir():
    """Fixture to provide temporary directory for file operations."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    if not getattr(config.option, "basetemp", None):
        forced_base = _RUNTIME_TMP_ROOT / f"pytest_basetemp_{uuid.uuid4().hex}"
        config.option.basetemp = str(forced_base)

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "heavy_model: marks tests that may download/load large external models"
    )


# Optional: Add command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--runheavymodel",
        action="store_true",
        default=False,
        help="run tests marked as heavy_model"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if not config.getoption("--runheavymodel"):
        skip_heavy_model = pytest.mark.skip(reason="need --runheavymodel option to run")
        for item in items:
            if "heavy_model" in item.keywords:
                item.add_marker(skip_heavy_model)


def pytest_sessionfinish(session, exitstatus):
    """Restore global monkeypatches applied in this conftest."""
    Path.mkdir = _ORIG_PATH_MKDIR
    tempfile.tempdir = _ORIG_TEMPFILE_TEMPDIR
    tempfile.gettempdir = _ORIG_TEMPFILE_GETTEMPDIR
    tempfile.gettempdirb = _ORIG_TEMPFILE_GETTEMPDIRB
    tempfile.mkdtemp = _ORIG_TEMPFILE_MKDTEMP
    shutil.rmtree = _ORIG_SHUTIL_RMTREE

    if _PYTEST_PATHLIB is not None and _ORIG_PYTEST_PATHLIB_CLEANUP is not None:
        _PYTEST_PATHLIB.cleanup_dead_symlinks = _ORIG_PYTEST_PATHLIB_CLEANUP
    if _PYTEST_TMPDIR is not None and _ORIG_PYTEST_TMPDIR_CLEANUP is not None:
        _PYTEST_TMPDIR.cleanup_dead_symlinks = _ORIG_PYTEST_TMPDIR_CLEANUP


# Set up logging for tests
@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
