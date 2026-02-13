"""
Pytest configuration and shared fixtures for AADS-ULoRA tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


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
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
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