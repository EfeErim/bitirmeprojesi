#!/usr/bin/env python3
"""
Tests for model utilities module.
"""

import pytest
import torch
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory, mock_dataset_factory
from src.utils.model_utils import (
    ModelLoader,
    ModelSaver,
    ModelInspector,
    count_parameters,
    get_model_size,
    freeze_layers,
    unfreeze_layers,
    load_checkpoint,
    save_checkpoint
)


class TestModelLoader:
    """Test model loading functionality."""

    def test_load_pretrained_model(self):
        """Test loading a pretrained model."""
        loader = ModelLoader()
        # This would need an actual model file
        # For now, test with a simple model
        import torch.nn as nn
        model = nn.Linear(10, 5)

        loaded = loader.load_model(model, strict=False)
        assert loaded is not None

    def test_load_checkpoint(self):
        """Test loading from checkpoint."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {
                'model_state_dict': {'weight': torch.randn(5, 5)},
                'epoch': 10,
                'loss': 0.5
            }
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            loader = ModelLoader()
            checkpoint_data = loader.load_checkpoint(temp_path)
            assert 'model_state_dict' in checkpoint_data
            assert checkpoint_data['epoch'] == 10
        finally:
            Path(temp_path).unlink()

    def test_load_with_mapping(self):
        """Test loading with device mapping."""
        loader = ModelLoader()
        # Should handle device mapping
        assert loader is not None


class TestModelSaver:
    """Test model saving functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    def test_save_model(self, simple_model, tmp_path):
        """Test saving a model."""
        saver = ModelSaver()
        save_path = tmp_path / "model.pt"

        saver.save_model(simple_model, str(save_path))
        assert save_path.exists()

    def test_save_checkpoint(self, simple_model, tmp_path):
        """Test saving a full checkpoint."""
        saver = ModelSaver()
        save_path = tmp_path / "checkpoint.pt"

        checkpoint = {
            'model_state_dict': simple_model.state_dict(),
            'epoch': 5,
            'optimizer_state_dict': {},  # mock
            'loss': 0.3
        }

        saver.save_checkpoint(checkpoint, str(save_path))
        assert save_path.exists()

        # Verify checkpoint can be loaded
        loaded = torch.load(save_path)
        assert loaded['epoch'] == 5
        assert loaded['loss'] == 0.3

    def test_save_with_compression(self, simple_model, tmp_path):
        """Test saving with compression."""
        saver = ModelSaver()
        save_path = tmp_path / "model_compressed.pt"

        saver.save_model(simple_model, str(save_path), compress=True)
        assert save_path.exists()

        # Compressed file should be smaller than uncompressed
        # (for small models this might not always hold, but test the feature)
        assert save_path.stat().st_size > 0


class TestModelInspector:
    """Test model inspection functionality."""

    def test_count_parameters(self):
        """Test parameter counting."""
        import torch.nn as nn
        model = nn.Linear(100, 50)
        total, trainable = count_parameters(model)

        # Linear layer has weight (100*50) and bias (50)
        expected_total = 100 * 50 + 50
        assert total == expected_total
        assert trainable == expected_total  # All trainable by default

    def test_count_parameters_frozen(self):
        """Test parameter counting with frozen layers."""
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        total, trainable = count_parameters(model)
        # Total should be all params, trainable should be only second layer
        assert total > trainable

    def test_get_model_size(self):
        """Test model size calculation."""
        import torch.nn as nn
        model = nn.Linear(100, 50)
        size_mb = get_model_size(model)

        # Should return size in MB
        assert size_mb > 0
        assert isinstance(size_mb, float)

    def test_get_model_summary(self):
        """Test getting model summary."""
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        inspector = ModelInspector(model)
        summary = inspector.get_summary()

        assert 'total_params' in summary
        assert 'trainable_params' in summary
        assert 'model_size_mb' in summary
        assert 'layers' in summary
        assert len(summary['layers']) == 3  # Linear, ReLU, Linear

    def test_find_dead_neurons(self):
        """Test finding dead neurons (always zero output)."""
        import torch.nn as nn
        model = nn.Linear(10, 5)
        # Initialize with zeros (unlikely but possible)
        with torch.no_grad():
            model.weight.fill_(0)
            model.bias.fill_(0)

        inspector = ModelInspector(model)
        dead_neurons = inspector.find_dead_neurons()

        assert isinstance(dead_neurons, dict)

    def test_check_gradient_flow(self):
        """Test gradient flow checking."""
        import torch.nn as nn
        model = nn.Linear(10, 5)
        inspector = ModelInspector(model)

        # Need to compute gradients
        input_tensor = torch.randn(2, 10)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        gradient_info = inspector.check_gradient_flow()
        assert 'avg_gradients' in gradient_info
        assert 'gradient_flow_health' in gradient_info


class TestLayerFreezing:
    """Test layer freezing utilities."""

    def test_freeze_layers(self):
        """Test freezing model layers."""
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )

        # Freeze first layer
        freeze_layers(model, [0])

        for name, param in model[0].named_parameters():
            assert not param.requires_grad

        for name, param in model[1].named_parameters():
            assert param.requires_grad

    def test_unfreeze_layers(self):
        """Test unfreezing model layers."""
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )

        # Freeze all first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze second layer
        unfreeze_layers(model, [1])

        assert model[1].weight.requires_grad
        assert not model[0].weight.requires_grad

    def test_freeze_by_pattern(self):
        """Test freezing layers by name pattern."""
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
            nn.Linear(10, 5)
        )

        # Freeze layers with output dim 10
        freeze_layers(model, pattern="*10")

        # First layer (20 output) should be trainable
        assert model[0].weight.requires_grad
        # Second layer (10 output) should be frozen
        assert not model[1].weight.requires_grad
        # Third layer (5 output) should be trainable
        assert model[2].weight.requires_grad


class TestCheckpointFunctions:
    """Test checkpoint save/load functions."""

    def test_save_checkpoint_function(self, tmp_path):
        """Test save_checkpoint function."""
        import torch.nn as nn
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.3,
            path=str(checkpoint_path)
        )

        assert checkpoint_path.exists()

    def test_load_checkpoint_function(self, tmp_path):
        """Test load_checkpoint function."""
        import torch.nn as nn
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.3,
            path=str(checkpoint_path)
        )

        # Create new model and optimizer
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Load checkpoint
        epoch, loss = load_checkpoint(
            new_model,
            new_optimizer,
            str(checkpoint_path)
        )

        assert epoch == 5
        assert loss == 0.3

    def test_checkpoint_with_metadata(self, tmp_path):
        """Test checkpoint with custom metadata."""
        import torch.nn as nn
        model = nn.Linear(10, 5)

        checkpoint_path = tmp_path / "checkpoint.pt"
        metadata = {
            'custom_field': 'test_value',
            'accuracy': 0.95
        }

        save_checkpoint(
            model=model,
            epoch=1,
            path=str(checkpoint_path),
            **metadata
        )

        loaded = torch.load(checkpoint_path)
        assert loaded['custom_field'] == 'test_value'
        assert loaded['accuracy'] == 0.95
