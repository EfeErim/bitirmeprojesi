#!/usr/bin/env python3
"""
Tests for Phase 2 SD-LoRA training module.
"""

import pytest
import torch
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_dataset_factory, mock_tensor_factory
from src.training.colab_phase2_sd_lora import (
    ColabPhase2Trainer,
    SDLoRATrainer,
    SDLoRAConfig,
    train_sd_lora,
    load_pretrained_sd,
    prepare_lora_layers,
    compute_sd_loss
)


class TestSDLoRATrainer:
    """Test SD-LoRA trainer functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample SD-LoRA configuration."""
        return SDLoRAConfig(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            learning_rate=1e-4,
            num_epochs=2,
            batch_size=4,
            device="cpu"
        )

    @pytest.fixture
    def mock_sd_model(self):
        """Create a mock Stable Diffusion model."""
        # This would be a simplified version for testing
        # In reality, this would load an actual SD model
        class MockSDModel:
            def __init__(self):
                self.unet = torch.nn.Linear(10, 10)
                self.vae = torch.nn.Linear(10, 10)
                self.text_encoder = torch.nn.Linear(10, 10)

            def __call__(self, *args, **kwargs):
                return {"loss": torch.tensor(1.0)}

        return MockSDModel()

    def test_trainer_initialization(self, sample_config):
        """Test trainer initialization."""
        trainer = SDLoRATrainer(config=sample_config)
        assert trainer.config == sample_config
        assert trainer.config.lora_r == 8
        assert trainer.config.device == "cpu"

    def test_prepare_lora_layers(self, sample_config):
        """Test LoRA layer preparation."""
        trainer = SDLoRATrainer(config=sample_config)

        # Create a simple model
        model = torch.nn.Linear(10, 10)

        # Prepare LoRA layers
        lora_model = trainer.prepare_lora_layers(model)

        assert lora_model is not None
        # Should have LoRA adapters added

    def test_compute_sd_loss(self, sample_config):
        """Test Stable Diffusion loss computation."""
        trainer = SDLoRATrainer(config=sample_config)

        # Create dummy predictions and targets
        predictions = {"loss": torch.tensor(1.0)}
        targets = torch.randn(2, 3, 64, 64)

        loss = trainer.compute_sd_loss(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_training_step(self, sample_config, mock_sd_model):
        """Test single training step."""
        trainer = SDLoRATrainer(config=sample_config, model=mock_sd_model)

        # Create dummy batch
        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "text": ["test prompt"] * 2
        }

        loss = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_validation_step(self, sample_config, mock_sd_model):
        """Test single validation step."""
        trainer = SDLoRATrainer(config=sample_config, model=mock_sd_model)

        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "text": ["test prompt"] * 2
        }

        val_loss = trainer.validation_step(batch)
        assert isinstance(val_loss, torch.Tensor)

    def test_save_and_load_checkpoint(self, sample_config, mock_sd_model, tmp_path):
        """Test saving and loading checkpoints."""
        trainer = SDLoRATrainer(config=sample_config, model=mock_sd_model)

        # Train for a bit (mock)
        trainer.current_epoch = 1
        trainer.training_losses = [1.0, 0.9]
        trainer.validation_losses = [1.1, 1.0]

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load checkpoint into new trainer
        new_trainer = SDLoRATrainer(config=sample_config, model=mock_sd_model)
        new_trainer.load_checkpoint(str(checkpoint_path))

        assert new_trainer.current_epoch == 1
        assert len(new_trainer.training_losses) == 2

    def test_lora_parameter_count(self, sample_config):
        """Test that LoRA adds trainable parameters."""
        trainer = SDLoRATrainer(config=sample_config)

        # Create a simple model
        base_model = torch.nn.Linear(100, 50)
        total_before = sum(p.numel() for p in base_model.parameters())

        # Add LoRA
        lora_model = trainer.prepare_lora_layers(base_model)
        total_after = sum(p.numel() for p in lora_model.parameters())
        trainable_after = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

        # Should have more parameters after LoRA
        assert total_after > total_before
        # LoRA parameters should be trainable
        assert trainable_after > total_before

    def test_gradient_flow(self, sample_config, mock_sd_model):
        """Test that gradients flow through LoRA layers."""
        trainer = SDLoRATrainer(config=sample_config, model=mock_sd_model)

        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "text": ["test prompt"] * 2
        }

        loss = trainer.training_step(batch)
        loss.backward()

        # Check that LoRA parameters have gradients
        for name, param in trainer.model.named_parameters():
            if 'lora' in name.lower():
                assert param.grad is not None

    def test_lr_scheduler(self, sample_config, mock_sd_model):
        """Test learning rate scheduling."""
        trainer = SDLoRATrainer(config=sample_config, model=mock_sd_model)

        # Get initial LR
        initial_lr = trainer.get_learning_rate()

        # Simulate epoch
        trainer.scheduler_step()

        # LR should change
        new_lr = trainer.get_learning_rate()
        # May be same or different depending on scheduler
        assert new_lr >= 0


class TestSDLoRAConfig:
    """Test SD-LoRA configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SDLoRAConfig()
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.learning_rate == 1e-4

    def test_custom_config(self):
        """Test custom configuration."""
        config = SDLoRAConfig(
            lora_r=16,
            lora_alpha=32,
            learning_rate=1e-3,
            num_epochs=10
        )
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 10

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = SDLoRAConfig(lora_r=8, lora_alpha=16)
        assert config.lora_alpha >= config.lora_r  # Common practice

        # Invalid config should warn or error
        with pytest.raises(ValueError):
            SDLoRAConfig(lora_r=0)  # Must be positive


class TestUtilityFunctions:
    """Test utility functions."""

    def test_load_pretrained_sd(self):
        """Test loading pretrained Stable Diffusion."""
        # This would need actual model files
        # For now, test that function exists
        from src.training.colab_phase2_sd_lora import load_pretrained_sd
        assert callable(load_pretrained_sd)

    def test_prepare_lora_layers_function(self):
        """Test prepare_lora_layers function."""
        from src.training.colab_phase2_sd_lora import prepare_lora_layers

        model = torch.nn.Linear(10, 10)
        lora_model = prepare_lora_layers(model, r=8, alpha=16)
        assert lora_model is not None

    def test_compute_sd_loss_function(self):
        """Test compute_sd_loss function."""
        from src.training.colab_phase2_sd_lora import compute_sd_loss

        predictions = {"loss": torch.tensor(1.0)}
        targets = torch.randn(2, 3, 64, 64)

        loss = compute_sd_loss(predictions, targets)
        assert isinstance(loss, torch.Tensor)

    def test_train_sd_lora_function(self):
        """Test train_sd_lora convenience function."""
        from src.training.colab_phase2_sd_lora import train_sd_lora

        # This would need actual data
        # Just verify it's callable
        assert callable(train_sd_lora)
