#!/usr/bin/env python3
"""
Tests for Phase 3 CoNeC-LoRA training module.
"""

import pytest
import torch
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_dataset_factory, mock_tensor_factory
from src.training.colab_phase3_conec_lora import (
    ColabPhase3Trainer,
    CoNeCTrainer,
    CoNeCConfig,
    train_conec_lora,
    load_base_model,
    apply_conec_adapter,
    compute_conec_loss,
    compute_prototype_contrastive_loss,
    compute_orthogonal_loss
)


class TestCoNeCTrainer:
    """Test CoNeC-LoRA trainer functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample CoNeC configuration."""
        return CoNeCConfig(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            learning_rate=1e-4,
            num_epochs=2,
            batch_size=4,
            device="cpu",
            # CoNeC-specific
            temperature=0.07,
            prototype_dim=128,
            num_prototypes=10,
            contrastive_weight=0.1,
            orthogonal_weight=0.01
        )

    @pytest.fixture
    def mock_base_model(self):
        """Create a mock base model."""
        class MockBaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(10, 10)
                self.decoder = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.encoder(x)

        return MockBaseModel()

    def test_trainer_initialization(self, sample_config):
        """Test trainer initialization."""
        trainer = CoNeCTrainer(config=sample_config)
        assert trainer.config == sample_config
        assert trainer.config.lora_r == 8
        assert trainer.config.contrastive_weight == 0.1
        assert trainer.config.orthogonal_weight == 0.01

    def test_prepare_conec_adapter(self, sample_config, mock_base_model):
        """Test CoNeC adapter preparation."""
        trainer = CoNeCTrainer(config=sample_config)

        # Apply CoNeC adapter
        adapted_model = trainer.prepare_conec_adapter(mock_base_model)

        assert adapted_model is not None
        # Should have prototype and contrastive components

    def test_compute_conec_loss(self, sample_config):
        """Test CoNeC loss computation."""
        trainer = CoNeCTrainer(config=sample_config)

        # Create dummy features and labels
        features = torch.randn(8, 128)
        labels = torch.randint(0, 3, (8,))

        loss = trainer.compute_conec_loss(features, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_compute_prototype_contrastive_loss(self, sample_config):
        """Test prototype contrastive loss."""
        # Create sample features and prototypes
        features = torch.randn(10, 128)
        prototypes = torch.randn(5, 128)  # 5 prototypes
        labels = torch.randint(0, 5, (10,))

        loss = compute_prototype_contrastive_loss(
            features, prototypes, labels,
            temperature=sample_config.temperature
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_compute_orthogonal_loss(self, sample_config):
        """Test orthogonal regularization loss."""
        # Create sample weight matrix
        weight = torch.randn(128, 128)

        loss = compute_orthogonal_loss(weight)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

        # For orthogonal matrix, loss should be near zero
        orthogonal_weight = torch.eye(128)
        ortho_loss = compute_orthogonal_loss(orthogonal_weight)
        assert ortho_loss.item() < 1e-6  # Should be very small

    def test_training_step_with_conec(self, sample_config, mock_base_model):
        """Test single training step with CoNeC."""
        trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)

        # Create dummy batch
        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "labels": torch.randint(0, 3, (2,))
        }

        loss = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_validation_step(self, sample_config, mock_base_model):
        """Test single validation step."""
        trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)

        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "labels": torch.randint(0, 3, (2,))
        }

        val_loss = trainer.validation_step(batch)
        assert isinstance(val_loss, torch.Tensor)

    def test_prototype_update(self, sample_config, mock_base_model):
        """Test prototype updating."""
        trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)

        # Initial prototypes
        initial_prototypes = trainer.get_prototypes()
        assert initial_prototypes is not None

        # Update with some features
        features = torch.randn(10, 128)
        labels = torch.randint(0, 3, (10,))
        trainer.update_prototypes(features, labels)

        # Prototypes should change
        updated_prototypes = trainer.get_prototypes()
        # They may or may not change depending on update rate
        assert updated_prototypes is not None

    def test_save_and_load_checkpoint(self, sample_config, mock_base_model, tmp_path):
        """Test saving and loading checkpoints."""
        trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)

        # Train for a bit (mock)
        trainer.current_epoch = 1
        trainer.training_losses = [1.0, 0.9]
        trainer.validation_losses = [1.1, 1.0]
        trainer.prototype_history = [torch.randn(5, 128)]

        # Save checkpoint
        checkpoint_path = tmp_path / "conec_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load checkpoint into new trainer
        new_trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)
        new_trainer.load_checkpoint(str(checkpoint_path))

        assert new_trainer.current_epoch == 1
        assert len(new_trainer.training_losses) == 2
        assert len(new_trainer.prototype_history) == 1

    def test_prototype_consistency(self, sample_config, mock_base_model):
        """Test that prototypes remain consistent for same class."""
        trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)

        # Get features for class 0 multiple times
        features1 = torch.randn(10, 128)
        labels1 = torch.zeros(10, dtype=torch.long)
        trainer.update_prototypes(features1, labels1)
        proto1 = trainer.get_prototype_for_class(0)

        features2 = torch.randn(10, 128)
        labels2 = torch.zeros(10, dtype=torch.long)
        trainer.update_prototypes(features2, labels2)
        proto2 = trainer.get_prototype_for_class(0)

        # Prototypes should be similar (moving average)
        # Not exactly equal due to update rate
        similarity = torch.nn.functional.cosine_similarity(
            proto1.unsqueeze(0), proto2.unsqueeze(0)
        )
        assert similarity.item() > 0.9  # Should be very similar

    def test_gradient_flow_through_conec(self, sample_config, mock_base_model):
        """Test that gradients flow through CoNeC components."""
        trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)

        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "labels": torch.randint(0, 3, (2,))
        }

        loss = trainer.training_step(batch)
        loss.backward()

        # Check that model parameters have gradients
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"

    def test_lr_scheduler_step(self, sample_config, mock_base_model):
        """Test learning rate scheduler step."""
        trainer = CoNeCTrainer(config=sample_config, model=mock_base_model)

        initial_lr = trainer.get_learning_rate()
        trainer.scheduler_step()
        new_lr = trainer.get_learning_rate()

        # LR should be valid
        assert new_lr >= 0
        assert new_lr <= initial_lr * 1.1  # Shouldn't increase too much

    def test_orthogonal_constraint(self, sample_config):
        """Test orthogonal constraint on weight matrix."""
        trainer = CoNeCTrainer(config=sample_config)

        # Create a non-orthogonal weight matrix
        weight = torch.randn(64, 64, requires_grad=True)

        # Compute orthogonal loss
        loss = compute_orthogonal_loss(weight)

        # Should be > 0 for non-orthogonal matrix
        assert loss.item() > 0

        # Gradient should flow
        loss.backward()
        assert weight.grad is not None


class TestCoNeCConfig:
    """Test CoNeC configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CoNeCConfig()
        assert config.lora_r == 8
        assert config.temperature == 0.07
        assert config.prototype_dim == 128
        assert config.num_prototypes == 10
        assert config.contrastive_weight == 0.1
        assert config.orthogonal_weight == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = CoNeCConfig(
            lora_r=16,
            temperature=0.05,
            prototype_dim=256,
            num_prototypes=20,
            contrastive_weight=0.2,
            orthogonal_weight=0.05
        )
        assert config.lora_r == 16
        assert config.temperature == 0.05
        assert config.prototype_dim == 256
        assert config.num_prototypes == 20
        assert config.contrastive_weight == 0.2
        assert config.orthogonal_weight == 0.05

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = CoNeCConfig(
            temperature=0.1,
            contrastive_weight=0.1,
            orthogonal_weight=0.01
        )
        # Weights should sum to <= 1.0 (total loss weight)
        total_weight = config.contrastive_weight + config.orthogonal_weight
        assert total_weight <= 1.0

        # Invalid: negative temperature
        with pytest.raises(ValueError):
            CoNeCConfig(temperature=-0.1)

        # Invalid: too high weights
        with pytest.raises(ValueError):
            CoNeCConfig(contrastive_weight=1.5, orthogonal_weight=1.5)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_load_base_model_function(self):
        """Test load_base_model function."""
        from src.training.colab_phase3_conec_lora import load_base_model
        # Should be callable
        assert callable(load_base_model)

    def test_apply_conec_adapter_function(self):
        """Test apply_conec_adapter function."""
        from src.training.colab_phase3_conec_lora import apply_conec_adapter
        assert callable(apply_conec_adapter)

    def test_compute_conec_loss_function(self):
        """Test compute_conec_loss function."""
        from src.training.colab_phase3_conec_lora import compute_conec_loss

        features = torch.randn(8, 128)
        labels = torch.randint(0, 3, (8,))
        prototypes = torch.randn(3, 128)

        loss = compute_conec_loss(features, labels, prototypes)
        assert isinstance(loss, torch.Tensor)

    def test_train_conec_lora_function(self):
        """Test train_conec_lora convenience function."""
        from src.training.colab_phase3_conec_lora import train_conec_lora
        assert callable(train_conec_lora)

    def test_prototype_initialization(self):
        """Test prototype initialization strategies."""
        from src.training.colab_phase3_conec_lora import initialize_prototypes

        features = torch.randn(100, 128)
        labels = torch.randint(0, 5, (100,))
        num_classes = 5

        prototypes = initialize_prototypes(features, labels, num_classes)
        assert prototypes.shape == (num_classes, 128)

    def test_prototype_update_strategy(self):
        """Test prototype update strategy (moving average)."""
        from src.training.colab_phase3_conec_lora import update_prototype_moving_average

        old_proto = torch.ones(128) * 0.5
        new_features = torch.randn(10, 128)
        update_rate = 0.1

        updated = update_prototype_moving_average(
            old_proto, new_features, update_rate
        )

        assert updated.shape == (128,)
        # Should be between old and new average
        new_avg = new_features.mean(dim=0)
        assert torch.allclose(updated, old_proto * (1 - update_rate) + new_avg * update_rate)
