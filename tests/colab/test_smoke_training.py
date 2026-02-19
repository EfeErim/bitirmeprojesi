"""
Smoke Tests for Colab Training Pipeline
Quick tests to verify training components work correctly.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.colab_phase1_training import ColabPhase1Trainer
from src.training.colab_phase2_sd_lora import ColabPhase2Trainer
from src.training.colab_phase3_conec_lora import ColabPhase3Trainer, CoNeCConfig


class TestPhase1Smoke:
    """Smoke tests for Phase 1 training."""

    def test_phase1_trainer_initialization(self):
        """Test Phase 1 trainer can be initialized."""
        trainer = ColabPhase1Trainer(
            model_name='facebook/dinov3-giant',
            num_classes=5,
            lora_r=32,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        assert trainer is not None
        assert trainer.device.type in ['cuda', 'cpu']
        assert trainer.num_classes == 5

    def test_phase1_optimizer_setup(self):
        """Test optimizer setup for Phase 1."""
        trainer = ColabPhase1Trainer(
            model_name='facebook/dinov3-giant',
            num_classes=5,
            device='cpu'  # Use CPU for testing
        )
        
        # Create a simple model for testing
        trainer.model = nn.Linear(10, 5)
        trainer.setup_optimizer()
        
        assert trainer.optimizer is not None
        assert len(trainer.optimizer.param_groups) > 0

    def test_phase1_training_step(self):
        """Test single training step."""
        trainer = ColabPhase1Trainer(
            model_name='facebook/dinov3-giant',
            num_classes=5,
            device='cpu'
        )
        
        trainer.model = nn.Linear(10, 5)
        trainer.setup_optimizer()
        
        # Create dummy batch
        batch = {
            'images': torch.randn(4, 3, 224, 224),
            'labels': torch.randint(0, 5, (4,))
        }
        
        # Should not raise an error
        loss = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_phase1_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        trainer = ColabPhase1Trainer(
            model_name='facebook/dinov3-giant',
            num_classes=5,
            device='cpu'
        )
        
        trainer.model = nn.Linear(10, 5)
        trainer.setup_optimizer()
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pth"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, loss=0.5)
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        assert trainer.current_epoch == 1


class TestPhase2Smoke:
    """Smoke tests for Phase 2 training."""

    def test_phase2_trainer_initialization(self):
        """Test Phase 2 trainer can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy adapter directory
            adapter_path = Path(tmpdir) / "dummy_adapter"
            adapter_path.mkdir()
            
            trainer = ColabPhase2Trainer(
                adapter_path=str(adapter_path),
                lora_r=16,
                device='cpu'
            )
            
            assert trainer is not None
            assert trainer.device.type == 'cpu'

    def test_phase2_optimizer_setup(self):
        """Test optimizer setup for Phase 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "dummy_adapter"
            adapter_path.mkdir()
            
            trainer = ColabPhase2Trainer(
                adapter_path=str(adapter_path),
                device='cpu'
            )
            
            # Mock model
            trainer.model = nn.Linear(10, 5)
            trainer.setup_optimizer()
            
            assert trainer.optimizer is not None


class TestPhase3Smoke:
    """Smoke tests for Phase 3 CoNeC-LoRA training."""

    def test_phase3_config_creation(self):
        """Test CoNeC config creation."""
        config = CoNeCConfig(
            lora_r=8,
            lora_alpha=16,
            learning_rate=5e-5,
            batch_size=16,
            device='cpu'
        )
        
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.learning_rate == 5e-5
        assert config.batch_size == 16

    def test_phase3_trainer_initialization(self):
        """Test Phase 3 trainer initialization."""
        config = CoNeCConfig(
            lora_r=8,
            lora_alpha=16,
            learning_rate=5e-5,
            batch_size=16,
            device='cpu'
        )
        
        trainer = ColabPhase3Trainer(config)
        
        assert trainer is not None
        assert trainer.device.type == 'cpu'
        assert trainer.config == config

    def test_phase3_prototype_manager(self):
        """Test prototype manager initialization."""
        config = CoNeCConfig(
            num_prototypes=10,
            prototype_dim=128,
            device='cpu'
        )
        
        trainer = ColabPhase3Trainer(config)
        
        prototypes = trainer.prototype_manager.get_prototypes()
        assert prototypes is not None
        assert prototypes.shape == (10, 128)

    def test_phase3_training_step(self):
        """Test single training step with CoNeC."""
        config = CoNeCConfig(
            lora_r=8,
            lora_alpha=16,
            learning_rate=5e-5,
            batch_size=4,
            device='cpu'
        )
        
        trainer = ColabPhase3Trainer(config)
        trainer.model = nn.Linear(10, 5)
        trainer.setup_optimizer()
        
        batch = {
            'images': torch.randn(4, 3, 224, 224),
            'labels': torch.randint(0, 5, (4,))
        }
        
        # Should not raise an error
        loss = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor)

    def test_phase3_validation_step(self):
        """Test validation step."""
        config = CoNeCConfig(device='cpu')
        trainer = ColabPhase3Trainer(config)
        trainer.model = nn.Linear(10, 5)
        
        batch = {
            'images': torch.randn(4, 3, 224, 224),
            'labels': torch.randint(0, 5, (4,))
        }
        
        metrics = trainer.validate([batch])
        assert 'loss' in metrics
        assert 'accuracy' in metrics

    def test_phase3_checkpoint_save_load(self, tmp_path):
        """Test checkpoint operations."""
        config = CoNeCConfig(device='cpu')
        trainer = ColabPhase3Trainer(config)
        trainer.model = nn.Linear(10, 5)
        trainer.setup_optimizer()
        
        # Save checkpoint
        checkpoint_dir = tmp_path / "checkpoints"
        trainer.save_checkpoint(str(checkpoint_dir), epoch=1, loss=0.5)
        
        # Check files exist
        assert (checkpoint_dir / 'checkpoint_epoch_1.pth').exists()
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_dir / 'checkpoint_epoch_1.pth'))
        assert trainer.current_epoch == 1

    def test_phase3_ood_detection(self):
        """Test OOD detection functionality."""
        config = CoNeCConfig(device='cpu')
        trainer = ColabPhase3Trainer(config)
        trainer.model = nn.Linear(10, 5)
        
        # Create dummy features
        features = torch.randn(10, 128)
        labels = torch.randint(0, 5, (10,))
        
        # Perform OOD detection
        ood_metrics = trainer._perform_ood_detection(features, labels)
        
        assert 'prototype_distances' in ood_metrics
        assert 'prototype_anomaly' in ood_metrics

    def test_phase3_prototype_update(self):
        """Test prototype updating."""
        config = CoNeCConfig(device='cpu')
        trainer = ColabPhase3Trainer(config)
        
        # Create dummy features and labels
        features = torch.randn(20, 128)
        labels = torch.randint(0, 5, (20,))
        
        # Update prototypes
        trainer._update_prototypes(features, labels)
        
        prototypes = trainer.prototype_manager.get_prototypes()
        assert prototypes is not None
        assert not torch.allclose(prototypes, torch.zeros_like(prototypes))


class TestMemoryOptimization:
    """Test memory optimization features."""

    def test_mixed_precision_enabled(self):
        """Test that mixed precision is enabled by default."""
        config = CoNeCConfig(device='cuda' if torch.cuda.is_available() else 'cpu')
        trainer = ColabPhase3Trainer(config)
        
        if torch.cuda.is_available():
            assert trainer.use_amp is True
            assert trainer.scaler.is_enabled()
        else:
            assert trainer.use_amp is False

    def test_gradient_accumulation(self):
        """Test gradient accumulation steps."""
        config = CoNeCConfig(
            gradient_accumulation_steps=4,
            device='cpu'
        )
        trainer = ColabPhase3Trainer(config)
        
        assert trainer.gradient_accumulation_steps == 4
        assert trainer.current_step == 0

    def test_memory_monitor(self):
        """Test memory monitor initialization."""
        config = CoNeCConfig(device='cpu')
        trainer = ColabPhase3Trainer(config)
        
        assert hasattr(trainer, 'memory_monitor')
        assert trainer.memory_monitor is not None


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_phase1_small_training(self):
        """Test small-scale Phase 1 training."""
        # Create dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 20
            
            def __getitem__(self, idx):
                return torch.randn(3, 224, 224), torch.randint(0, 3, (1,)).item()
        
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        
        # Create trainer
        trainer = ColabPhase1Trainer(
            model_name='facebook/dinov3-giant',
            num_classes=3,
            lora_r=8,  # Smaller for testing
            batch_size=2,
            num_epochs=1,
            device='cpu'
        )
        
        # Create a simple model for testing
        trainer.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(384, 3)  # DINOv3 hidden size
        )
        trainer.setup_optimizer()
        
        # Create loaders
        from src.dataset.colab_dataloader import ColabDataLoader
        train_loader = ColabDataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = ColabDataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Train for one epoch
        history = trainer.train(train_loader, val_loader, num_epochs=1)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = CoNeCConfig(
            temperature=0.07,
            contrastive_weight=0.1,
            orthogonal_weight=0.01
        )
        assert config.temperature == 0.07
        assert config.contrastive_weight + config.orthogonal_weight <= 1.0
        
        # Invalid config - negative temperature
        with pytest.raises(ValueError):
            CoNeCConfig(temperature=-0.1)
        
        # Invalid config - weights sum > 1
        with pytest.raises(ValueError):
            CoNeCConfig(contrastive_weight=0.8, orthogonal_weight=0.3)


class TestPerformance:
    """Performance regression tests."""

    def test_training_speed_not_degraded(self):
        """Ensure training speed is within acceptable range."""
        # This is a placeholder for actual performance tests
        # In production, you would measure iterations per second
        # and compare against baseline
        
        config = CoNeCConfig(device='cpu', batch_size=2)
        trainer = ColabPhase3Trainer(config)
        trainer.model = nn.Linear(10, 5)
        trainer.setup_optimizer()
        
        batch = {
            'images': torch.randn(2, 3, 224, 224),
            'labels': torch.randint(0, 5, (2,))
        }
        
        # Measure time for a few steps
        import time
        start = time.time()
        for _ in range(5):
            trainer.training_step(batch)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (adjust as needed)
        assert elapsed < 10.0  # 5 steps in under 10 seconds on CPU

    def test_memory_usage_reasonable(self):
        """Test that memory usage is within limits."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = CoNeCConfig(device='cuda', batch_size=4)
        trainer = ColabPhase3Trainer(config)
        
        # Measure initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run a training step
        batch = {
            'images': torch.randn(4, 3, 224, 224).cuda(),
            'labels': torch.randint(0, 5, (4,)).cuda()
        }
        trainer.training_step(batch)
        
        # Check memory usage
        peak_memory = torch.cuda.max_memory_allocated()
        memory_increase = peak_memory - initial_memory
        
        # Should not use excessive memory (adjust based on model)
        # For a small test, should be under 1GB
        assert memory_increase < 1e9  # 1GB in bytes


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'smoke'])