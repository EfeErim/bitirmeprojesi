"""
End-to-End Integration Tests for Colab Migration
Tests the complete training pipeline from data preparation to evaluation.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.colab_phase1_training import ColabPhase1Trainer
from src.training.colab_phase2_sd_lora import ColabPhase2Trainer
from src.training.colab_phase3_conec_lora import ColabPhase3Trainer, CoNeCConfig
from src.dataset.colab_datasets import ColabCropDataset
from src.dataset.colab_dataloader import ColabDataLoader


@pytest.mark.heavy_model
class TestColabEndToEnd:
    """Complete end-to-end integration tests."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create temporary workspace structure."""
        workspace = tmp_path / "colab_workspace"
        workspace.mkdir()
        
        # Create directory structure
        dirs = ['data', 'models', 'checkpoints', 'logs', 'outputs', 'config']
        for d in dirs:
            (workspace / d).mkdir(exist_ok=True)
        
        # Create config
        config = {
            'colab': {
                'enabled': True,
                'gpu_type': 'Tesla T4',
                'memory_optimization': {
                    'gradient_checkpointing': True,
                    'mixed_precision': False  # Disable for CPU testing
                },
                'training': {
                    'gradient_accumulation_steps': 1,
                    'num_workers': 0,
                    'checkpoint_interval': 1,
                    'early_stopping_patience': 5
                }
            },
            'training': {
                'phase1': {
                    'model_name': 'facebook/dinov3-giant',
                    'num_classes': 3,
                    'lora_r': 8,
                    'lora_alpha': 8,
                    'learning_rate': 1e-4,
                    'num_epochs': 2,
                    'batch_size': 4
                },
                'phase2': {
                    'lora_r': 4,
                    'lora_alpha': 4,
                    'learning_rate': 1e-4,
                    'num_epochs': 1,
                    'batch_size': 2
                },
                'phase3': {
                    'lora_r': 4,
                    'lora_alpha': 8,
                    'learning_rate': 5e-5,
                    'num_epochs': 2,
                    'batch_size': 4,
                    'conec': {
                        'temperature': 0.07,
                        'prototype_dim': 64,
                        'num_prototypes': 3,
                        'contrastive_weight': 0.1,
                        'orthogonal_weight': 0.01
                    }
                }
            }
        }
        
        with open(workspace / 'config' / 'colab.json', 'w') as f:
            json.dump(config, f)
        
        return workspace

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for testing."""
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=20, num_classes=3, domain_shift=False):
                self.size = size
                self.num_classes = num_classes
                self.domain_shift = domain_shift
                self.classes = [f'class{i}' for i in range(num_classes)]
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                image = torch.randn(3, 224, 224)
                label = idx % self.num_classes
                
                if self.domain_shift:
                    # Domain shift dataset returns additional info
                    return {
                        'images': image,
                        'labels': label,
                        'domain': 1 if idx >= self.size // 2 else 0
                    }
                return {
                    'images': image,
                    'labels': label
                }
        
        return DummyDataset

    def test_complete_pipeline_phase1(self, temp_workspace, mock_dataset):
        """Test complete Phase 1 pipeline."""
        # Load config
        with open(temp_workspace / 'config' / 'colab.json', 'r') as f:
            config = json.load(f)
        
        phase1_config = config['training']['phase1']
        
        # Create datasets
        train_dataset = mock_dataset(size=20, num_classes=phase1_config['num_classes'])
        val_dataset = mock_dataset(size=10, num_classes=phase1_config['num_classes'])
        
        # Create data loader
        train_loader = ColabDataLoader(
            train_dataset,
            batch_size=phase1_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = ColabDataLoader(
            val_dataset,
            batch_size=phase1_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Create trainer
        trainer = ColabPhase1Trainer(
            model_name=phase1_config['model_name'],
            num_classes=phase1_config['num_classes'],
            lora_r=phase1_config['lora_r'],
            lora_alpha=phase1_config['lora_alpha'],
            learning_rate=phase1_config['learning_rate'],
            batch_size=phase1_config['batch_size'],
            device='cpu',
            colab_mode=True,
            checkpoint_dir=str(temp_workspace / 'checkpoints' / 'phase1')
        )
        
        trainer.setup_optimizer()
        
        # Train for one epoch
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=phase1_config['num_epochs'],
            save_dir=str(temp_workspace / 'checkpoints' / 'phase1')
        )
        
        # Verify training completed
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_accuracy' in history
        assert 'val_accuracy' in history
        assert len(history['train_loss']) == phase1_config['num_epochs']
        
        # Verify checkpoint was saved
        checkpoint_dir = temp_workspace / 'checkpoints' / 'phase1'
        assert any(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        # Save adapter
        adapter_path = temp_workspace / 'models' / 'phase1_adapter'
        trainer.save_adapter(str(adapter_path))
        assert adapter_path.exists()
        
        print("✅ Phase 1 integration test passed")

    def test_complete_pipeline_phase2(self, temp_workspace, mock_dataset):
        """Test complete Phase 2 pipeline."""
        # Assume Phase 1 adapter exists
        phase1_adapter_path = temp_workspace / 'models' / 'phase1_adapter'
        phase1_adapter_path.mkdir(parents=True)
        
        # Create a dummy checkpoint
        checkpoint = {
            'model_state_dict': {},
            'config': {'hidden_size': 384, 'num_labels': 3}
        }
        torch.save(checkpoint, phase1_adapter_path / 'checkpoint.pth')
        
        # Load config
        with open(temp_workspace / 'config' / 'colab.json', 'r') as f:
            config = json.load(f)
        
        phase2_config = config['training']['phase2']
        
        # Create datasets
        train_dataset = mock_dataset(size=15, num_classes=3)
        val_dataset = mock_dataset(size=8, num_classes=3)
        
        # Create data loaders
        train_loader = ColabDataLoader(
            train_dataset,
            batch_size=phase2_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = ColabDataLoader(
            val_dataset,
            batch_size=phase2_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Create trainer
        trainer = ColabPhase2Trainer(
            adapter_path=str(phase1_adapter_path),
            lora_r=phase2_config['lora_r'],
            lora_alpha=phase2_config['lora_alpha'],
            learning_rate=phase2_config['learning_rate'],
            batch_size=phase2_config['batch_size'],
            device='cpu'
        )
        
        trainer.setup_optimizer()
        
        # Train for one epoch
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=phase2_config['num_epochs'],
            save_dir=str(temp_workspace / 'checkpoints' / 'phase2')
        )
        
        # Verify training
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == phase2_config['num_epochs']
        
        # Save adapter
        adapter_path = temp_workspace / 'models' / 'phase2_adapter'
        trainer.save_adapter(str(adapter_path))
        assert adapter_path.exists()
        
        print("✅ Phase 2 integration test passed")

    def test_complete_pipeline_phase3(self, temp_workspace, mock_dataset):
        """Test complete Phase 3 pipeline."""
        # Assume Phase 2 adapter exists
        phase2_adapter_path = temp_workspace / 'models' / 'phase2_adapter'
        phase2_adapter_path.mkdir(parents=True)
        
        checkpoint = {
            'model_state_dict': {},
            'config': {'hidden_size': 384, 'num_labels': 3}
        }
        torch.save(checkpoint, phase2_adapter_path / 'checkpoint.pth')
        
        # Load config
        with open(temp_workspace / 'config' / 'colab.json', 'r') as f:
            config = json.load(f)
        
        phase3_config = config['training']['phase3']
        
        # Create datasets with domain shift
        train_dataset = mock_dataset(size=20, num_classes=3, domain_shift=True)
        val_dataset = mock_dataset(size=10, num_classes=3, domain_shift=True)
        
        # Create data loaders
        train_loader = ColabDataLoader(
            train_dataset,
            batch_size=phase3_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = ColabDataLoader(
            val_dataset,
            batch_size=phase3_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Create CoNeC config
        conec_config = CoNeCConfig(
            lora_r=phase3_config['lora_r'],
            lora_alpha=phase3_config['lora_alpha'],
            learning_rate=phase3_config['learning_rate'],
            batch_size=phase3_config['batch_size'],
            device='cpu',
            temperature=phase3_config['conec']['temperature'],
            prototype_dim=phase3_config['conec']['prototype_dim'],
            num_prototypes=phase3_config['conec']['num_prototypes'],
            contrastive_weight=phase3_config['conec']['contrastive_weight'],
            orthogonal_weight=phase3_config['conec']['orthogonal_weight'],
            gradient_accumulation_steps=1,
            use_amp=False
        )
        
        # Create trainer
        trainer = ColabPhase3Trainer(
            config=conec_config,
            checkpoint_dir=str(temp_workspace / 'checkpoints' / 'phase3')
        )
        
        trainer.setup_optimizer()
        
        # Train for one epoch
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=phase3_config['num_epochs'],
            save_dir=str(temp_workspace / 'checkpoints' / 'phase3')
        )
        
        # Verify training
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'contrastive_loss' in history
        assert 'orthogonal_loss' in history
        assert 'accuracy' in history
        assert len(history['train_loss']) == phase3_config['num_epochs']
        
        # Verify prototype manager
        prototypes = trainer.prototype_manager.get_prototypes()
        assert prototypes is not None
        assert prototypes.shape == (phase3_config['conec']['num_prototypes'], 
                                   phase3_config['conec']['prototype_dim'])
        
        # Save checkpoint
        checkpoint_path = temp_workspace / 'checkpoints' / 'phase3' / 'final.pth'
        trainer.save_checkpoint(str(checkpoint_path), epoch=0, loss=history['val_loss'][-1])
        assert checkpoint_path.exists()
        
        print("✅ Phase 3 integration test passed")

    def test_config_loading_and_validation(self, temp_workspace):
        """Test configuration loading and validation."""
        config_path = temp_workspace / 'config' / 'colab.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verify structure
        assert 'colab' in config
        assert 'training' in config
        assert 'phase1' in config['training']
        assert 'phase2' in config['training']
        assert 'phase3' in config['training']
        
        # Verify values
        assert config['colab']['enabled'] is True
        assert config['training']['phase1']['lora_r'] == 8
        assert config['training']['phase3']['conec']['temperature'] == 0.07
        
        print("✅ Configuration validation test passed")

    def test_checkpoint_resume(self, temp_workspace, mock_dataset):
        """Test resuming training from checkpoint."""
        # Phase 1: Train and save
        with open(temp_workspace / 'config' / 'colab.json', 'r') as f:
            config = json.load(f)
        
        phase1_config = config['training']['phase1']
        dataset = mock_dataset(size=10, num_classes=phase1_config['num_classes'])
        loader = ColabDataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        trainer = ColabPhase1Trainer(
            model_name=phase1_config['model_name'],
            num_classes=phase1_config['num_classes'],
            lora_r=phase1_config['lora_r'],
            device='cpu'
        )
        trainer.model = nn.Linear(10, phase1_config['num_classes'])
        trainer.setup_optimizer()
        
        # Train for 1 epoch
        history1 = trainer.train(loader, loader, num_epochs=1, 
                               save_dir=str(temp_workspace / 'checkpoints' / 'phase1'))
        
        # Save checkpoint
        checkpoint_path = temp_workspace / 'checkpoints' / 'phase1' / 'checkpoint_epoch_1.pth'
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        trainer2 = ColabPhase1Trainer(
            model_name=phase1_config['model_name'],
            num_classes=phase1_config['num_classes'],
            lora_r=phase1_config['lora_r'],
            device='cpu'
        )
        trainer2.model = nn.Linear(10, phase1_config['num_classes'])
        trainer2.setup_optimizer()
        trainer2.load_checkpoint(str(checkpoint_path))
        
        # Verify state was loaded
        assert trainer2.current_epoch == 1
        
        # Continue training
        history2 = trainer2.train(loader, loader, num_epochs=1, 
                                save_dir=str(temp_workspace / 'checkpoints' / 'phase1'))
        
        # Should have 2 epochs total
        assert len(history2['train_loss']) == 2
        
        print("✅ Checkpoint resume test passed")

    def test_metrics_consistency(self, temp_workspace, mock_dataset):
        """Test that metrics are consistently computed."""
        with open(temp_workspace / 'config' / 'colab.json', 'r') as f:
            config = json.load(f)
        
        phase1_config = config['training']['phase1']
        dataset = mock_dataset(size=10, num_classes=phase1_config['num_classes'])
        loader = ColabDataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        trainer = ColabPhase1Trainer(
            model_name=phase1_config['model_name'],
            num_classes=phase1_config['num_classes'],
            device='cpu'
        )
        trainer.model = nn.Linear(10, phase1_config['num_classes'])
        trainer.setup_optimizer()
        
        # Train
        history = trainer.train(loader, loader, num_epochs=1)
        
        # Verify metrics are valid
        assert all(isinstance(loss, float) for loss in history['train_loss'])
        assert all(isinstance(acc, float) for acc in history['train_accuracy'])
        assert all(0 <= acc <= 1 for acc in history['train_accuracy'])
        assert all(loss >= 0 for loss in history['train_loss'])
        
        print("✅ Metrics consistency test passed")

    def test_ood_detection_pipeline(self, temp_workspace):
        """Test OOD detection in Phase 3."""
        config = CoNeCConfig(
            num_prototypes=3,
            prototype_dim=64,
            device='cpu'
        )
        
        trainer = ColabPhase3Trainer(config)
        trainer.model = nn.Linear(10, 5)
        
        # Create synthetic features
        features = torch.randn(10, 64)
        labels = torch.randint(0, 3, (10,))
        
        # Update prototypes
        trainer._update_prototypes(features, labels)
        
        # Perform OOD detection
        ood_metrics = trainer._perform_ood_detection(features, labels)
        
        # Verify metrics
        assert 'prototype_distances' in ood_metrics
        assert 'prototype_anomaly' in ood_metrics
        assert ood_metrics['prototype_distances'].shape == (10,)
        assert ood_metrics['prototype_anomaly'].shape == (10,)
        
        print("✅ OOD detection pipeline test passed")

    def test_memory_optimization_features(self, temp_workspace):
        """Test that memory optimization features are properly configured."""
        config = CoNeCConfig(
            gradient_accumulation_steps=4,
            use_amp=False,  # Disabled for CPU
            device='cpu'
        )
        
        trainer = ColabPhase3Trainer(config)
        
        # Verify gradient accumulation
        assert trainer.gradient_accumulation_steps == 4
        assert trainer.current_step == 0
        
        # Verify mixed precision
        if torch.cuda.is_available():
            assert trainer.use_amp is True
        else:
            assert trainer.use_amp is False
        
        print("✅ Memory optimization test passed")

    def test_error_handling(self, temp_workspace):
        """Test error handling in training pipeline."""
        config = CoNeCConfig(device='cpu')
        trainer = ColabPhase3Trainer(config)
        
        # Test with invalid batch
        batch = {
            'images': torch.randn(2, 3, 224, 224),
            'labels': torch.randint(0, 5, (2,))
        }
        
        # Should handle NaN/Inf gracefully
        with pytest.raises(RuntimeError):
            # Force NaN loss
            with torch.no_grad():
                trainer.model = lambda x: torch.tensor(float('nan'))
                trainer.training_step(batch)
        
        print("✅ Error handling test passed")

    def test_model_save_load_consistency(self, temp_workspace):
        """Test that saved models can be loaded and produce consistent results."""
        config = CoNeCConfig(device='cpu')
        trainer1 = ColabPhase3Trainer(config)
        trainer1.model = nn.Linear(10, 5)
        trainer1.setup_optimizer()
        
        # Create input
        test_input = torch.randn(2, 3, 224, 224)
        
        # Get output before saving
        with torch.no_grad():
            output1 = trainer1.model(test_input)
        
        # Save checkpoint
        checkpoint_dir = temp_workspace / 'test_checkpoint'
        trainer1.save_checkpoint(str(checkpoint_dir), epoch=0, loss=0.5)
        
        # Load into new trainer
        trainer2 = ColabPhase3Trainer(config)
        trainer2.model = nn.Linear(10, 5)
        trainer2.load_checkpoint(str(checkpoint_dir / 'checkpoint_epoch_0.pth'))
        
        # Get output after loading
        with torch.no_grad():
            output2 = trainer2.model(test_input)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)
        
        print("✅ Model save/load consistency test passed")


class TestDataPipelineIntegration:
    """Test data pipeline integration."""

    def test_dataset_to_loader_pipeline(self, temp_workspace):
        """Test complete data pipeline from dataset to loader."""
        # Create mock dataset
        class TestDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {
                    'images': torch.randn(3, 224, 224),
                    'labels': idx % 3
                }
        
        dataset = TestDataset()
        
        # Create transforms
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Create Colab dataset wrapper
        colab_dataset = ColabCropDataset(temp_workspace / 'data', transform=transform)
        
        # Mock the underlying dataset
        colab_dataset.dataset = dataset
        
        # Create loader
        loader = ColabDataLoader(
            colab_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        # Iterate through batches
        batches = list(loader)
        assert len(batches) == 5  # 10 samples / batch_size 2
        
        for batch in batches:
            assert 'images' in batch
            assert 'labels' in batch
            assert batch['images'].shape[0] <= 2
            assert batch['labels'].shape[0] <= 2
        
        print("✅ Dataset to loader pipeline test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])