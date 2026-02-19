"""
Colab Data Pipeline Tests
Tests for data loading, preprocessing, and domain shift simulation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset.colab_datasets import ColabCropDataset, ColabDomainShiftDataset
from src.dataset.colab_dataloader import ColabDataLoader


class TestColabCropDataset:
    """Test Colab-optimized crop dataset."""

    def test_dataset_initialization(self, tmp_path):
        """Test dataset initialization with mock data."""
        # Create mock data directory
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Mock ImageFolder behavior
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['class1', 'class2']
            mock_dataset.__len__.return_value = 100
            mock_imagefolder.return_value = mock_dataset
            
            dataset = ColabCropDataset(data_dir, transform=None)
            
            assert dataset is not None
            assert len(dataset) == 100
            assert dataset.classes == ['class1', 'class2']

    def test_dataset_with_transform(self):
        """Test dataset with transformation."""
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['tomato', 'potato']
            mock_dataset.__len__.return_value = 50
            mock_imagefolder.return_value = mock_dataset
            
            transform = MagicMock()
            dataset = ColabCropDataset(Path('./data'), transform=transform)
            
            assert dataset.transform == transform

    def test_get_item(self):
        """Test __getitem__ method."""
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['class1', 'class2']
            mock_dataset.__len__.return_value = 10
            
            # Mock __getitem__ to return image and label
            mock_dataset.__getitem__.return_value = (torch.randn(3, 224, 224), 1)
            mock_imagefolder.return_value = mock_dataset
            
            dataset = ColabCropDataset(Path('./data'), transform=None)
            image, label = dataset[0]
            
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 224, 224)
            assert label == 1


class TestColabDomainShiftDataset:
    """Test domain shift dataset for Phase 3."""

    def test_domain_shift_initialization(self):
        """Test domain shift dataset initialization."""
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['class1', 'class2']
            mock_dataset.__len__.return_value = 100
            mock_imagefolder.return_value = mock_dataset
            
            dataset = ColabDomainShiftDataset(
                Path('./data'),
                transform=None,
                domain_label=0
            )
            
            assert dataset.domain_label == 0
            assert hasattr(dataset, 'domain_labels')

    def test_domain_label_assignment(self):
        """Test that domain labels are correctly assigned."""
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['class1', 'class2']
            mock_dataset.__len__.return_value = 10
            
            # Mock __getitem__ to return image and label
            def mock_getitem(idx):
                return torch.randn(3, 224, 224), idx % 2
            
            mock_dataset.__getitem__ = mock_getitem
            mock_imagefolder.return_value = mock_dataset
            
            dataset = ColabDomainShiftDataset(
                Path('./data'),
                transform=None,
                domain_label=1
            )
            
            # Check that domain labels are set correctly
            assert hasattr(dataset, 'domain_labels')
            assert len(dataset.domain_labels) == 10
            assert all(label == 1 for label in dataset.domain_labels)


class TestColabDataLoader:
    """Test Colab-optimized data loader."""

    def test_dataloader_creation(self):
        """Test creating a ColabDataLoader."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        
        loader = ColabDataLoader(
            mock_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2
        )
        
        assert loader.batch_size == 16
        assert loader.shuffle is True
        assert loader.num_workers == 2
        assert loader.pin_memory is True

    def test_dataloader_with_mock_data(self):
        """Test data loading with mock data."""
        # Create a properly indexable mock dataset
        class MockDataset:
            def __len__(self):
                return 32
            
            def __getitem__(self, idx):
                return {
                    'images': torch.randn(3, 224, 224),
                    'labels': idx % 5
                }
        
        mock_dataset = MockDataset()
        
        loader = ColabDataLoader(mock_dataset, batch_size=16, num_workers=0)  # Force single-process
        batches = list(loader)
        
        assert len(batches) == 2  # 32 items / 16 batch_size = 2 batches
        assert all('images' in batch and 'labels' in batch for batch in batches)


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""

    def test_end_to_end_data_loading(self, tmp_path):
        """Test complete data loading pipeline."""
        # Create mock data structure
        data_dir = tmp_path / "test_dataset"
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)
        
        # Mock ImageFolder to return simple dataset
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['class0', 'class1', 'class2']
            mock_dataset.__len__.return_value = 30
            
            def mock_getitem(self, idx):
                return torch.randn(3, 224, 224), idx % 3
            
            mock_dataset.__getitem__ = mock_getitem
            mock_imagefolder.return_value = mock_dataset
            
            # Create dataset
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            
            dataset = ColabCropDataset(train_dir, transform=transform)
            
            # Create dataloader
            loader = ColabDataLoader(
                dataset,
                batch_size=5,
                shuffle=False,
                num_workers=0
            )
            
            # Iterate through batches
            batch_count = 0
            for batch in loader:
                # Batch is a list of [images_tensor, labels_tensor] from default collate
                assert len(batch) == 2, f"Expected 2 items in batch, got {len(batch)}"
                images, labels = batch
                assert isinstance(images, torch.Tensor)
                assert isinstance(labels, torch.Tensor)
                assert images.shape[0] <= 5  # batch size
                batch_count += 1
            
            assert batch_count > 0

    def test_domain_shift_pipeline(self):
        """Test domain shift dataset integration."""
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['class0', 'class1']
            mock_dataset.__len__.return_value = 20
            
            def mock_getitem(self, idx):
                return torch.randn(3, 224, 224), idx % 2
            
            mock_dataset.__getitem__ = mock_getitem
            mock_imagefolder.return_value = mock_dataset
            
            # Create domain shift dataset
            dataset = ColabDomainShiftDataset(
                Path('./data'),
                transform=None,
                domain_label=0
            )
            
            # Create dataloader
            loader = ColabDataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            
            # Check that domain information is preserved
            # Note: default_collate will batch dict items
            batch = next(iter(loader))
            assert isinstance(batch, dict), f"Expected dict batch, got {type(batch)}"
            assert 'images' in batch
            assert 'labels' in batch
            assert 'domain' in batch
            assert batch['images'].shape[0] == 4


class TestMemoryOptimization:
    """Test memory optimization features."""

    def test_pin_memory(self):
        """Test that pin_memory is properly set."""
        mock_dataset = MagicMock()
        loader = ColabDataLoader(mock_dataset, pin_memory=True)
        assert loader.pin_memory is True

    def test_num_workers(self):
        """Test that num_workers is properly set."""
        mock_dataset = MagicMock()
        loader = ColabDataLoader(mock_dataset, num_workers=4)
        assert loader.num_workers == 4

    def test_prefetch_factor(self):
        """Test that prefetch_factor is properly set."""
        mock_dataset = MagicMock()
        loader = ColabDataLoader(mock_dataset, prefetch_factor=3)
        assert loader.prefetch_factor == 3


class TestDataAugmentation:
    """Test data augmentation transforms."""

    def test_transform_pipeline(self):
        """Test that transforms are correctly applied."""
        with patch('torchvision.datasets.ImageFolder') as mock_imagefolder:
            mock_dataset = MagicMock()
            mock_dataset.classes = ['class1']
            mock_dataset.__len__.return_value = 1
            
            # Create mock that returns a tensor directly (simulating PIL image after transform)
            def mock_getitem(self, idx):
                # Return tensor directly as if transform already applied
                return torch.randn(3, 300, 300), 0
            
            mock_dataset.__getitem__ = mock_getitem
            mock_imagefolder.return_value = mock_dataset
            
            # Create transform
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ColabCropDataset(Path('./data'), transform=transform)
            image, label = dataset[0]
            
            assert isinstance(image, torch.Tensor)
            assert image.shape == (3, 224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])