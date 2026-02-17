#!/usr/bin/env python3
"""
Tests for dataset preparation module.
"""

import pytest
import torch
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_dataset_factory, mock_tensor_factory
from src.dataset.preparation import (
    DatasetPreparer,
    DatasetConfig,
    prepare_dataset,
    split_dataset,
    balance_dataset,
    augment_dataset
)


class TestDatasetPreparer:
    """Test dataset preparation functionality."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory for test dataset."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config(self):
        """Create sample dataset configuration."""
        return DatasetConfig(
            name="test_dataset",
            classes=["healthy", "disease1", "disease2"],
            image_size=(224, 224),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )

    def test_dataset_preparer_initialization(self, sample_config):
        """Test DatasetPreparer initialization."""
        preparer = DatasetPreparer(config=sample_config)
        assert preparer.config == sample_config
        assert preparer.config.name == "test_dataset"
        assert len(preparer.config.classes) == 3

    def test_prepare_dataset_creates_structure(self, temp_dataset_dir, sample_config):
        """Test that prepare_dataset creates proper directory structure."""
        preparer = DatasetPreparer(config=sample_config, output_dir=temp_dataset_dir)
        preparer.prepare_dataset()

        # Check that directories were created
        assert (temp_dataset_dir / "train").exists()
        assert (temp_dataset_dir / "val").exists()
        assert (temp_dataset_dir / "test").exists()

        # Check class subdirectories
        for split in ["train", "val", "test"]:
            split_dir = temp_dataset_dir / split
            for class_name in sample_config.classes:
                assert (split_dir / class_name).exists()

    def test_split_dataset_proportions(self, temp_dataset_dir, sample_config):
        """Test that dataset splits have correct proportions."""
        preparer = DatasetPreparer(config=sample_config, output_dir=temp_dataset_dir)
        splits = preparer.split_dataset()

        total_samples = sum(len(samples) for samples in splits.values())
        assert total_samples > 0

        # Check proportions are approximately correct
        train_ratio = len(splits["train"]) / total_samples
        val_ratio = len(splits["val"]) / total_samples
        test_ratio = len(splits["test"]) / total_samples

        assert abs(train_ratio - sample_config.train_split) < 0.05
        assert abs(val_ratio - sample_config.val_split) < 0.05
        assert abs(test_ratio - sample_config.test_split) < 0.05

    def test_balance_dataset(self):
        """Test dataset balancing."""
        # Create imbalanced dataset
        class_counts = {"class_a": 100, "class_b": 50, "class_c": 25}
        balanced = balance_dataset(class_counts, strategy="oversample")

        # All classes should have same count after balancing
        counts = list(balanced.values())
        assert len(set(counts)) == 1  # All equal

    def test_augment_dataset(self, temp_dataset_dir):
        """Test dataset augmentation."""
        # Create sample image file
        from PIL import Image
        import numpy as np

        img_dir = temp_dataset_dir / "train" / "class_a"
        img_dir.mkdir(parents=True)

        # Create a test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(img_dir / "test.jpg")

        # Augment
        augmented = augment_dataset(
            str(temp_dataset_dir),
            augmentation_factor=2,
            augmentations=["rotate", "flip", "color"]
        )

        # Should have created additional augmented images
        assert augmented is not None

    def test_dataset_config_validation(self):
        """Test dataset configuration validation."""
        # Valid config
        config = DatasetConfig(
            name="valid",
            classes=["a", "b"],
            image_size=(224, 224),
            train_split=0.7,
            val_split=0.2,
            test_split=0.1
        )
        assert config.train_split + config.val_split + config.test_split == 1.0

        # Invalid splits should raise error
        with pytest.raises(ValueError):
            DatasetConfig(
                name="invalid",
                classes=["a"],
                image_size=(224, 224),
                train_split=0.8,
                val_split=0.3,  # Sum > 1
                test_split=0.1
            )

    def test_prepare_dataset_function(self, temp_dataset_dir):
        """Test the convenience function prepare_dataset."""
        config = DatasetConfig(
            name="test",
            classes=["c1", "c2"],
            image_size=(128, 128),
            train_split=0.6,
            val_split=0.2,
            test_split=0.2
        )

        result = prepare_dataset(
            config=config,
            source_dir=temp_dataset_dir,
            output_dir=temp_dataset_dir / "prepared"
        )

        assert result is not None
        assert (temp_dataset_dir / "prepared").exists()


class TestDatasetUtils:
    """Test dataset utility functions."""

    def test_calculate_class_weights(self):
        """Test class weight calculation for imbalanced datasets."""
        class_counts = {"class_a": 100, "class_b": 50, "class_c": 25}
        weights = DatasetPreparer.calculate_class_weights(class_counts)

        assert "class_a" in weights
        assert "class_b" in weights
        assert "class_c" in weights
        # Class with fewer samples should have higher weight
        assert weights["class_c"] > weights["class_a"]

    def test_validate_image_files(self, temp_dataset_dir):
        """Test image file validation."""
        from PIL import Image
        import numpy as np

        # Create valid image
        img_dir = temp_dataset_dir / "valid"
        img_dir.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(img_dir / "valid.jpg")

        # Create invalid file
        invalid_file = temp_dataset_dir / "invalid.txt"
        invalid_file.write_text("not an image")

        preparer = DatasetPreparer()
        valid_files = preparer.validate_image_files(str(temp_dataset_dir))

        # Should find only the valid image
        assert len(valid_files) == 1
        assert valid_files[0].suffix == ".jpg"

    def test_get_dataset_stats(self, temp_dataset_dir):
        """Test dataset statistics calculation."""
        # Create sample dataset structure
        train_dir = temp_dataset_dir / "train"
        train_dir.mkdir()

        for class_name in ["class1", "class2"]:
            class_dir = train_dir / class_name
            class_dir.mkdir()
            # Create dummy image files
            for i in range(10):
                (class_dir / f"img_{i}.jpg").write_bytes(b"fake image data")

        preparer = DatasetPreparer()
        stats = preparer.get_dataset_stats(str(temp_dataset_dir))

        assert "total_images" in stats
        assert "class_distribution" in stats
        assert stats["total_images"] == 20
        assert len(stats["class_distribution"]) == 2
