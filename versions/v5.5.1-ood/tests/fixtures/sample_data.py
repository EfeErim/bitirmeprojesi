#!/usr/bin/env python3
"""
Test fixtures and sample data generation for AADS-ULoRA tests
"""

import torch
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple

def create_dummy_dataset(
    base_dir: Path,
    crop: str,
    classes: List[str],
    images_per_class: int = 10
) -> Path:
    """
    Create a dummy dataset for testing.
    
    Args:
        base_dir: Base directory to create dataset in
        crop: Crop name
        classes: List of class names
        images_per_class: Number of dummy images per class
        
    Returns:
        Path to created dataset
    """
    data_dir = base_dir / "data" / crop / "phase1"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    for class_name in classes:
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create dummy images
        for i in range(images_per_class):
            # Create random noise image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"dummy_{i:03d}.jpg")
    
    return base_dir / "data"

def create_dummy_adapter_checkpoint(
    save_dir: Path,
    num_classes: int = 2,
    feature_dim: int = 768
) -> Path:
    """
    Create a dummy adapter checkpoint for testing.
    
    Args:
        save_dir: Directory to save checkpoint
        num_classes: Number of classes
        feature_dim: Feature dimension
        
    Returns:
        Path to checkpoint
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy classifier
    classifier = torch.nn.Linear(feature_dim, num_classes)
    torch.save(classifier.state_dict(), save_dir / "classifier.pth")
    
    # Create dummy adapter metadata
    metadata = {
        "adapter_id": "test_adapter",
        "crop": "test_crop",
        "num_classes": num_classes,
        "class_to_idx": {f"class_{i}": i for i in range(num_classes)}
    }
    
    import json
    with open(save_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    return save_dir

def create_dummy_prototypes(
    num_classes: int = 2,
    feature_dim: int = 768
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Create dummy prototypes and stds for testing.
    
    Returns:
        (prototypes, class_stds)
    """
    prototypes = torch.randn(num_classes, feature_dim)
    class_stds = {
        i: torch.rand(feature_dim) * 0.1 + 0.05
        for i in range(num_classes)
    }
    
    return prototypes, class_stds

def create_dummy_ood_scores(
    num_in_dist: int = 50,
    num_ood: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dummy OOD scores for testing.
    
    Returns:
        (in_dist_scores, ood_scores)
    """
    np.random.seed(seed)
    
    # In-distribution: lower scores (mean=10, std=2)
    in_dist = np.random.normal(10, 2, num_in_dist)
    
    # OOD: higher scores (mean=25, std=5)
    ood = np.random.normal(25, 5, num_ood)
    
    return in_dist, ood

def get_test_image_tensor() -> torch.Tensor:
    """
    Get a dummy image tensor for testing.
    
    Returns:
        Tensor of shape (1, 3, 224, 224)
    """
    return torch.randn(1, 3, 224, 224)

def create_test_config() -> dict:
    """
    Create a minimal test configuration.
    
    Returns:
        Configuration dictionary
    """
    return {
        "adapter_id": "test_v55",
        "architecture": "independent_multicrop_dynamic_ood",
        "crops": ["tomato", "pepper"],
        "per_crop": {
            "model_name": "facebook/dinov2-giant",
            "use_dora": True,
            "lora_r": 8,  # Small for testing
            "lora_alpha": 8,
            "loraplus_lr_ratio": 16,
            "phase1_epochs": 1,
            "phase2_epochs": 1,
            "phase3_epochs": 1
        },
        "ood_detection": {
            "threshold_factor": 2.0,
            "min_val_samples_per_class": 3,
            "fallback_threshold": 25.0
        },
        "data": {
            "image_size": 224,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }

if __name__ == "__main__":
    # Quick test
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        
        print("Creating dummy dataset...")
        data_dir = create_dummy_dataset(base, "tomato", ["healthy", "disease"], images_per_class=5)
        print(f"Dataset created at: {data_dir}")
        
        print("Creating dummy adapter...")
        adapter_dir = create_dummy_adapter_checkpoint(base / "adapter", num_classes=2)
        print(f"Adapter created at: {adapter_dir}")
        
        print("Creating dummy prototypes...")
        prototypes, stds = create_dummy_prototypes(2, 768)
        print(f"Prototypes shape: {prototypes.shape}")
        print(f"Stds: {len(stds)} classes")
        
        print("Creating OOD scores...")
        in_dist, ood = create_dummy_ood_scores()
        print(f"In-dist: {len(in_dist)}, OOD: {len(ood)}")
        
        print("All fixtures created successfully!")