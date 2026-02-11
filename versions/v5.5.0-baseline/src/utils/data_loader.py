#!/usr/bin/env python3
"""
Data Loading Utilities for AADS-ULoRA v5.5
Provides dataset classes for crop images, domain shift data, and preprocessing.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class CropDataset(Dataset):
    """
    Dataset for crop disease images.
    
    Expected directory structure:
    data/
      {crop}/
        phase1/
          {class1}/
          {class2}/
        val/
        test/
    """
    
    def __init__(
        self,
        data_dir: str,
        crop: str,
        split: str = 'train',
        transform: bool = True,
        target_size: int = 224
    ):
        """
        Initialize crop dataset.
        
        Args:
            data_dir: Base data directory
            crop: Crop name (tomato, pepper, corn)
            split: Split to load ('train', 'val', 'test')
            transform: Whether to apply augmentations (True for train, False for val/test)
            target_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.crop = crop
        self.split = split
        self.target_size = target_size
        
        # Define class mapping based on crop
        self.classes = self._get_crop_classes(crop)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_data()
        
        # Define transforms
        self.transform = self._get_transforms() if transform else self._get_val_transforms()
        
        logger.info(f"Loaded {len(self.image_paths)} images for {crop} {split} split")
        logger.info(f"Classes: {self.classes}")
    
    def _get_crop_classes(self, crop: str) -> List[str]:
        """Return class list for each crop type."""
        crop_classes = {
            'tomato': ['healthy', 'early_blight', 'late_blight', 'septoria_leaf_spot', 'bacterial_spot'],
            'pepper': ['healthy', 'bell_pepper_bacterial_spot'],
            'corn': ['healthy', 'common_rust', 'northern_leaf_blight']
        }
        return crop_classes.get(crop, ['healthy'])
    
    def _load_data(self) -> Tuple[List[Path], List[int]]:
        """Load image paths and labels from directory structure."""
        image_paths = []
        labels = []
        
        # Determine which subdirectory to use based on split
        if self.split == 'train':
            base_dir = self.data_dir / self.crop / 'phase1'
        elif self.split == 'val':
            base_dir = self.data_dir / self.crop / 'val'
        elif self.split == 'test':
            base_dir = self.data_dir / self.crop / 'test'
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        if not base_dir.exists():
            logger.warning(f"Directory not found: {base_dir}")
            return [], []
        
        # Iterate through class directories
        for class_name in self.classes:
            class_dir = base_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Get all image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    image_paths.append(img_path)
                    labels.append(self.class_to_idx[class_name])
        
        return image_paths, labels
    
    def _get_transforms(self) -> transforms.Compose:
        """Get training augmentations."""
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self) -> transforms.Compose:
        """Get validation/test transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder
            return torch.zeros(3, self.target_size, self.target_size), label

class DomainShiftDataset(Dataset):
    """
    Dataset for domain-shifted images used in Phase 3 (CONEC-LoRA).
    Contains images with different lighting, camera angles, or environmental conditions.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: bool = True,
        target_size: int = 224
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_size = target_size
        
        # Load all images from domain_shift directory
        self.image_paths, self.labels = self._load_domain_shift_data()
        
        self.transform = self._get_transforms() if transform else self._get_val_transforms()
        
        logger.info(f"Loaded {len(self.image_paths)} domain-shifted images")
    
    def _load_domain_shift_data(self) -> Tuple[List[Path], List[int]]:
        """Load domain-shifted images."""
        image_paths = []
        labels = []
        
        domain_shift_dir = self.data_dir / 'domain_shift'
        if not domain_shift_dir.exists():
            logger.warning(f"Domain shift directory not found: {domain_shift_dir}")
            return [], []
        
        # Assume same class structure as original data
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        
        for class_dir in domain_shift_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                # Try to map to class index (requires metadata)
                # For now, use 0 as placeholder
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        image_paths.append(img_path)
                        labels.append(0)  # Placeholder
        
        return image_paths, labels
    
    def _get_transforms(self) -> transforms.Compose:
        """Get training augmentations for domain shift."""
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self) -> transforms.Compose:
        """Get validation transforms."""
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, self.target_size, self.target_size), label

def preprocess_image(
    image: Image.Image,
    target_size: int = 224
) -> torch.Tensor:
    """
    Preprocess a single PIL image for model inference.
    
    Args:
        image: Input PIL image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)

def create_data_loaders(
    data_dir: str,
    crop: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = CropDataset(
            data_dir=data_dir,
            crop=crop,
            split=split,
            transform=(split == 'train')
        )
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders

if __name__ == "__main__":
    # Test the data loaders
    logging.basicConfig(level=logging.INFO)
    
    data_dir = "./data"
    crop = "tomato"
    
    try:
        loaders = create_data_loaders(data_dir, crop, batch_size=16)
        for split, loader in loaders.items():
            print(f"{split}: {len(loader.dataset)} samples, {len(loader)} batches")
    except Exception as e:
        print(f"Error: {e}")