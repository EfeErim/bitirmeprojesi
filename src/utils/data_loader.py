#!/usr/bin/env python3
"""
Data Loading Utilities for AADS-ULoRA v5.5
Provides dataset classes for crop images, domain shift data, and preprocessing.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import logging
import time

logger = logging.getLogger(__name__)

class LRUCache:
    """Efficient LRU cache implementation using OrderedDict for O(1) operations."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity: int = capacity
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()  # Maintains insertion order, O(1) move_to_end
        self.timestamps: Dict[str, float] = {}
        self.ttl_seconds: Optional[float] = None

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get value from cache with O(1) access and TTL checking."""
        if key not in self.cache:
            return None

        # Check TTL
        if self.ttl_seconds is not None:
            ts = self.timestamps.get(key, 0)
            if time.time() - ts > self.ttl_seconds:
                # Expired - remove it
                del self.cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]
                return None

        # Move to end (most recently used) - O(1) with OrderedDict
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: torch.Tensor) -> None:
        """Put value in cache with O(1) eviction if needed."""
        if key in self.cache:
            # Update existing and move to end
            self.cache[key] = value
            self.cache.move_to_end(key)
            self.timestamps[key] = time.time()
        else:
            # Add new entry
            if len(self.cache) >= self.capacity:
                # Remove least recently used (first item in OrderedDict)
                lru_key, _ = self.cache.popitem(last=False)
                if lru_key in self.timestamps:
                    del self.timestamps[lru_key]

            self.cache[key] = value
            self.timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()

    def __len__(self) -> int:
        return len(self.cache)

    # Mapping-style access used by unit tests
    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        return self.put(key, value)

    def __getitem__(self, key: str) -> Optional[torch.Tensor]:
        return self.get(key)

    def __delitem__(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]

    def set_ttl(self, seconds: Optional[float]) -> None:
        self.ttl_seconds = seconds

class CropDataset(Dataset):
    """
    Dataset for crop disease images.
    
    Expected directory structure:
    data/
      {crop}/
        continual/
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
        target_size: int = 224,
        use_cache: bool = True,
        cache_size: int = 1000
    ) -> None:
        """
        Initialize crop dataset.
        
        Args:
            data_dir: Base data directory
            crop: Crop name (tomato, pepper, corn)
            split: Split to load ('train', 'val', 'test')
            transform: Whether to apply augmentations (True for train, False for val/test)
            target_size: Target image size
            use_cache: Whether to use image caching
            cache_size: Size of LRU cache
        """
        self.data_dir = Path(data_dir)
        self.crop = crop
        self.split = split
        self.target_size = target_size
        self.use_cache = use_cache
        
        # Define class mapping based on crop
        self.classes = self._get_crop_classes(crop)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load image paths and labels
        self.image_paths, self.labels = self._load_data()
        
        # Define transforms
        self.transform = self._get_transforms() if transform else self._get_val_transforms()
        
        # Initialize cache
        self.cache = LRUCache(cache_size) if use_cache else None
        
        logger.info(f"Loaded {len(self.image_paths)} images for {crop} {split} split")
        logger.info(f"Classes: {self.classes}")
        if use_cache:
            logger.info(f"Image caching enabled with capacity {cache_size}")
    
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
            base_dir = self.data_dir / self.crop / 'continual'
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
    
    def _load_image_cv2(self, img_path: Path) -> np.ndarray:
        """Load image using OpenCV (faster than PIL)."""
        # Read image with OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
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
            # Check cache first (for raw images only, to preserve augmentation diversity)
            if self.use_cache and self.split != 'train':
                cache_key = str(img_path)
                cached = self.cache.get(cache_key)
                if cached is not None:
                    image = cached
                else:
                    # Load and cache raw image (for val/test only)
                    image = self._load_image_cv2(img_path)
                    self.cache.put(cache_key, image)
            else:
                # For training, always load raw without caching to maintain augmentation diversity
                image = self._load_image_cv2(img_path)

            # Convert to PIL for torchvision transforms
            image = transforms.ToPILImage()(image)

            # Apply transforms (always fresh for training)
            image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a placeholder
            return torch.zeros(3, self.target_size, self.target_size), label
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self.use_cache and self.cache:
            return {
                'cache_size': len(self.cache),
                'cache_capacity': self.cache.capacity
            }
        return {'cache_size': 0, 'cache_capacity': 0}

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
        target_size: int = 224,
        use_cache: bool = True,
        cache_size: int = 1000
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_size = target_size
        self.use_cache = use_cache
        
        # Load all images from domain_shift directory
        self.image_paths, self.labels = self._load_domain_shift_data()
        
        self.transform = self._get_transforms() if transform else self._get_val_transforms()
        
        # Initialize cache
        self.cache = LRUCache(cache_size) if use_cache else None
        
        logger.info(f"Loaded {len(self.image_paths)} domain-shifted images")
        if use_cache:
            logger.info(f"Image caching enabled with capacity {cache_size}")
    
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
    
    def _load_image_cv2(self, img_path: Path) -> np.ndarray:
        """Load image using OpenCV."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
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
            # Check cache first (for raw images only, to preserve augmentation diversity)
            if self.use_cache and self.split != 'train':
                cache_key = str(img_path)
                cached = self.cache.get(cache_key)
                if cached is not None:
                    image = cached
                else:
                    # Load and cache raw image (for val/test only)
                    image = self._load_image_cv2(img_path)
                    self.cache.put(cache_key, image)
            else:
                # For training, always load raw without caching to maintain augmentation diversity
                image = self._load_image_cv2(img_path)

            # Convert to PIL for torchvision transforms
            image = transforms.ToPILImage()(image)

            # Apply transforms (always fresh for training)
            image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, self.target_size, self.target_size), label
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self.use_cache and self.cache:
            return {
                'cache_size': len(self.cache),
                'cache_capacity': self.cache.capacity
            }
        return {'cache_size': 0, 'cache_capacity': 0}

def preprocess_image(
    image: Union[np.ndarray, Image.Image],
    target_size: int = 224
) -> torch.Tensor:
    """
    Preprocess a single image for model inference.
    
    Args:
        image: Input image (OpenCV numpy array or PIL Image)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed tensor
        
    Raises:
        ValueError: If image format is invalid
    """
    """
    Preprocess a single image for model inference.
    
    Args:
        image: Input image (OpenCV numpy array or PIL Image)
        target_size: Target size for resizing
        
    Returns:
        Preprocessed tensor
        
    Raises:
        ValueError: If image format is invalid
    """
    # Validate and convert input
    if isinstance(image, np.ndarray):
        # Validate numpy array dimensions
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")
        
        # Handle grayscale (2D) images
        if image.ndim == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            # Handle different channel configurations
            if image.shape[2] == 3:
                # BGR to RGB (OpenCV default)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                # BGRA to RGB (with alpha channel)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 1:
                # Single channel grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"Invalid number of channels: {image.shape[2]}, expected 1, 3, or 4")
        
        # Convert numpy to PIL
        image = transforms.ToPILImage()(image)
    
    # Ensure PIL Image
    if not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)

def create_data_loaders(
    data_dir: str,
    crop: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_cache: bool = True,
    cache_size: int = 1000
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Base data directory
        crop: Crop type
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_cache: Whether to enable image caching
        cache_size: Size of LRU cache per dataset
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = CropDataset(
            data_dir=data_dir,
            crop=crop,
            split=split,
            transform=(split == 'train'),
            use_cache=use_cache,
            cache_size=cache_size
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
            # Print cache stats
            if hasattr(loader.dataset, 'get_cache_stats'):
                stats = loader.dataset.get_cache_stats()
                print(f"  Cache: {stats}")
    except Exception as e:
        print(f"Error: {e}")
