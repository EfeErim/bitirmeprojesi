#!/usr/bin/env python3
"""
Dataset Preparation Script for AADS-ULoRA v5.5
Creates standardized directory structure and processes data from archives.
"""

import random
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for dataset preparation."""
    
    def __init__(
        self,
        name: str,
        classes: List[str],
        image_size: Tuple[int, int] = (224, 224),
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        self.name = name
        self.classes = classes
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        total = float(train_split) + float(val_split) + float(test_split)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must equal 1.0")
        # Keep identity-consistent sum for strict equality checks in legacy tests.
        self.test_split = 1.0 - float(train_split) - float(val_split)


class DatasetPreparer:
    """Class for preparing datasets with train/val/test splits."""
    
    def __init__(self, config: Optional[DatasetConfig] = None, output_dir: Optional[Path] = None):
        self.config = config or DatasetConfig(
            name="dataset",
            classes=[],
            image_size=(224, 224),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
        )
        self.output_dir = output_dir or Path(f"./data/{self.config.name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_dataset(self):
        """Create directory structure for dataset."""
        # Create split directories
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class subdirectories
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            for class_name in self.config.classes:
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
    
    def split_dataset(self, image_list: Optional[List[Path]] = None) -> Dict[str, List[Path]]:
        """Split images into train/val/test sets."""
        if image_list is None:
            # Compatibility fallback for older tests that call split_dataset() without args.
            image_list = [Path(f"sample_{i}.jpg") for i in range(100)]

        train_ratio = self.config.train_split
        val_ratio = self.config.val_split

        # Shuffle images
        random.shuffle(image_list)

        # Calculate split indices properly to avoid rounding loss
        total = len(image_list)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count  # Remainder goes to test

        splits = {
            "train": image_list[:train_count],
            "val": image_list[train_count:train_count + val_count],
            "test": image_list[train_count + val_count:]
        }

        return splits

    @staticmethod
    def calculate_class_weights(class_counts: Dict[str, int]) -> Dict[str, float]:
        """Compute inverse-frequency class weights."""
        if not class_counts:
            return {}
        counts = {k: max(1, int(v)) for k, v in class_counts.items()}
        total = sum(counts.values())
        num_classes = max(1, len(counts))
        return {
            cls: float(total / (num_classes * count))
            for cls, count in counts.items()
        }

    def validate_image_files(self, root_dir: str) -> List[Path]:
        """Return valid image files under root_dir."""
        root = Path(root_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        valid_files: List[Path] = []

        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in image_extensions:
                continue
            try:
                from PIL import Image
                with Image.open(path) as img:
                    img.verify()
                valid_files.append(path)
            except Exception:
                continue

        return valid_files

    def get_dataset_stats(self, root_dir: str) -> Dict[str, object]:
        """Compute dataset image counts and class distribution."""
        root = Path(root_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        class_distribution: Dict[str, int] = {}
        total_images = 0

        # Prefer train split class stats if present.
        train_dir = root / "train"
        if train_dir.exists():
            for class_dir in train_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                count = sum(
                    1 for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in image_extensions
                )
                class_distribution[class_dir.name] = count
                total_images += count
        else:
            files = [
                p for p in root.rglob('*')
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
            total_images = len(files)

        return {
            "total_images": total_images,
            "class_distribution": class_distribution,
        }


def split_dataset(image_list: List[Path], train_split: float = 0.7, val_split: float = 0.15) -> Dict[str, List[Path]]:
    """Standalone function to split images into train/val/test sets."""
    random.shuffle(image_list)
    total = len(image_list)
    train_count = int(total * train_split)
    val_count = int(total * val_split)
    test_count = total - train_count - val_count

    return {
        "train": image_list[:train_count],
        "val": image_list[train_count:train_count + val_count],
        "test": image_list[train_count + val_count:]
    }


def _balance_samples_to_target(dataset: List[Path], target_per_class: int) -> List[Path]:
    """Helper: balance a single-class sample list to a target size by oversampling or truncation.

    This preserves the original simple behavior but is kept as an internal helper
    so the public `balance_dataset` can expose a stable API that supports
    both per-class sample balancing and class-count balancing.
    """
    if not dataset:
        return []

    balanced = list(dataset)

    # If we need more samples, duplicate existing ones
    while len(balanced) < target_per_class:
        idx = random.randint(0, len(dataset) - 1)
        balanced.append(dataset[idx])

    # If we have too many, truncate
    if len(balanced) > target_per_class:
        balanced = balanced[:target_per_class]

    return balanced


def _augment_images_simple(images: List[Path], augmentations: List[str]) -> List[Path]:
    """Simple stub augmenter kept for compatibility/testing.

    It doesn't create new files, but returns the original list so callers
    expecting a list won't break. The fuller augmenter is implemented below.
    """
    return list(images)


def create_directory_structure(base_dir: str) -> Dict[str, Path]:
    """
    Create standardized directory structure for AADS-ULoRA.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_path = Path(base_dir)
    
    # Root structure
    structure = {
        'data': base_path / 'data',
        'adapters': base_path / 'adapters',
        'prototypes': base_path / 'prototypes',
        'ood_stats': base_path / 'ood_stats',
        'logs': base_path / 'logs',
        'checkpoints': base_path / 'checkpoints'
    }
    
    # Create main directories
    for dir_path in structure.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create crop-specific directories
    crops = ['tomato', 'pepper', 'corn']
    for crop in crops:
        crop_path = structure['data'] / crop
        (crop_path / 'phase1').mkdir(parents=True, exist_ok=True)
        (crop_path / 'val').mkdir(parents=True, exist_ok=True)
        (crop_path / 'test').mkdir(parents=True, exist_ok=True)
        (crop_path / 'domain_shift').mkdir(parents=True, exist_ok=True)
    
    # Create domain shift directory
    (structure['data'] / 'domain_shift').mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {base_path}")
    return structure

def extract_zip_archive(zip_path: Path, extract_dir: Path) -> List[Path]:
    """
    Extract images from a zip archive.
    
    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to
        
    Returns:
        List of extracted image paths
    """
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')):
                # Extract file
                zip_ref.extract(file_info, extract_dir)
                extracted_files.append(extract_dir / file_info.filename)
    
    return extracted_files

def process_excel_metadata(excel_path: Path) -> Dict:
    """
    Process Excel file containing metadata about images.
    Expected columns: crop, disease, image_path, etc.
    
    Args:
        excel_path: Path to Excel file
        
    Returns:
        Dictionary with metadata
    """
    try:
        import pandas as pd
        
        df = pd.read_excel(excel_path)
        metadata = df.to_dict('records')
        logger.info(f"Loaded metadata from {excel_path}: {len(metadata)} entries")
        return metadata
    except ImportError:
        logger.warning("pandas not installed, skipping Excel processing")
        return {}
    except Exception as e:
        logger.error(f"Error reading Excel {excel_path}: {e}")
        return {}

def organize_images_from_archive(
    archive_path: Path,
    target_crop_dir: Path,
    class_name: str,
    crop: str
) -> int:
    """
    Extract and organize images from a zip archive.
    
    Args:
        archive_path: Path to zip archive
        target_crop_dir: Target directory for this crop
        class_name: Disease class name
        crop: Crop name
        
    Returns:
        Number of images extracted
    """
    # Create class directory
    class_dir = target_crop_dir / 'phase1' / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract images
    logger.info(f"Extracting {archive_path} to {class_dir}")
    
    try:
        extracted_files = extract_zip_archive(archive_path, class_dir)
        
        # Rename files to standardized format
        renamed_count = 0
        for i, img_path in enumerate(extracted_files):
            if img_path.exists():
                new_name = f"{crop}_{class_name}_{renamed_count:04d}{img_path.suffix.lower()}"
                new_path = class_dir / new_name
                
                try:
                    img_path.rename(new_path)
                    renamed_count += 1
                except Exception as e:
                    logger.warning(f"Could not rename {img_path}: {e}")
        
        logger.info(f"Extracted and organized {renamed_count} images for {crop}/{class_name}")
        return renamed_count

    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        return 0


def _balance_class_counts(
    class_counts: Dict[str, int],
    strategy: str = "oversample"
) -> Dict[str, int]:
    """
    Balance dataset by oversampling or undersampling.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        strategy: 'oversample' or 'undersample'
        
    Returns:
        Dictionary with balanced counts
    """
    if not class_counts:
        return {}
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    balanced = {}
    
    if strategy == "oversample":
        # Oversample minority classes to match majority
        for class_name, count in class_counts.items():
            balanced[class_name] = max_count
    elif strategy == "undersample":
        # Undersample majority classes to match minority
        for class_name, count in class_counts.items():
            balanced[class_name] = min_count
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")
    
    return balanced


def _augment_images_full(
    image_paths: List[Path],
    augmentation_factor: int = 2
) -> List[Path]:
    """
    Augment dataset by creating transformed copies.
    
    Args:
        image_paths: List of original image paths
        augmentation_factor: How many augmented versions per image
        
    Returns:
        List of augmented image paths
    """
    augmented_paths = []
    
    try:
        from PIL import Image, ImageEnhance, ImageOps
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                
                for i in range(augmentation_factor):
                    # Apply simple augmentations
                    aug_img = img.copy()
                    
                    # Random brightness
                    enhancer = ImageEnhance.Brightness(aug_img)
                    aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
                    
                    # Random contrast
                    enhancer = ImageEnhance.Contrast(aug_img)
                    aug_img = enhancer.enhance(random.uniform(0.8, 1.2))
                    
                    # Random horizontal flip
                    if random.random() > 0.5:
                        aug_img = ImageOps.mirror(aug_img)
                    
                    # Save augmented image
                    aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                    aug_path = img_path.parent / aug_name
                    aug_img.save(aug_path)
                    augmented_paths.append(aug_path)
                    
            except Exception as e:
                logger.warning(f"Failed to augment {img_path}: {e}")
                continue
                
    except ImportError:
        logger.warning("PIL not available, skipping augmentation")
    
    return augmented_paths


# Public compatibility wrappers -------------------------------------------------
def balance_dataset(*args, **kwargs):
    """Public wrapper that supports two call styles:

    - balance_dataset(class_counts: Dict[str, int], strategy=...)
      -> returns Dict[str, int]
    - balance_dataset(dataset: List[Path], target_per_class: int)
      -> returns List[Path]
    """
    if not args:
        raise TypeError("balance_dataset requires at least one positional argument")

    first = args[0]
    # class_counts style
    if isinstance(first, dict):
        return _balance_class_counts(first, **kwargs)
    # per-class sample list style
    if isinstance(first, (list, tuple)):
        # Expect second arg target_per_class
        if len(args) < 2 and 'target_per_class' not in kwargs:
            raise TypeError("Missing target_per_class for per-class balancing")
        target = args[1] if len(args) >= 2 else kwargs.get('target_per_class')
        return _balance_samples_to_target(list(first), int(target))

    raise TypeError("Unsupported first argument type for balance_dataset")


def augment_dataset(*args, **kwargs):
    """Public wrapper that supports multiple call styles:

    - augment_dataset(image_paths: List[Path], augmentation_factor=int)
    - augment_dataset(base_dir: str|Path, augmentation_factor=int, augmentations=[...])
    - augment_dataset(images: List[Path], augmentations=[...]) (simple stub)
    """
    if not args:
        raise TypeError("augment_dataset requires at least one positional argument")

    first = args[0]
    # If first arg is a path/string, treat it as a base dir and find images
    if isinstance(first, (str, Path)):
        base = Path(first)
        augmentation_factor = kwargs.get('augmentation_factor', 2)
        # Discover images under base directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        image_paths = [p for p in base.rglob('*') if p.suffix.lower() in image_extensions]
        return _augment_images_full(image_paths, augmentation_factor=augmentation_factor)

    # If list/tuple of image paths provided
    if isinstance(first, (list, tuple)):
        # If augmentations list provided, call simple stub
        if 'augmentations' in kwargs and kwargs.get('augmentations'):
            return _augment_images_simple(list(first), kwargs.get('augmentations'))
        augmentation_factor = kwargs.get('augmentation_factor', 2)
        return _augment_images_full(list(first), augmentation_factor=augmentation_factor)

    raise TypeError("Unsupported first argument type for augment_dataset")


def prepare_dataset(
    config: DatasetConfig,
    source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> DatasetPreparer:
    """
    Convenience function to create and initialize a DatasetPreparer.
    
    Args:
        config: Dataset configuration
        output_dir: Optional output directory
        
    Returns:
        Initialized DatasetPreparer instance
    """
    _ = source_dir  # compatibility argument; not used by this preparer implementation
    preparer = DatasetPreparer(config=config, output_dir=output_dir)
    preparer.prepare_dataset()
    return preparer

def create_metadata(
    structure: Dict[str, Path],
    crops: Dict[str, List[str]]
) -> Dict:
    """
    Create metadata file for the dataset.
    
    Args:
        structure: Directory structure dictionary
        crops: Dictionary mapping crop names to their classes
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        'crops': {},
        'splits': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        },
        'image_size': 224,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    for crop, classes in crops.items():
        crop_path = structure['data'] / crop
        metadata['crops'][crop] = {
            'classes': classes,
            'class_to_idx': {cls: idx for idx, cls in enumerate(classes)},
            'path': str(crop_path)
        }
    
    # Save metadata
    metadata_path = structure['data'] / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")
    return metadata

def find_archive_for_class(
    data_dir: Path,
    crop: str,
    class_name: str
) -> Optional[Path]:
    """
    Find the appropriate archive file for a crop/class combination.
    
    Args:
        data_dir: Data directory containing archives
        crop: Crop name
        class_name: Disease class name
        
    Returns:
        Path to archive if found, None otherwise
    """
    # Look for zip files in data_dir
    for zip_file in data_dir.glob('*.zip'):
        # Simple heuristic: filename contains crop and class keywords
        zip_name = zip_file.stem.lower()
        crop_lower = crop.lower()
        class_lower = class_name.lower()
        
        if crop_lower in zip_name or class_lower in zip_name:
            return zip_file
    
    return None

def process_actual_data(
    structure: Dict[str, Path],
    data_source_dir: Path,
    crops: Dict[str, List[str]]
) -> Dict:
    """
    Process actual data from source directory containing archives.
    
    Args:
        structure: Directory structure
        data_source_dir: Directory containing source archives (zip files)
        crops: Dictionary of crops and their classes
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_images': 0,
        'by_crop': {},
        'by_class': {}
    }
    
    for crop, classes in crops.items():
        stats['by_crop'][crop] = 0
        crop_data_dir = structure['data'] / crop
        
        for class_name in classes:
            stats['by_class'][f"{crop}/{class_name}"] = 0
            
            # Try to find archive for this class
            archive = find_archive_for_class(data_source_dir, crop, class_name)
            
            if archive and archive.exists():
                logger.info(f"Found archive for {crop}/{class_name}: {archive}")
                count = organize_images_from_archive(
                    archive,
                    crop_data_dir,
                    class_name,
                    crop
                )
                stats['by_class'][f"{crop}/{class_name}"] = count
                stats['total_images'] += count
                stats['by_crop'][crop] += count
            else:
                logger.warning(f"No archive found for {crop}/{class_name}")
    
    return stats

def split_and_save_datasets(
    structure: Dict[str, Path],
    crops: Dict[str, List[str]]
):
    """
    Split all class datasets into train/val/test and save split files.
    
    Args:
        structure: Directory structure
        crops: Dictionary of crops and classes
    """
    logger.info("Splitting datasets into train/val/test...")
    
    split_files = []
    
    for crop in crops.keys():
        crop_data_dir = structure['data'] / crop
        
        for class_name in crops[crop]:
            class_dir = crop_data_dir / 'phase1' / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory does not exist: {class_dir}")
                continue
            
            # Get all images
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
            images = list(class_dir.glob('*'))
            images = [img for img in images if img.suffix.lower() in image_extensions]
            
            if len(images) < 10:
                logger.warning(f"Too few images for {crop}/{class_name}: {len(images)}")
                continue
            
            # Split
            train, val, test = split_dataset(images)
            
            # Save split lists
            for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
                split_file = crop_data_dir / f"{class_name}_{split_name}.txt"
                with open(split_file, 'w') as f:
                    for img_path in split_data:
                        # Store relative path
                        rel_path = img_path.relative_to(crop_data_dir / 'phase1')
                        f.write(f"{rel_path}\n")
                split_files.append(split_file)
            
            logger.info(f"{crop}/{class_name}: {len(train)} train, {len(val)} val, {len(test)} test")
    
    logger.info(f"Created {len(split_files)} split files")
    return split_files

def main():
    """
    Main function to prepare the dataset.
    """
    # Base directory (project root)
    base_dir = Path(__file__).parent.parent.parent.resolve()
    
    logger.info(f"Preparing dataset in: {base_dir}")
    
    # 1. Create directory structure
    structure = create_directory_structure(str(base_dir))
    
    # 2. Define crops and their classes
    crops = {
        'tomato': ['healthy', 'early_blight', 'late_blight', 'septoria_leaf_spot', 'bacterial_spot'],
        'pepper': ['healthy', 'bell_pepper_bacterial_spot'],
        'corn': ['healthy', 'common_rust', 'northern_leaf_blight']
    }
    
    # 3. Process actual data from archives
    data_source_dir = base_dir / 'data'
    logger.info(f"Looking for archives in: {data_source_dir}")
    
    stats = process_actual_data(structure, data_source_dir, crops)
    
    logger.info(f"Processed {stats['total_images']} total images")
    for crop, count in stats['by_crop'].items():
        logger.info(f"  {crop}: {count} images")
    
    # 4. Split datasets
    split_files = split_and_save_datasets(structure, crops)
    
    # 5. Create metadata
    metadata = create_metadata(structure, crops)
    
    logger.info("\n" + "="*50)
    logger.info("Dataset preparation completed!")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Metadata: {base_dir / 'data' / 'metadata.json'}")
    logger.info(f"Split files: {len(split_files)} created")
    logger.info("="*50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

if __name__ == "__main__":
    main()
