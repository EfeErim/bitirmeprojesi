#!/usr/bin/env python3
"""
Dataset Preparation Script for AADS-ULoRA v5.5
Creates standardized directory structure and processes data from archives.
"""

import os
import shutil
import random
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

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

def split_dataset(
    image_list: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split images into train/val/test sets.
    
    Args:
        image_list: List of image paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    # Shuffle images
    random.shuffle(image_list)
    
    # Calculate split indices
    total = len(image_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = image_list[:train_end]
    val_set = image_list[train_end:val_end]
    test_set = image_list[val_end:]
    
    return train_set, val_set, test_set

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
