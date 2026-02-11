#!/usr/bin/env python3
"""
Download and organize the Tomato Disease dataset from Kaggle.
This script is designed to run in Google Colab with A100 GPU.
"""

import os
import shutil
import random
from pathlib import Path
import subprocess
import sys


def install_kaggle():
    """Install Kaggle API if not already installed."""
    print("Installing Kaggle API...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    print("✅ Kaggle API installed")


def setup_kaggle_credentials():
    """
    Setup Kaggle API credentials.
    In Colab, you need to upload kaggle.json manually.
    """
    from google.colab import files
    import json

    print("\n" + "="*60)
    print("Kaggle API Setup Required")
    print("="*60)
    print("\nTo download the dataset, you need a Kaggle API token.")
    print("\nSteps:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. A file named 'kaggle.json' will be downloaded")
    print("5. Upload that file now when prompted")
    print("\n" + "="*60)

    uploaded = files.upload()

    if 'kaggle.json' not in uploaded:
        raise FileNotFoundError("kaggle.json not uploaded. Please upload your Kaggle API credentials.")

    # Move to .kaggle directory
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / 'kaggle.json'
    shutil.move('kaggle.json', kaggle_json)
    os.chmod(kaggle_json, 0o600)

    print("✅ Kaggle API credentials configured")
    return True


def download_dataset():
    """Download the tomato disease dataset from Kaggle."""
    print("\nDownloading Tomato Disease dataset...")
    print("Dataset: cookiefinder/tomato-disease-multiple-sources")
    print("This may take several minutes depending on size...\n")

    # Create data directory
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    try:
        # Download and unzip
        subprocess.check_call([
            'kaggle', 'datasets', 'download',
            '-d', 'cookiefinder/tomato-disease-multiple-sources',
            '-p', str(data_dir),
            '--unzip'
        ])
        print("✅ Dataset downloaded and extracted successfully!")
        return data_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\nPossible reasons:")
        print("1. Kaggle API not authenticated (run setup_kaggle_credentials first)")
        print("2. Dataset not accessible")
        print("3. Network issues")
        raise


def detect_dataset_structure(data_dir: Path):
    """
    Detect the structure of the downloaded dataset.
    Returns: (train_dir, val_dir, test_dir, class_names)
    """
    print("\nDetecting dataset structure...")

    # Common directory names
    possible_train = ['train', 'Train', 'TRAIN', 'training', 'Training']
    possible_val = ['val', 'valid', 'validation', 'Val', 'Valid', 'Validation']
    possible_test = ['test', 'Test', 'TEST', 'testing', 'Testing']

    train_dir = None
    val_dir = None
    test_dir = None

    # Search for directories
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue

        name_lower = item.name.lower()
        if any(t.lower() in name_lower for t in possible_train):
            train_dir = item
        elif any(v.lower() in name_lower for v in possible_val):
            val_dir = item
        elif any(ts.lower() in name_lower for ts in possible_test):
            test_dir = item

    # If no explicit train/val/test, assume the main directory contains class folders
    if train_dir is None:
        # Check if data_dir directly contains class folders
        class_folders = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        # If there are 3-15 folders, likely class folders
        if 3 <= len(class_folders) <= 15:
            train_dir = data_dir
            print("  → No explicit train/ directory found. Using root as training data.")
        else:
            raise ValueError(f"Could not detect dataset structure. Folders found: {[d.name for d in data_dir.iterdir() if d.is_dir()]}")

    # Get class names from train directory
    class_folders = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    class_names = sorted([d.name for d in class_folders])

    print(f"  Train directory: {train_dir}")
    print(f"  Val directory: {val_dir if val_dir else 'Not found (will split from train)'}")
    print(f"  Test directory: {test_dir if test_dir else 'Not found (will split from train)'}")
    print(f"  Detected classes: {class_names}")

    return train_dir, val_dir, test_dir, class_names


def organize_dataset(data_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path, class_names: list):
    """
    Organize dataset into AADS-ULoRA expected structure:

    data/tomato/
    ├── phase1/  (training data)
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── val/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/
        ├── class1/
        ├── class2/
        └── ...
    """
    print("\nOrganizing dataset into AADS-ULoRA structure...")

    tomato_dir = data_dir / 'tomato'
    phase1_dir = tomato_dir / 'phase1'
    val_target_dir = tomato_dir / 'val'
    test_target_dir = tomato_dir / 'test'

    # Create target directories
    for split_dir in [phase1_dir, val_target_dir, test_target_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_name in class_names:
            (split_dir / class_name).mkdir(exist_ok=True)

    # Organize training data
    print("\n1. Organizing training data (phase1)...")
    for class_name in class_names:
        source_class = train_dir / class_name
        if not source_class.exists():
            print(f"  ⚠️  Class {class_name} not found in train directory, skipping")
            continue

        target_class = phase1_dir / class_name
        image_files = list(source_class.glob('*.jpg')) + \
                     list(source_class.glob('*.jpeg')) + \
                     list(source_class.glob('*.png')) + \
                     list(source_class.glob('*.webp')) + \
                     list(source_class.glob('*.bmp')) + \
                     list(source_class.glob('*.tiff'))

        for img in image_files:
            shutil.copy2(img, target_class / img.name)

        print(f"  {class_name}: {len(image_files)} images")

    # Organize validation data
    if val_dir and val_dir.exists():
        print("\n2. Organizing validation data...")
        for class_name in class_names:
            source_class = val_dir / class_name
            if not source_class.exists():
                print(f"  ⚠️  Class {class_name} not found in val directory, skipping")
                continue

            target_class = val_target_dir / class_name
            image_files = list(source_class.glob('*.jpg')) + \
                         list(source_class.glob('*.jpeg')) + \
                         list(source_class.glob('*.png')) + \
                         list(source_class.glob('*.webp')) + \
                         list(source_class.glob('*.bmp')) + \
                         list(source_class.glob('*.tiff'))

            for img in image_files:
                shutil.copy2(img, target_class / img.name)

            print(f"  {class_name}: {len(image_files)} images")
    else:
        print("\n2. No validation directory found. Will split from training data (80/20)...")
        for class_name in class_names:
            source_class = phase1_dir / class_name
            if not source_class.exists():
                continue

            images = list(source_class.glob('*.*'))
            if len(images) < 2:
                print(f"  ⚠️  {class_name}: only {len(images)} images, cannot split")
                continue

            random.shuffle(images)
            split_idx = int(len(images) * 0.8)
            val_images = images[split_idx:]

            for img in val_images:
                target_class = val_target_dir / class_name
                shutil.move(str(img), str(target_class / img.name))

            print(f"  {class_name}: moved {len(val_images)} to validation")

    # Organize test data
    if test_dir and test_dir.exists():
        print("\n3. Organizing test data...")
        for class_name in class_names:
            source_class = test_dir / class_name
            if not source_class.exists():
                print(f"  ⚠️  Class {class_name} not found in test directory, skipping")
                continue

            target_class = test_target_dir / class_name
            image_files = list(source_class.glob('*.jpg')) + \
                         list(source_class.glob('*.jpeg')) + \
                         list(source_class.glob('*.png')) + \
                         list(source_class.glob('*.webp')) + \
                         list(source_class.glob('*.bmp')) + \
                         list(source_class.glob('*.tiff'))

            for img in image_files:
                shutil.copy2(img, target_class / img.name)

            print(f"  {class_name}: {len(image_files)} images")
    else:
        print("\n3. No test directory found. Creating from training data (10%)...")
        for class_name in class_names:
            source_class = phase1_dir / class_name
            if not source_class.exists():
                continue

            images = list(source_class.glob('*.*'))
            if len(images) < 10:
                print(f"  ⚠️  {class_name}: only {len(images)} images, cannot create test set")
                continue

            test_count = max(1, int(len(images) * 0.1))
            test_images = images[-test_count:]

            for img in test_images:
                target_class = test_target_dir / class_name
                shutil.move(str(img), str(target_class / img.name))

            print(f"  {class_name}: moved {len(test_images)} to test")

    print("\n✅ Dataset organization complete!")
    print(f"\nFinal structure at: {tomato_dir}")
    print("\nClass distribution:")
    for split in ['phase1', 'val', 'test']:
        split_dir = tomato_dir / split
        if split_dir.exists():
            print(f"\n  {split}/")
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob('*.*')))
                    print(f"    {class_dir.name}: {count} images")


def main():
    """Main execution function."""
    print("="*60)
    print("Tomato Disease Dataset Downloader for AADS-ULoRA v5.5")
    print("="*60)

    # Check if in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ Running in Google Colab environment")
    except ImportError:
        IN_COLAB = False
        print("⚠️  Not running in Colab. This script is optimized for Colab.")

    # Install Kaggle API
    install_kaggle()

    # Setup credentials
    if IN_COLAB:
        setup_kaggle_credentials()
    else:
        # For local execution, check if kaggle.json exists
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_json.exists():
            print("\n⚠️  Kaggle credentials not found!")
            print(f"Please place your kaggle.json at: {kaggle_json}")
            print("Or run this script in Colab to upload interactively.")
            return

    # Download dataset
    data_dir = download_dataset()

    # Detect structure
    train_dir, val_dir, test_dir, class_names = detect_dataset_structure(data_dir)

    # Organize dataset
    organize_dataset(data_dir, train_dir, val_dir, test_dir, class_names)

    print("\n" + "="*60)
    print("Dataset is ready for training!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify the data structure in ./data/tomato/")
    print("2. Run the training notebook cells")
    print("3. Monitor GPU usage with nvidia-smi")


if __name__ == "__main__":
    main()
