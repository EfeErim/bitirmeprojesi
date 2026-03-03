#!/usr/bin/env python3
"""Setup script for AADS v6."""

from pathlib import Path

from setuptools import find_packages, setup

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="aads-v6",
    version="6.0.0",
    author="Agricultural AI Development Team",
    author_email="contact@aads.com",
    description="Agricultural AI Development System v6 (continual SD-LoRA runtime)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EfeErim/bitirmeprojesi",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.50.0,<5.0.0",
        "peft>=0.8.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.7.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "mobile": [
            "tflite-runtime>=2.13.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Agriculture",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Agriculture",
    ],
    keywords=[
        "agriculture",
        "crop-disease",
        "deep-learning",
        "vit",
        "lora",
        "continual-learning",
        "ood-detection",
        "parameter-efficient-fine-tuning",
    ],
)
