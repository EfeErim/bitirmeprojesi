#!/usr/bin/env python3
"""
Setup script for AADS-ULoRA v5.5
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="aads-ulora-v5.5",
    version="5.5.0",
    author="Agricultural AI Development Team",
    author_email="contact@aads.com",
    description="Agricultural AI Development System - ULoRA v5.5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/aads-ulora-v5.5",
    
    # Packages
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers==4.56.0",
        "peft>=0.8.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.7.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "redis>=4.5.0",
        "boto3>=1.28.0",
        "gradio>=3.45.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    
    # Optional dependencies
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
        ]
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "aads-train=src.training.phase1_training:main",
            "aads-demo=demo.app:main",
            "aads-pipeline=src.pipeline.independent_multi_crop_pipeline:main",
        ],
    },
    
    # Classifiers
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
    
    # Keywords
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