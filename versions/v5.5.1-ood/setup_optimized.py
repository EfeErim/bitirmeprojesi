#!/usr/bin/env python3
"""
Optimized setup.py for AADS-ULoRA v5.5
Based on dependency analysis and usage patterns.
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
        # Core ML Framework
        "torch==2.1.0",
        "torchvision==0.16.0",
        
        # Transformers and PEFT
        "transformers==4.33.2",
        "peft==0.8.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        
        # Data Processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.7.0",
        "albumentations>=1.3.0",
        
        # Machine Learning
        "scikit-learn>=1.3.0",
        
        # Web API
        "fastapi==0.104.1",
        "uvicorn[standard]==0.23.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
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
        ],
        "api": [
            "psycopg2-binary>=2.9.0",
            "redis>=4.5.0",
            "boto3>=1.28.0",
            "minio>=7.0.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "colab": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "tensorboard>=2.12.0",
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