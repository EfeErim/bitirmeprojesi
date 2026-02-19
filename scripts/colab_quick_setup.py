#!/usr/bin/env python3
"""
AADS-ULoRA Colab Quick Setup
Run this cell in Google Colab to start the complete auto-training pipeline.
"""

# Copy these lines directly into a Colab cell and run:

import subprocess
import sys
from pathlib import Path

print("🚀 AADS-ULoRA Auto-Training Setup")
print("=" * 60)

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup directory
project_root = Path('/content/drive/MyDrive/aads_ulora')
project_root.mkdir(parents=True, exist_ok=True)

# Clone repo
repo_path = project_root / 'bitirmeprojesi'
if not repo_path.exists():
    print("Cloning repository...")
    subprocess.run(
        ['git', 'clone', 'https://github.com/EfeErim/bitirmeprojesi.git', str(repo_path)],
        check=True
    )
else:
    print("Repository already exists. Updating...")
    import os
    os.chdir(repo_path)
    subprocess.run(['git', 'pull', 'origin', 'master'], check=True)

print("\n✅ Setup complete!")
print(f"📁 Repository location: {repo_path}")
print("\n📓 Open this notebook to start training:")
print(f"   {repo_path / 'colab_notebooks' / '0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb'}")
print("\nOr run the auto-orchestrator:")
print(f"   python {repo_path / 'scripts' / 'colab_auto_orchestrator.py'}")
