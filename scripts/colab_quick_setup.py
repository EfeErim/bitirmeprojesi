#!/usr/bin/env python3
"""
AADS v6 Colab quick setup snippet.

Run this script in Colab (or copy the steps) to clone/update the repository
and open the active two-notebook flow.
"""

from pathlib import Path
import os
import subprocess

print("AADS v6 Colab Quick Setup")
print("=" * 60)

from google.colab import drive

drive.mount('/content/drive')

project_root = Path('/content/drive/MyDrive/aads_v6')
project_root.mkdir(parents=True, exist_ok=True)
repo_path = project_root / 'bitirmeprojesi'

if not repo_path.exists():
    print("Cloning repository...")
    subprocess.run(
        ['git', 'clone', 'https://github.com/EfeErim/bitirmeprojesi.git', str(repo_path)],
        check=True,
    )
else:
    print("Repository already exists. Updating...")
    os.chdir(repo_path)
    subprocess.run(['git', 'pull', 'origin', 'master'], check=True)

print("\nSetup complete.")
print(f"Repository location: {repo_path}")
print("\nOpen notebooks in this order:")
print(f"  1) {repo_path / 'colab_notebooks' / '1_crop_router_pipeline.ipynb'}")
print(f"  2) {repo_path / 'colab_notebooks' / '2_interactive_adapter_training.ipynb'}")
print("\nOptional automation:")
print(f"  python {repo_path / 'scripts' / 'colab_auto_orchestrator.py'}")
