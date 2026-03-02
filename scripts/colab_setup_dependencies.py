#!/usr/bin/env python3
"""
Colab setup script for AADS v6 notebooks and optional router diagnostics.

IMPORTANT: Run these cells FIRST in your Colab notebook:

    # Cell 1: Clone repo and change directory
    !git clone https://github.com/EfeErim/bitirmeprojesi.git /content/bitirmeprojesi 2>/dev/null || !cd /content/bitirmeprojesi && git pull
    %cd /content/bitirmeprojesi

    # Cell 2: Run this setup script
    %run scripts/colab_setup_dependencies.py
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_URL = "https://github.com/EfeErim/bitirmeprojesi.git"
REPO_DIR = Path("/content/bitirmeprojesi")


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def run_command(cmd: Sequence[str], description: str = "", cwd: Optional[Path] = None) -> bool:
    """Run command with error handling and without shell expansion."""
    try:
        print(f"\n{'=' * 60}")
        print(f"Setting up: {description}")
        print(f"Command: {_format_cmd(cmd)}")
        print(f"{'=' * 60}")
        result = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print(f"[OK] {description} - SUCCESS")
            return True
        print(f"[WARN] {description} - WARNING (check logs above)")
        return False
    except Exception as exc:
        print(f"[ERROR] {description} - ERROR: {exc}")
        return False


def ensure_repo_checkout(repo_dir: Path) -> bool:
    """Clone repository when missing, otherwise pull latest changes."""
    if not repo_dir.exists():
        return run_command(
            ["git", "clone", REPO_URL, str(repo_dir)],
            "Clone repository",
        )
    return run_command(
        ["git", "pull"],
        "Update repository",
        cwd=repo_dir,
    )


def main() -> None:
    """Main setup function."""
    failed_steps = []

    # Check if we're in the repo directory, if not try to cd or clone
    if not Path("src").exists():
        print("\n[WARN] Not in bitirmeprojesi directory. Preparing checkout...")
        if not ensure_repo_checkout(REPO_DIR):
            failed_steps.append("Clone/update repository")
        os.chdir(REPO_DIR)
        print("[OK] Changed to repository directory")
    else:
        # If already in repository, still try to sync latest code.
        if not ensure_repo_checkout(Path.cwd()):
            failed_steps.append("Clone/update repository")

    print("\n" + "=" * 70)
    print(" AADS v6 Colab Setup")
    print("=" * 70)

    setup_steps = [
        # 1. Update pip
        ([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrade pip"),
        # 2. Install transformers with SAM3 support
        ([sys.executable, "-m", "pip", "install", "--upgrade", "transformers>=4.41.0"], "Install transformers (SAM3 support)"),
        # 3. Install open_clip for BioCLIP-2.5
        ([sys.executable, "-m", "pip", "install", "open-clip-torch"], "Install open-clip (BioCLIP-2.5)"),
        # 4. Install ultralytics for SAM2.1
        ([sys.executable, "-m", "pip", "install", "ultralytics"], "Install ultralytics (SAM2.1 fallback)"),
        # 5. Install GroundingDINO for fallback
        ([sys.executable, "-m", "pip", "install", "groundingdino-hf"], "Install GroundingDINO (DINO fallback)"),
        # 6. Install torch-vision for image processing
        ([sys.executable, "-m", "pip", "install", "torchvision"], "Install torchvision (image ops)"),
        # 7. Install huggingface_hub for model access
        ([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface-hub"], "Install huggingface_hub (model hub)"),
    ]

    for cmd, desc in setup_steps:
        if not run_command(cmd, desc):
            failed_steps.append(desc)

    # Summary
    print("\n" + "=" * 70)
    print(" SETUP COMPLETE")
    print("=" * 70)

    if not failed_steps:
        print("\n[OK] ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("\nYou can now run:")
        print("  %cd /content/bitirmeprojesi")
        print("  %run scripts/validate_notebook_imports.py")
    else:
        print(f"\n[WARN] {len(failed_steps)} step(s) had warnings:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nYou may still be able to proceed, but check the output above.")

    print("\n" + "=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    print("""
1. Set HF_TOKEN as a Colab secret:
   - Click the Secrets tab in left sidebar
   - Click "+ Add new secret"
   - Name: HF_TOKEN
   - Value: Your HuggingFace token (from https://huggingface.co/settings/tokens)
   - Enable "Notebook access"
   - Save

2. Run training readiness check:
   %cd /content/bitirmeprojesi
   %run scripts/validate_notebook_imports.py

3. Start the continual training notebook:
   Open: colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb
   Set DATASET_ROOT_INPUT to your class-root dataset path
   Set RUN_PIPELINE=True when ready

4. Optional router-only VLM sanity check:
   %run scripts/colab_vlm_quick_test.py
""")


if __name__ == "__main__":
    main()
