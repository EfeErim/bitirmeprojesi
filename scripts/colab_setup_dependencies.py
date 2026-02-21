#!/usr/bin/env python3
"""
Colab Setup Script: Install all dependencies for SAM3+BioCLIP-2.5 pipeline

IMPORTANT: Run these cells FIRST in your Colab notebook:

    # Cell 1: Clone repo and change directory
    !git clone https://github.com/EfeErim/bitirmeprojesi.git /content/bitirmeprojesi 2>/dev/null || !cd /content/bitirmeprojesi && git pull
    %cd /content/bitirmeprojesi

    # Cell 2: Run this setup script
    %run scripts/colab_setup_dependencies.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command with error handling."""
    try:
        print(f"\n{'='*60}")
        print(f"Setting up: {description}")
        print(f"Command: {cmd}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"⚠️ {description} - WARNING (check logs above)")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Main setup function."""
    import os
    
    # Check if we're in the repo directory, if not try to cd or clone
    if not os.path.exists('src'):
        print("\n⚠️  Not in bitirmeprojesi directory. Cloning now...")
        os.system("git clone https://github.com/EfeErim/bitirmeprojesi.git /content/bitirmeprojesi 2>/dev/null || (cd /content/bitirmeprojesi && git pull)")
        os.chdir('/content/bitirmeprojesi')
        print("✅ Cloned/updated and changed to repository directory")
    
    print("\n" + "="*70)
    print(" AADS-ULoRA SAM3+BioCLIP-2.5 Pipeline - Colab Setup")
    print("="*70)
    
    setup_steps = [
        # 1. Update pip
        ("pip install --upgrade pip", "Upgrade pip"),
        
        # 2. Clone/update repo
        ("git clone https://github.com/EfeErim/bitirmeprojesi.git /content/bitirmeprojesi 2>/dev/null || cd /content/bitirmeprojesi && git pull", 
         "Clone/update repository"),
        
        # 3. Install transformers with SAM3 support
        ("pip install transformers>=4.41.0 --upgrade", 
         "Install transformers (SAM3 support)"),
        
        # 4. Install open_clip for BioCLIP-2.5
        ("pip install open-clip-torch", 
         "Install open-clip (BioCLIP-2.5)"),
        
        # 5. Install ultralytics for SAM2.1
        ("pip install ultralytics", 
         "Install ultralytics (SAM2.1 fallback)"),
        
        # 6. Install GroundingDINO for fallback
        ("pip install groundingdino-hf", 
         "Install GroundingDINO (DINO fallback)"),
        
        # 7. Install torch-vision for image processing
        ("pip install torchvision",
         "Install torchvision (image ops)"),
        
        # 8. Install huggingface_hub for model access
        ("pip install huggingface-hub --upgrade",
         "Install huggingface_hub (model hub)"),
    ]
    
    failed_steps = []
    
    for cmd, desc in setup_steps:
        if not run_command(cmd, desc):
            failed_steps.append(desc)
    
    # Summary
    print("\n" + "="*70)
    print(" SETUP COMPLETE")
    print("="*70)
    
    if not failed_steps:
        print("\n✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("\nYou can now run:")
        print("  %cd /content/bitirmeprojesi")
        print("  %run scripts/colab_vlm_quick_test.py")
    else:
        print(f"\n⚠️ {len(failed_steps)} step(s) had warnings:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nYou may still be able to proceed, but check the output above.")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("""
1. Set HF_TOKEN as a Colab secret:
   - Click 🔑 Secrets in left sidebar
   - Click "+ Add new secret"
   - Name: HF_TOKEN
   - Value: Your HuggingFace token (from https://huggingface.co/settings/tokens)
   - Enable "Notebook access"
   - Save

2. Run the pipeline test:
   %cd /content/bitirmeprojesi
   %run scripts/colab_vlm_quick_test.py

3. Upload your grape leaf image when prompted
""")

if __name__ == "__main__":
    main()
