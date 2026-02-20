#!/usr/bin/env python3
"""
Direct test of BioCLIP-2 using open_clip from GitHub.
Clone open_clip from GitHub and test the preprocessing fix.
"""

import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

print("="*70)
print("SETUP: Clone open_clip from GitHub")
print("="*70)

# Clone open_clip
open_clip_path = "/tmp/open_clip"
if not os.path.exists(open_clip_path):
    print(f"\nCloning open_clip from GitHub to {open_clip_path}...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/mlfoundations/open_clip.git", open_clip_path],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error cloning: {result.stderr}")
        sys.exit(1)
    print("✅ open_clip cloned")
else:
    print(f"✅ open_clip already exists at {open_clip_path}")

# Add to path
sys.path.insert(0, open_clip_path)
print(f"✅ Added {open_clip_path} to Python path")

# Now test
print("\n" + "="*70)
print("TEST: BioCLIP-2 with GitHub open_clip")
print("="*70)

import torch
from PIL import Image
import urllib.request

print(f"\nGPU Available: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Import open_clip (now from GitHub)
import open_clip
print(f"✅ open_clip imported from: {open_clip.__file__}")

# Load BioCLIP-2
print("\n" + "-"*70)
print("Loading BioCLIP-2...")
print("-"*70)

model_id = "imageomics/bioclip-2"
hub_model_id = f"hf-hub:{model_id}"

print(f"Model ID: {hub_model_id}")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
tokenizer = open_clip.get_tokenizer(hub_model_id)

model = model.to(device)
model.eval()
print(f"✅ BioCLIP-2 loaded on {device}")

# Test text encoding
print("\n" + "-"*70)
print("Text Encoding Test")
print("-"*70)

prompts = ["a photo of grape", "a photo of potato", "a photo of tomato", "a photo of strawberry"]
print(f"Prompts: {prompts}\n")

tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_embeds = model.encode_text(tokens)
    print(f"Text embeddings shape: {text_embeds.shape}")
    print(f"Text embeddings norm: {text_embeds.norm(dim=-1)}")
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

print("✅ Text encoding works")

# Download image
print("\n" + "-"*70)
print("Image Download")
print("-"*70)

strawberry_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Fragaria_x_ananassa_Foto_by_CF_Weise.jpg/800px-Fragaria_x_ananassa_Foto_by_CF_Weise.jpg"
image_path = "/tmp/test_strawberry.jpg"

try:
    print(f"Downloading strawberry image...")
    urllib.request.urlretrieve(strawberry_url, image_path)
    image = Image.open(image_path).convert('RGB')
    print(f"✅ Image loaded: {image.size}")
except Exception as e:
    print(f"Download failed: {e}")
    image = Image.new('RGB', (224, 224), color=(200, 0, 0))
    print(f"✅ Using red test image")

# Critical test
print("\n" + "="*70)
print("CRITICAL TEST: preprocess_train vs preprocess_val")
print("="*70)

results = {}

for preprocess_name, preprocess_fn in [("TRAIN", preprocess_train), ("VAL", preprocess_val)]:
    print(f"\n--- Using preprocess_{preprocess_name} ---")
    
    # Preprocess
    image_tensor = preprocess_fn(image).unsqueeze(0).to(device)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"  Min: {image_tensor.min():.4f}, Max: {image_tensor.max():.4f}")
    print(f"  Mean: {image_tensor.mean():.4f}, Std: {image_tensor.std():.4f}")
    
    # Encode
    with torch.no_grad():
        image_embeds = model.encode_image(image_tensor)
        print(f"Image embeddings norm (before): {image_embeds.norm(dim=-1).item():.4f}")
        
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        print(f"Image embeddings norm (after): {image_embeds.norm(dim=-1).item():.4f}")
        
        # Similarity
        logits = image_embeds @ text_embeds.T
        probs = torch.softmax(logits, dim=-1)
        
        print(f"Logits: {logits.squeeze()}")
        print(f"Probabilities: {probs.squeeze()}")
        
        pred_idx = torch.argmax(probs)
        pred_conf = probs[0, pred_idx].item()
        pred_label = prompts[pred_idx].replace("a photo of ", "")
        
        print(f"🎯 Prediction: {pred_label} ({pred_conf:.1%})")
        results[preprocess_name] = (pred_label, pred_conf, probs.squeeze().cpu())

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

train_label, train_conf, train_probs = results["TRAIN"]
val_label, val_conf, val_probs = results["VAL"]

train_is_random = (train_probs - 0.25).abs().max() < 0.02
val_is_random = (val_probs - 0.25).abs().max() < 0.02

print(f"\npreprocess_TRAIN: {train_label} ({train_conf:.1%})")
print(f"  Is random (~25% all): {train_is_random}")

print(f"\npreprocess_VAL: {val_label} ({val_conf:.1%})")
print(f"  Is random (~25% all): {val_is_random}")

print("\n" + "-"*70)
if train_is_random and val_conf > 0.4:
    print("✅ SUCCESS! Preprocessing fix is working!")
    print(f"  preprocess_train shows random {train_conf:.1%} (old buggy behavior)")
    print(f"  preprocess_val shows correct {val_conf:.1%} (fixed behavior)")
elif val_is_random:
    print("❌ Still broken - both preprocesses showing random 25%")
elif val_conf > train_conf:
    print(f"✅ preprocess_val working better: {val_conf:.1%} vs {train_conf:.1%}")
else:
    print("⚠️ Unexpected results - check output above")

print("\n" + "="*70)
print("Done!")
print("="*70)
