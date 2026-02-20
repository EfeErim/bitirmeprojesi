#!/usr/bin/env python3
"""
BioCLIP-2 Preprocessing Test - FRESH VERSION
No caching issues, runs directly in Colab.
"""

print("="*70)
print("BioCLIP-2 Preprocessing Test")
print("="*70)

# Install dependencies
print("\n1. Installing dependencies...")
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open-clip-torch"])
print("✅ Installed")

# Check GPU
print("\n2. Checking GPU...")
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ GPU Available: {torch.cuda.is_available()}, Device: {device}")

# Load BioCLIP-2
print("\n3. Loading BioCLIP-2...")
import open_clip
from PIL import Image, ImageDraw
import numpy as np

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
model = model.to(device).eval()
print(f"✅ Loaded on {device}")

# Create HIGH-QUALITY synthetic strawberry image (not solid color!)
print("\n4. Creating test image...")

# FIRST: Try to download real image
real_image = None
try:
    print("  Attempting to download real strawberry image...")
    import urllib.request
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Fragaria_x_ananassa_Foto_by_CF_Weise.jpg/800px-Fragaria_x_ananassa_Foto_by_CF_Weise.jpg"
    with urllib.request.urlopen(url) as response:
        import io
        image_data = response.read()
    real_image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = real_image
    print(f"✅ Downloaded real strawberry image: {image.size}")
except Exception as e:
    print(f"  Download failed: {e}")
    print("  Creating synthetic strawberry instead...")
    
    # Fallback: Create synthetic image
    image = Image.new('RGB', (224, 224), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)

    # Strawberry body (red circle)
    draw.ellipse([40, 50, 160, 170], fill=(220, 30, 60), outline=(180, 20, 40))

    # Green leafy crown (multiple leaves)
    leaves = [
        (80, 40, 70, 20),
        (110, 35, 120, 15),
        (140, 45, 150, 25),
        (100, 35, 110, 10),
    ]
    for x1, y1, x2, y2 in leaves:
        draw.line([(x1, y1), (x2, y2)], fill=(34, 139, 34), width=3)

    # Seed texture (small yellow dots)
    for i in range(45, 155, 12):
        for j in range(60, 165, 12):
            draw.ellipse([i, j, i+4, j+4], fill=(255, 200, 50))

    # Highlight (make it 3D-looking)
    draw.ellipse([60, 65, 90, 95], fill=(255, 100, 100))
    
    print(f"✅ Created synthetic strawberry image: {image.size}")

# DEBUG output
print("\n" + "="*70)
print("DEBUG: Check preprocessing functions")
print("="*70)
print(f"preprocess_train == preprocess_val: {preprocess_train == preprocess_val}")
print(f"preprocess_train: {preprocess_train}")
print(f"preprocess_val: {preprocess_val}")

# Encode text
print("\n5. Encoding text labels...")
prompts = ["a photo of grape", "a photo of potato", "a photo of tomato", "a photo of strawberry"]
tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_embeds = model.encode_text(tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
print(f"✅ Text embeddings: {text_embeds.shape}")

# Test TRAIN preprocessing
print("\n" + "="*70)
print("6. Testing preprocess_TRAIN (with augmentation)")
print("="*70)

img_tensor_train = preprocess_train(image).unsqueeze(0).to(device)
print(f"Tensor shape: {img_tensor_train.shape}")
print(f"Min/Max: [{img_tensor_train.min():.3f}, {img_tensor_train.max():.3f}]")

with torch.no_grad():
    img_embeds_train = model.encode_image(img_tensor_train)
    img_embeds_train = img_embeds_train / img_embeds_train.norm(dim=-1, keepdim=True)
    logits_train = img_embeds_train @ text_embeds.T
    probs_train = torch.softmax(logits_train, dim=-1)

print(f"Logits: {logits_train.squeeze()}")
conf_train = probs_train[0, torch.argmax(probs_train)].item()
label_train = prompts[torch.argmax(probs_train).item()].replace("a photo of ", "")
print(f"🎯 Prediction: {label_train} ({conf_train:.1%})")

# Test VAL preprocessing
print("\n" + "="*70)
print("7. Testing preprocess_VAL (no augmentation)")
print("="*70)

img_tensor_val = preprocess_val(image).unsqueeze(0).to(device)
print(f"Tensor shape: {img_tensor_val.shape}")
print(f"Min/Max: [{img_tensor_val.min():.3f}, {img_tensor_val.max():.3f}]")

with torch.no_grad():
    img_embeds_val = model.encode_image(img_tensor_val)
    img_embeds_val = img_embeds_val / img_embeds_val.norm(dim=-1, keepdim=True)
    logits_val = img_embeds_val @ text_embeds.T
    probs_val = torch.softmax(logits_val, dim=-1)

print(f"Logits: {logits_val.squeeze()}")
conf_val = probs_val[0, torch.argmax(probs_val)].item()
label_val = prompts[torch.argmax(probs_val).item()].replace("a photo of ", "")
print(f"🎯 Prediction: {label_val} ({conf_val:.1%})")

# Comparison
print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)

print(f"\nTensors identical: {torch.allclose(img_tensor_train, img_tensor_val)}")
print(f"\npreprocess_TRAIN: {label_train} ({conf_train:.1%})")
print(f"preprocess_VAL:   {label_val} ({conf_val:.1%})")

# Analyze the logits
print(f"\nLogits difference:")
logits_diff = (logits_train - logits_val).abs().squeeze()
print(f"  Max difference: {logits_diff.max():.4f}")
print(f"  Mean difference: {logits_diff.mean():.4f}")

# Check if logits are flat (all same - indicates no meaningful features)
print(f"\nLogits variance:")
print(f"  TRAIN variance: {logits_train.squeeze().var():.4f}")
print(f"  VAL variance: {logits_val.squeeze().var():.4f}")

# Check embedding norms
img_norm_train = img_embeds_train.norm(dim=-1).item()
img_norm_val = img_embeds_val.norm(dim=-1).item()
print(f"\nImage embedding norms (before normalization):")
print(f"  TRAIN: {img_norm_train:.4f}")
print(f"  VAL: {img_norm_val:.4f}")

print(f"\n" + "-"*70)

if conf_train < 0.3 and conf_val > 0.4:
    print("✅✅✅ FIX CONFIRMED! ✅✅✅")
    print(f"preprocess_train shows random {conf_train:.1%}")
    print(f"preprocess_val shows correct {conf_val:.1%}")
elif conf_val > conf_train:
    print(f"✅ preprocess_val better: {conf_val:.1%} vs {conf_train:.1%}")
elif logits_train.squeeze().var() < 0.01 and logits_val.squeeze().var() < 0.01:
    print(f"\n⚠️ BOTH show flat logits - features not being captured!")
    print(f"  This suggests the image isn't recognized as a crop by BioCLIP")
    print(f"  Try with a real photo instead of synthetic image")
else:
    print(f"⚠️ Results unclear - diff: {conf_val - conf_train:.1%}")

print("\n" + "="*70)
