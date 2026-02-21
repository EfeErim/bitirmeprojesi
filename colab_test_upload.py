#!/usr/bin/env python3
"""
BioCLIP-2 Test with Image Upload
Upload your own crop images and test preprocessing
"""

print("="*70)
print("BioCLIP-2 Preprocessing Test with Image Upload")
print("="*70)

# Install & setup
print("\n1. Installing dependencies...")
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open-clip-torch"])
print("✅ Installed")

print("\n2. Checking GPU...")
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ GPU: {torch.cuda.is_available()}, Device: {device}")

print("\n3. Loading BioCLIP-2...")
import open_clip
from PIL import Image
import os

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
model = model.to(device).eval()
print(f"✅ Loaded on {device}")

logit_scale = float(model.logit_scale.exp().item()) if hasattr(model, 'logit_scale') else 1.0
logit_scale = max(1.0, min(logit_scale, 100.0))
print(f"✅ CLIP logit_scale: {logit_scale:.2f}")

# Upload file
print("\n4. Uploading image...")
try:
    from google.colab import files
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]
    image = Image.open(image_path).convert('RGB')
    print(f"✅ Loaded: {image_path} ({image.size})")
except Exception as e:
    print(f"Error: {e}")
    print("Running in non-Colab environment")
    print("Please provide image path or upload via Colab")
    sys.exit(1)

# Test labels
print("\n5. Setting up test labels...")
prompts = ["a photo of grape", "a photo of potato", "a photo of tomato", "a photo of strawberry"]
print(f"Testing labels: {prompts}")

# Encode text
print("\n6. Encoding text labels...")
tokens = tokenizer(prompts).to(device)
with torch.no_grad():
    text_embeds = model.encode_text(tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
print(f"✅ Text embeddings: {text_embeds.shape}")

# Test TRAIN preprocessing
print("\n" + "="*70)
print("Testing: preprocess_TRAIN (with augmentation)")
print("="*70)

img_tensor_train = preprocess_train(image).unsqueeze(0).to(device)
print(f"Tensor: shape={img_tensor_train.shape}, min={img_tensor_train.min():.3f}, max={img_tensor_train.max():.3f}")

with torch.no_grad():
    img_embeds_train = model.encode_image(img_tensor_train)
    img_embeds_train_norm = img_embeds_train.norm(dim=-1).item()
    img_embeds_train = img_embeds_train / img_embeds_train.norm(dim=-1, keepdim=True)
    logits_train = (img_embeds_train @ text_embeds.T) * logit_scale
    probs_train = torch.softmax(logits_train, dim=-1)

print(f"Embedding norm (before): {img_embeds_train_norm:.4f}")
print(f"Logits: {logits_train.squeeze()}")
print(f"Probs:  {probs_train.squeeze()}")

idx_train = torch.argmax(probs_train).item()
conf_train = probs_train[0, idx_train].item()
label_train = prompts[idx_train].replace("a photo of ", "")
print(f"🎯 Prediction: {label_train} ({conf_train:.1%})")

# Test VAL preprocessing
print("\n" + "="*70)
print("Testing: preprocess_VAL (no augmentation)")
print("="*70)

img_tensor_val = preprocess_val(image).unsqueeze(0).to(device)
print(f"Tensor: shape={img_tensor_val.shape}, min={img_tensor_val.min():.3f}, max={img_tensor_val.max():.3f}")

with torch.no_grad():
    img_embeds_val = model.encode_image(img_tensor_val)
    img_embeds_val_norm = img_embeds_val.norm(dim=-1).item()
    img_embeds_val = img_embeds_val / img_embeds_val.norm(dim=-1, keepdim=True)
    logits_val = (img_embeds_val @ text_embeds.T) * logit_scale
    probs_val = torch.softmax(logits_val, dim=-1)

print(f"Embedding norm (before): {img_embeds_val_norm:.4f}")
print(f"Logits: {logits_val.squeeze()}")
print(f"Probs:  {probs_val.squeeze()}")

idx_val = torch.argmax(probs_val).item()
conf_val = probs_val[0, idx_val].item()
label_val = prompts[idx_val].replace("a photo of ", "")
print(f"🎯 Prediction: {label_val} ({conf_val:.1%})")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print(f"\nTensors identical: {torch.allclose(img_tensor_train, img_tensor_val)}")
print(f"\nPreprocessing difference:")
print(f"  Tensor min/max TRAIN: [{img_tensor_train.min():.3f}, {img_tensor_train.max():.3f}]")
print(f"  Tensor min/max VAL:   [{img_tensor_val.min():.3f}, {img_tensor_val.max():.3f}]")

print(f"\nEmbedding norms (before normalization):")
print(f"  TRAIN: {img_embeds_train_norm:.4f}")
print(f"  VAL:   {img_embeds_val_norm:.4f}")

logits_diff = (logits_train - logits_val).abs().squeeze()
print(f"\nLogits difference:")
print(f"  Max: {logits_diff.max():.4f}")
print(f"  Mean: {logits_diff.mean():.4f}")

print(f"\nLogits variance:")
print(f"  TRAIN: {logits_train.squeeze().var():.4f}")
print(f"  VAL:   {logits_val.squeeze().var():.4f}")

print(f"\nPredictions:")
print(f"  preprocess_TRAIN: {label_train} ({conf_train:.1%})")
print(f"  preprocess_VAL:   {label_val} ({conf_val:.1%})")
print(f"  Confidence diff: {abs(conf_train - conf_val):.1%}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if conf_train > 0.4 and conf_val > 0.4:
    print("\n✅ GOOD: Both preprocesses work (>40% confidence)")
    if abs(conf_train - conf_val) > 0.05:
        print(f"   Different results: TRAIN={conf_train:.1%}, VAL={conf_val:.1%}")
    else:
        print(f"   Similar results: Preprocessing doesn't matter for this image")
        
elif conf_train < 0.3 and conf_val > 0.4:
    print("\n✅✅✅ FIX CONFIRMED!")
    print(f"   preprocess_train shows random {conf_train:.1%}")
    print(f"   preprocess_val shows correct {conf_val:.1%}")
    
elif conf_train < 0.3 and conf_val < 0.3:
    print("\n❌ Both fail (<30% confidence)")
    print(f"   This suggests BioCLIP doesn't recognize the crop well")
    print(f"   Try with a clearer/closer image of the crop")
    
else:
    print(f"\n⚠️ Mixed results")
    print(f"   TRAIN: {conf_train:.1%}, VAL: {conf_val:.1%}")
    print(f"   Analysis: Embedding norms are {img_embeds_train_norm:.4f} / {img_embeds_val_norm:.4f}")

print("\n" + "="*70)
