#!/usr/bin/env python3
"""
BioCLIP-2 Test with Image Upload
Upload your own crop images and test preprocessing
Canonical script location: scripts/colab_test_upload.py
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
crop_known_prompts = ["a photo of grape", "a photo of potato", "a photo of tomato", "a photo of strawberry"]
crop_unknown_prompts = [
    "a photo of an unknown plant",
    "a photo of a non-crop plant",
    "a photo of random foliage",
    "an unclear plant image",
]
part_known_prompts = [
    "a photo of a plant leaf",
    "a photo of a plant fruit",
    "a photo of a plant stem",
    "a photo of a whole plant",
    "a photo of a plant tuber",
    "a photo of a plant ear",
]
part_unknown_prompts = [
    "a photo of an unknown plant part",
    "a photo of a non-plant object",
    "an unclear close-up image",
]

print(f"Crop labels: {crop_known_prompts}")
print(f"Part labels: {part_known_prompts}")

open_set_min_confidence = 0.55
open_set_margin = 0.10


def encode_prompts(prompt_list):
    tokens = tokenizer(prompt_list).to(device)
    with torch.no_grad():
        embeds = model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds


def classify_open_set(img_embeds, known_prompts, unknown_prompts, text_embeds):
    logits = (img_embeds @ text_embeds.T) * logit_scale
    probs = torch.softmax(logits, dim=-1)

    known_count = len(known_prompts)
    known_probs = probs[:, :known_count]
    unknown_probs = probs[:, known_count:]
    unknown_prob = unknown_probs.max(dim=-1).values if unknown_probs.numel() > 0 else torch.zeros_like(known_probs[:, 0])

    top_conf, top_idx = torch.max(known_probs, dim=-1)
    second_conf = torch.topk(known_probs, k=min(2, known_count), dim=-1).values[:, -1]
    margin = (top_conf - second_conf).item() if known_count > 1 else top_conf.item()

    label = known_prompts[top_idx.item()].replace("a photo of ", "").replace("a plant ", "")
    conf = top_conf.item()

    if unknown_prob.item() >= top_conf.item() or top_conf.item() < open_set_min_confidence or margin < open_set_margin:
        label = "unknown"
        conf = max(unknown_prob.item(), top_conf.item())

    return {
        'label': label,
        'confidence': conf,
        'unknown_prob': unknown_prob.item(),
        'margin': margin,
        'logits': logits,
        'probs': probs,
    }


# Encode text
print("\n6. Encoding text labels...")
crop_prompts = crop_known_prompts + crop_unknown_prompts
part_prompts = part_known_prompts + part_unknown_prompts
crop_text_embeds = encode_prompts(crop_prompts)
part_text_embeds = encode_prompts(part_prompts)
print(f"✅ Crop text embeddings: {crop_text_embeds.shape}")
print(f"✅ Part text embeddings: {part_text_embeds.shape}")

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
crop_train = classify_open_set(img_embeds_train, crop_known_prompts, crop_unknown_prompts, crop_text_embeds)
part_train = classify_open_set(img_embeds_train, part_known_prompts, part_unknown_prompts, part_text_embeds)

print(f"Embedding norm (before): {img_embeds_train_norm:.4f}")
print(f"Crop logits: {crop_train['logits'].squeeze()}")
print(f"Crop probs:  {crop_train['probs'].squeeze()}")
print(f"Part logits: {part_train['logits'].squeeze()}")
print(f"Part probs:  {part_train['probs'].squeeze()}")
print(f"🎯 Crop prediction: {crop_train['label']} ({crop_train['confidence']:.1%})")
print(f"🎯 Part prediction: {part_train['label']} ({part_train['confidence']:.1%})")

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
crop_val = classify_open_set(img_embeds_val, crop_known_prompts, crop_unknown_prompts, crop_text_embeds)
part_val = classify_open_set(img_embeds_val, part_known_prompts, part_unknown_prompts, part_text_embeds)

print(f"Embedding norm (before): {img_embeds_val_norm:.4f}")
print(f"Crop logits: {crop_val['logits'].squeeze()}")
print(f"Crop probs:  {crop_val['probs'].squeeze()}")
print(f"Part logits: {part_val['logits'].squeeze()}")
print(f"Part probs:  {part_val['probs'].squeeze()}")
print(f"🎯 Crop prediction: {crop_val['label']} ({crop_val['confidence']:.1%})")
print(f"🎯 Part prediction: {part_val['label']} ({part_val['confidence']:.1%})")

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

logits_diff = (crop_train['logits'] - crop_val['logits']).abs().squeeze()
print(f"\nLogits difference:")
print(f"  Max: {logits_diff.max():.4f}")
print(f"  Mean: {logits_diff.mean():.4f}")

print(f"\nLogits variance:")
print(f"  Crop TRAIN: {crop_train['logits'].squeeze().var():.4f}")
print(f"  Crop VAL:   {crop_val['logits'].squeeze().var():.4f}")
print(f"  Part TRAIN: {part_train['logits'].squeeze().var():.4f}")
print(f"  Part VAL:   {part_val['logits'].squeeze().var():.4f}")

print(f"\nPredictions:")
print(f"  preprocess_TRAIN crop: {crop_train['label']} ({crop_train['confidence']:.1%})")
print(f"  preprocess_VAL   crop: {crop_val['label']} ({crop_val['confidence']:.1%})")
print(f"  preprocess_TRAIN part: {part_train['label']} ({part_train['confidence']:.1%})")
print(f"  preprocess_VAL   part: {part_val['label']} ({part_val['confidence']:.1%})")
print(f"  Crop confidence diff: {abs(crop_train['confidence'] - crop_val['confidence']):.1%}")
print(f"  Part confidence diff: {abs(part_train['confidence'] - part_val['confidence']):.1%}")
print(f"\nOpen-set thresholds:")
print(f"  min_confidence: {open_set_min_confidence:.2f}")
print(f"  min_margin: {open_set_margin:.2f}")
print(f"  crop unknown_prob TRAIN/VAL: {crop_train['unknown_prob']:.1%} / {crop_val['unknown_prob']:.1%}")
print(f"  part unknown_prob TRAIN/VAL: {part_train['unknown_prob']:.1%} / {part_val['unknown_prob']:.1%}")
print(f"  crop margin TRAIN/VAL: {crop_train['margin']:.1%} / {crop_val['margin']:.1%}")
print(f"  part margin TRAIN/VAL: {part_train['margin']:.1%} / {part_val['margin']:.1%}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if crop_train['confidence'] > 0.4 and crop_val['confidence'] > 0.4:
    print("\n✅ GOOD: Both preprocesses work (>40% confidence)")
    if abs(crop_train['confidence'] - crop_val['confidence']) > 0.05:
        print(f"   Different crop results: TRAIN={crop_train['confidence']:.1%}, VAL={crop_val['confidence']:.1%}")
    else:
        print(f"   Similar results: Preprocessing doesn't matter for this image")
    print(f"   Part summary: TRAIN={part_train['label']} ({part_train['confidence']:.1%}), VAL={part_val['label']} ({part_val['confidence']:.1%})")

elif crop_train['confidence'] < 0.3 and crop_val['confidence'] > 0.4:
    print("\n✅✅✅ FIX CONFIRMED!")
    print(f"   preprocess_train shows random {crop_train['confidence']:.1%}")
    print(f"   preprocess_val shows correct {crop_val['confidence']:.1%}")

elif crop_train['confidence'] < 0.3 and crop_val['confidence'] < 0.3:
    print("\n❌ Both fail (<30% confidence)")
    print(f"   This suggests BioCLIP doesn't recognize the crop well")
    print(f"   Try with a clearer/closer image of the crop")

else:
    print(f"\n⚠️ Mixed results")
    print(f"   Crop TRAIN: {crop_train['confidence']:.1%}, Crop VAL: {crop_val['confidence']:.1%}")
    print(f"   Analysis: Embedding norms are {img_embeds_train_norm:.4f} / {img_embeds_val_norm:.4f}")

print("\n" + "="*70)
