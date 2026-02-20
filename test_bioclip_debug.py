#!/usr/bin/env python3
"""
Deep diagnosis of BioCLIP-2 alignment issues
Test with different prompts and compare with standard CLIP
"""

print("="*70)
print("BioCLIP-2 Deep Diagnosis")
print("="*70)

import subprocess, sys, torch
from PIL import Image

# Install
print("\n1. Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open-clip-torch"])
print("✅ Installed")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Device: {device}")

# Try to get image
print("\n2. Getting test image...")
try:
    from google.colab import files
    print("Uploading image...")
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]
    image = Image.open(image_path).convert('RGB')
    print(f"✅ Loaded: {image_path} ({image.size})")
except:
    print("⚠️  Running without Colab - creating test image")
    import requests
    from io import BytesIO
    # Download a real strawberry image
    url = "https://images.unsplash.com/photo-1599599810694-b5ac4dd7f5ad?w=224"
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"✅ Downloaded test image (224x224)")
    except:
        # Fallback: create solid red image
        image = Image.new('RGB', (224, 224), color='red')
        print("☑️ Using solid red fallback image")

# Load BioCLIP-2
print("\n3. Loading BioCLIP-2...")
import open_clip

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
model = model.to(device).eval()
print("✅ BioCLIP-2 loaded")

preprocess = preprocess_val  # Use validation preprocessing

# Prepare image embedding
print("\n4. Encoding image...")
img_tensor = preprocess(image).unsqueeze(0).to(device)
with torch.no_grad():
    img_embed = model.encode_image(img_tensor)
    img_embed_norm_raw = img_embed.norm(dim=-1).item()
    img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

print(f"Image embedding norm (raw): {img_embed_norm_raw:.4f}")
print(f"Image embedding (normalized): {img_embed.shape}")
print(f"Sample values: {img_embed[0, :5].cpu()}")

# Test different prompts
print("\n" + "="*70)
print("Testing Different Prompt Formats")
print("="*70)

prompt_sets = {
    "Full prompts": [
        "a photo of grape",
        "a photo of potato", 
        "a photo of tomato",
        "a photo of strawberry",
    ],
    "Simple names": [
        "grape",
        "potato",
        "tomato", 
        "strawberry",
    ],
    "Plant prompts": [
        "a plant of grape",
        "a plant of potato",
        "a plant of tomato",
        "a plant of strawberry",
    ],
    "Crop prompts": [
        "a crop of grape",
        "a crop of potato",
        "a crop of tomato",
        "a crop of strawberry",
    ],
    "Biology prompts": [
        "a photo of a grape fruit organism",
        "a photo of a potato plant organism",
        "a photo of a tomato fruit organism",
        "a photo of a strawberry fruit organism",
    ],
}

for prompt_name, prompts in prompt_sets.items():
    print(f"\n{prompt_name}:")
    print("-" * 50)
    
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds_norms = text_embeds.norm(dim=-1)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity (cosine distance = dot product after L2 normalization)
        logits = img_embed @ text_embeds.T
        probs = torch.softmax(logits, dim=-1)
        
    logits_vals = logits.squeeze().cpu()
    probs_vals = probs.squeeze().cpu()
    
    for i, prompt in enumerate(prompts):
        label = prompt.replace("a photo of ", "").replace("a plant of ", "").replace("a crop of ", "").replace("a photo of a ", "").replace(" organism", "")
        print(f"  {label:20s} | logit={logits_vals[i]:6.4f} | prob={probs_vals[i]:6.1%}")
    
    best_idx = torch.argmax(probs).item()
    best_label = prompts[best_idx].split()[-1]
    best_prob = probs_vals[best_idx].item()
    logits_variance = logits_vals.var().item()
    
    print(f"  {'BEST':20s} | {best_label:20s} ({best_prob:.1%}), variance={logits_variance:.6f}")

# Test with standard objects to verify model works
print("\n" + "="*70)
print("Sanity Check: Does BioCLIP work at all?")
print("="*70)

sanity_checks = {
    "Colors": ["red", "blue", "green", "yellow"],
    "Common objects": ["dog", "cat", "car", "tree"],
    "Body parts": ["eye", "hand", "nose", "mouth"],
}

print("\nTesting with sanity check prompts:")
for check_name, prompts in sanity_checks.items():
    print(f"\n{check_name}:")
    
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logits = img_embed @ text_embeds.T
    
    logits_vals = logits.squeeze().cpu()
    logits_variance = logits_vals.var().item()
    logits_range = (logits_vals.max() - logits_vals.min()).item()
    
    print(f"  Range: {logits_range:.4f}, Variance: {logits_variance:.6f}")
    
    for i, prompt in enumerate(prompts):
        print(f"    {prompt}: {logits_vals[i]:.4f}")

# Final analysis
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("""
Interpretation guide:
- Logits range = (max - min). Higher = more discrimination
- Logits variance. Higher = more discrimination between classes
- If variance < 0.001 for all prompt formats → Model outputs are nearly uniform
  → This means image embedding is equidistant from all text embeddings
  → This suggests:
     1. Image and text are in different embedding spaces (model issue)
     2. BioCLIP doesn't recognize this image type
     3. Text prompts don't match BioCLIP training distribution

Next steps if variance < 0.001:
1. Try the sanity check prompts above
   - If those work (variance > 0.01), BioCLIP is fine but doesn't know crops
   - If those also fail, something is wrong with model loading
2. Try a regular CLIP model (openai/clip-vit-base-patch32)
3. Check if image needs preprocessing (resize, crop, etc.)
""")
