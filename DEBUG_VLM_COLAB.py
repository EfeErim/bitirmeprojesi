#!/usr/bin/env python3
"""
🐛 VLM PIPELINE DEBUG GUIDE - TRY THIS IN COLAB

This script provides step-by-step debugging of the VLM pipeline on GPU.
Copy each cell into Colab and run in order to diagnose image encoding issues.
"""

# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================
"""
!nvidia-smi
!pip install -q torch torchvision transformers
!pip install -q git+https://github.com/mlfoundations/open_clip.git
"""

# ============================================================================
# CELL 2: Minimal BioCLIP-2 Direct Test
# ============================================================================
"""
import torch
import logging
from PIL import Image
import urllib.request
import open_clip

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check GPU
print(f"GPU: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load BioCLIP-2 directly
print("\\nLoading BioCLIP-2...")
model_id = "imageomics/bioclip-2"
hub_model_id = f"hf-hub:{model_id}"

print(f"Model ID: {hub_model_id}")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
tokenizer = open_clip.get_tokenizer(hub_model_id)

model = model.to(device)
model.eval()
print(f"✅ BioCLIP-2 loaded on {device}")

# Text encoding test
print("\\n" + "="*60)
print("TEXT ENCODING TEST")
print("="*60)
prompts = ["a photo of grape", "a photo of potato", "a photo of tomato", "a photo of strawberry"]
tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_embeds = model.encode_text(tokens)
    print(f"Text embeddings shape: {text_embeds.shape}")
    print(f"Text embeddings norm: {text_embeds.norm(dim=-1)}")
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

# Download test image
print("\\n" + "="*60)
print("IMAGE DOWNLOAD")
print("="*60)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Fragaria_x_ananassa_Foto_by_CF_Weise.jpg/800px-Fragaria_x_ananassa_Foto_by_CF_Weise.jpg"
try:
    urllib.request.urlretrieve(url, "/tmp/strawberry.jpg")
    image = Image.open("/tmp/strawberry.jpg").convert('RGB')
    print(f"✅ Image loaded: {image.size}")
except Exception as e:
    print(f"Download failed: {e}")
    image = Image.new('RGB', (224, 224), color=(200, 0, 0))
    print(f"Using red test image")

# Image encoding test (CRITICAL)
print("\\n" + "="*60)
print("IMAGE ENCODING TEST (CRITICAL - THIS IS THE BUG)")
print("="*60)

# Test BOTH preprocess_train and preprocess_val
for preprocess_name, preprocess_fn in [("train", preprocess_train), ("val", preprocess_val)]:
    print(f"\\n--- Testing preprocess_{preprocess_name} ---")
    
    image_tensor = preprocess_fn(image).unsqueeze(0).to(device)
    print(f"After preprocessing: shape={image_tensor.shape}, dtype={image_tensor.dtype}")
    print(f"  Min: {image_tensor.min():.3f}, Max: {image_tensor.max():.3f}")
    print(f"  Mean: {image_tensor.mean():.3f}, Std: {image_tensor.std():.3f}")
    
    with torch.no_grad():
        image_embeds = model.encode_image(image_tensor)
        print(f"Image embeddings shape: {image_embeds.shape}")
        print(f"Image embeddings norm (before): {image_embeds.norm(dim=-1)}")
        
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        print(f"Image embeddings norm (after): {image_embeds.norm(dim=-1)}")
        
        # Compute similarity
        logits = image_embeds @ text_embeds.T
        probs = torch.softmax(logits, dim=-1)
        
        print(f"Logits: {logits.squeeze()}")
        print(f"Probabilities: {probs.squeeze()}")
        
        pred_idx = torch.argmax(probs)
        pred_conf = probs[0, pred_idx].item()
        pred_label = prompts[pred_idx].replace("a photo of ", "")
        print(f"Prediction: {pred_label} ({pred_conf:.1%})")
"""

# ============================================================================
# CELL 3: If Prediction is Wrong (25% per class = random), Check These:
# ============================================================================
"""
# KEY DIAGNOSTICS:
# 1. Image tensor min/max should be in [-1, 1] or [0, 1] after preprocessing
#    If not: preprocessing is broken
#
# 2. Image embeddings norm (before normalization) should NOT be ~0.001
#    If ~0.001: encoder is broken or image is being ignored
#
# 3. Image embeddings norm (after normalization) should be 1.0
#    If not 1.0: normalization failed
#
# 4. Logits should vary significantly between classes
#    If all ~0.63: features are identical = BUG
#
# 5. Probabilities should NOT be ~25% for all 4 classes
#    If ~25%: prediction is random, encoder broken

print("\\nCOMPON ENT DIAGNOSTICS:")
print(f"preprocess_train == preprocess_val: {preprocess_train == preprocess_val}")
print(f"Model on {device}: {next(model.parameters()).device}")
print(f"Model in eval mode: {model.training == False}")
"""

# ============================================================================
# CELL 4: Test Full VLM Pipeline if BioCLIP-2 test passes
# ============================================================================
"""
import sys
sys.path.insert(0, '/content/gdrive/MyDrive/bitirme_projesi')  # Adjust path

# Clone repo or import:
# git clone https://github.com/your-repo/aads-ulora.git
# sys.path.insert(0, '/content/aads-ulora')

from src.router.vlm_pipeline import VLMPipeline

config = {
    'vlm_enabled': True,
    'vlm_confidence_threshold': 0.5,
    'vlm_max_detections': 10,
    'vlm_strict_model_loading': False,
}

pipeline = VLMPipeline(config, device='cuda' if torch.cuda.is_available() else 'cpu')
pipeline.load_models()

# Test  
if pipeline.is_ready():
    print("✅ Pipeline ready")
    crop_label, conf = pipeline._classify_with_preencoded(image, 'crop')
    print(f"Classification: {crop_label} ({conf:.1%})")
else:
    print("❌ Pipeline not ready")
"""

# ============================================================================
# ROOT CAUSE CHECKLIST
# ============================================================================
"""
If predictions show ~25% confidence for all classes (random guessing):

❌ LIKELY CAUSES:
1. Using preprocess_train instead of preprocess_val
   - Fix: Change to preprocess_val (already done in latest code)

2. Image tensor has wrong value range after preprocessing
   - Check: Print min/max values after preprocessing
   - Should be in [-1, 1] or [0, 1], NOT [0, 255]

3. Model encoder broken or not being called
   - Check: What's the norm of embeddings before normalization?
   - Should be > 1.0 for valid features, not ~0.001

4. Running on CPU instead of GPU
   - Check: torch.cuda.is_available()
   - Some models have precision issues on CPU

5. Wrong BioCLIP model ID
   - Should be: "imageomics/bioclip-2" (already fixed)
   - Not: "imageomics/bioclip"

6. Tokenizer not compatible with BioCLIP-2
   - Check: tokenizer = open_clip.get_tokenizer(hub_model_id)
   - This should auto-load correct tokenizer

✅ WHAT TO SHARE:
When you run the test above and see wrong predictions, share:
1. Output of "Image tensor min/max" after preprocessing
2. Output of "Image embeddings norm (before)" - should be > 1.0
3. Output of "Logits:" and "Probabilities:" - should vary, not all ~0.25
4. GPU/device info from nvidia-smi
5. The error message if any step fails
"""
