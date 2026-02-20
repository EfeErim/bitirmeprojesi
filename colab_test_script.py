"""
BioCLIP-2 Preprocessing Test for Colab

Copy this entire script and paste it into a Colab cell. It will test
the preprocess_train vs preprocess_val fix.

Expected output:
- preprocess_train: strawberry (~25.3%) - RANDOM (the bug)
- preprocess_val: strawberry (>50%) - FIXED

If you see this pattern, the fix is working!
"""

# STEP 0: Install dependencies (can also use !pip in Colab)
import subprocess
import sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open-clip-torch", "torch", "torchvision", "pillow"])
print("✅ Dependencies installed\n")

# STEP 1: Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# STEP 2: Import and load BioCLIP-2
import open_clip
from PIL import Image
import urllib.request

print("Loading BioCLIP-2...")
model_id = "imageomics/bioclip-2"
hub_id = f"hf-hub:{model_id}"

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(hub_id)
tokenizer = open_clip.get_tokenizer(hub_id)

model = model.to(device)
model.eval()
print(f"✅ BioCLIP-2 loaded on {device}\n")

# STEP 3: Download test image
print("Downloading test image...")
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Fragaria_x_ananassa_Foto_by_CF_Weise.jpg/800px-Fragaria_x_ananassa_Foto_by_CF_Weise.jpg"
try:
    # Use different method for Python 3.10+
    import urllib.request
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')
    with urllib.request.urlopen(req) as response:
        image_data = response.read()
    import io
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    print(f"✅ Image downloaded: {image.size}\n")
except Exception as e:
    print(f"Download failed ({e}), using fallback...")
    # Try alternative URL
    try:
        url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Solanum_lycopersicum_var._cerasiforme_29.jpg/640px-Solanum_lycopersicum_var._cerasiforme_29.jpg"
        req = urllib.request.Request(url2)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req) as response:
            image_data = response.read()
        import io
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        print(f"✅ Tomato image downloaded: {image.size}\n")
    except:
        print(f"All downloads failed, using synthetic test image...")
        # Create a more interesting test image (gradient)
        import numpy as np
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:112, :, 0] = 200  # Red top half
        arr[112:, :, 1] = 200  # Green bottom half
        image = Image.fromarray(arr)
        print(f"Using gradient test image: {image.size}\n")

# STEP 4: Encode text labels
prompts = ["a photo of grape", "a photo of potato", "a photo of tomato", "a photo of strawberry"]
tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_embeds = model.encode_text(tokens)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

print(f"✅ Text embeddings encoded\n")

# DEBUG: Check if preprocess functions are actually different
print("="*70)
print("DEBUG: Checking if preprocess_train and preprocess_val are different")
print("="*70)
print(f"preprocess_train type: {type(preprocess_train)}")
print(f"preprocess_val type:   {type(preprocess_val)}")
print(f"Are they the same function? {preprocess_train == preprocess_val}")
print(f"preprocess_train: {preprocess_train}")
print(f"preprocess_val:   {preprocess_val}\n")

# STEP 5: Test TRAIN preprocessing (the old problematic one)
print("="*70)
print("Testing: preprocess_TRAIN (with augmentation)")
print("="*70)

img_tensor_train = preprocess_train(image).unsqueeze(0).to(device)
print(f"Image tensor: shape={img_tensor_train.shape}, min={img_tensor_train.min():.2f}, max={img_tensor_train.max():.2f}")
print(f"Tensor mean: {img_tensor_train.mean():.2f}, std: {img_tensor_train.std():.2f}")

with torch.no_grad():
    img_embeds = model.encode_image(img_tensor_train)
    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    logits = img_embeds @ text_embeds.T
    probs = torch.softmax(logits, dim=-1)

print(f"Logits: {logits.squeeze()}")
print(f"Probs:  {probs.squeeze()}")

idx_train = torch.argmax(probs).item()
conf_train = probs[0, idx_train].item()
label_train = prompts[idx_train].replace("a photo of ", "")

print(f"\n🎯 Prediction: {label_train} ({conf_train:.1%})")
print(f"Is random (all ~25%): {conf_train < 0.3}\n")

# STEP 6: Test VAL preprocessing (the fixed one)
print("="*70)
print("Testing: preprocess_VAL (no augmentation)")
print("="*70)

img_tensor_val = preprocess_val(image).unsqueeze(0).to(device)
print(f"Image tensor: shape={img_tensor_val.shape}, min={img_tensor_val.min():.2f}, max={img_tensor_val.max():.2f}")
print(f"Tensor mean: {img_tensor_val.mean():.2f}, std: {img_tensor_val.std():.2f}")

# Check if the tensors are actually different
print(f"\nAre preprocessed tensors different? {not torch.allclose(img_tensor_train, img_tensor_val)}")
if torch.allclose(img_tensor_train, img_tensor_val):
    print("⚠️ WARNING: Tensors are IDENTICAL! preprocess_train and preprocess_val produce the same output!")

with torch.no_grad():
    img_embeds = model.encode_image(img_tensor_val)
    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    logits = img_embeds @ text_embeds.T
    probs = torch.softmax(logits, dim=-1)

print(f"Logits: {logits.squeeze()}")
print(f"Probs:  {probs.squeeze()}")

idx_val = torch.argmax(probs).item()
conf_val = probs[0, idx_val].item()
label_val = prompts[idx_val].replace("a photo of ", "")

print(f"\n🎯 Prediction: {label_val} ({conf_val:.1%})")
print(f"Is working (>40%): {conf_val > 0.4}\n")

# STEP 7: Analysis
print("="*70)
print("FINAL RESULT")
print("="*70)

train_random = conf_train < 0.3
val_good = conf_val > 0.4

print(f"\npreprocess_train: {conf_train:.1%} - {'RANDOM (bug)' if train_random else 'working'}")
print(f"preprocess_val:   {conf_val:.1%} - {'GOOD (fixed)' if val_good else 'needs check'}")

# Check tensor equivalence
same_tensors = torch.allclose(img_tensor_train, img_tensor_val)
print(f"\nTensors are identical: {same_tensors}")

if same_tensors:
    print("\n⚠️ IMPORTANT: preprocess_train and preprocess_val produced identical tensors!")
    print("This means they might be the SAME function, or the image has no features.")
    print("\nPossible issues:")
    print("1. Gradient test image has no distinctive features")
    print("2. preprocess_train and preprocess_val are actually the same")
    print("3. BioCLIP encodings are all producing similar values")
elif train_random and val_good:
    print("\n✅✅✅ FIX CONFIRMED! ✅✅✅")
    print(f"\npreprocess_train shows {conf_train:.1%} - this is the OLD BUG")
    print(f"preprocess_val shows {conf_val:.1%} - this is FIXED")
    print("\nConclusion: Using preprocess_val in the code is CORRECT!")
elif val_good:
    print(f"\n✅ preprocess_val is working better than preprocess_train")
    print(f"Improvement: {conf_val - conf_train:.1%}")
else:
    print("⚠️ Results unclear - check the logits and tensor stats above")

print("\n" + "="*70)
print("TEST COMPLETE - Copy the results above and share them!")
print("="*70)
