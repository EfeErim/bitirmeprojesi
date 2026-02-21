# 🚀 Colab Quick Start - Plant Identification Pipeline

## Setup (4 Cells)

### **Cell 1: Clone Repository & Change Directory**
```python
!git clone https://github.com/EfeErim/bitirmeprojesi.git /content/bitirmeprojesi 2>/dev/null || !cd /content/bitirmeprojesi && git pull
%cd /content/bitirmeprojesi
print("✅ Repository ready")
```

### **Cell 2: Install Dependencies**
```python
!pip install --upgrade pip
!pip install transformers>=4.41.0
!pip install open-clip-torch
!pip install ultralytics
!pip install groundingdino-hf
!pip install huggingface-hub
print("✅ Dependencies installed")
```

### **Cell 3: Set HF_TOKEN from Secrets**
```python
from google.colab import userdata
import os

hf_token = userdata.get('HF_TOKEN')
os.environ['HF_TOKEN'] = hf_token
print("✅ HF_TOKEN configured")
```

🔑 **First time?** Go to Secrets (🔑 icon in left sidebar) and add:
- Name: `HF_TOKEN`
- Value: Your HuggingFace token from https://huggingface.co/settings/tokens

### **Cell 4: Run Pipeline**
```python
from google.colab import files
from src.router.vlm_pipeline import VLMPipeline
import json

# Load config
config = json.load(open('config/colab.json'))

# Create & load pipeline
pipeline = VLMPipeline(config)
print("🔄 Loading models...")
pipeline.load_models()
print(f"✅ Pipeline ready: {pipeline.actual_pipeline.upper()}")

# Upload plant image
print("\n📷 Upload your plant image...")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Analyze
print(f"\n🌱 Analyzing: {image_path}")
result = pipeline.analyze_image(image_path)

# Display results
print(f"\n✅ Done ({result.get('processing_time_ms', 0):.0f}ms)")
print(f"📊 Detected {len(result.get('detections', []))} regions\n")

for i, det in enumerate(result.get('detections', [])[:10], 1):
    crop = det.get('crop', 'unknown')
    crop_conf = det.get('crop_confidence', 0)
    part = det.get('part', 'unknown')
    part_conf = det.get('part_confidence', 0)
    
    print(f"Region {i}:")
    print(f"  🌾 CROP: {crop:20s} ({crop_conf:6.1%})")
    print(f"  🍃 PART: {part:20s} ({part_conf:6.1%})")
    print()
```

---

## What to Expect

### First Run
- ⏱️ **Time**: 2-5 minutes (downloading ~1-2 GB models)
- 📊 **Output**: `✅ Pipeline ready: SAM3` or `✅ Pipeline ready: DINO`
  - **SAM3**: Primary pipeline (+5.7% accuracy)
  - **DINO**: Fallback pipeline (if SAM3 unavailable)

### Results
```
✅ Done (2450.34ms)
📊 Detected 5 regions

Region 1:
  🌾 CROP: Grape                (94.32%)
  🍃 PART: Leaf                 (87.12%)

Region 2:
  🌾 CROP: Grape                (92.15%)
  🍃 PART: Stem                 (79.44%)
```

---

## Troubleshooting

### ❌ ModuleNotFoundError: No module named 'src'
**Solution**: Make sure you ran Cell 1 with `%cd /content/bitirmeprojesi`

### ❌ Authentication Error
**Solution**: 
1. Go to 🔑 Secrets in left sidebar
2. Add `HF_TOKEN` with your HuggingFace token
3. Restart the notebook

### ❌ Device Error / CUDA Out of Memory
**Solution**: Pipeline automatically falls back to CPU. If very slow, use Colab Pro+ for better GPU.

### ❌ Model Download Timeout
**Solution**: 
- First run takes 2-5 min (normal)
- Subsequent runs use cached models (<10 sec)
- If timeout, run Cell 4 again

---

## Next Steps

1. ✅ Upload your plant image (grape leaf, tomato, wheat, etc.)
2. ✅ View crop + part classifications with confidence
3. ✅ Inspect which pipeline loaded (SAM3 or DINO)
4. ✅ Save results for further analysis

**Remember**: This pipeline **identifies crop + plant part**, not disease.  
A human (or downstream model) diagnoses disease based on visual symptoms.

---

Commit: d455b6d (latest)
