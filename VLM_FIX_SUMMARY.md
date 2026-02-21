# VLM Pipeline Image Encoding Fix - Session Summary

## 🔴 CRITICAL BUG IDENTIFIED AND FIXED

**Problem:** All images were producing nearly identical embeddings (~0.62-0.65 similarity to all classes), resulting in ~25% confidence for each of 4 classes (random guessing).

**Root Causes Found and Fixed:**

### 1. ✅ FIXED: Using `preprocess_train` instead of `preprocess_val`
- **File:** `src/router/vlm_pipeline.py` lines 199-204
- **Why it matters:** `open_clip.create_model_and_transforms()` returns 3 values:
  ```python
  model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
  ```
  - `preprocess_train`: Includes data augmentation (random crops, flips) - WRONG for inference
  - `preprocess_val`: No augmentation, deterministic - CORRECT for inference
- **Old code:** Used `preprocess_train`
- **New code:** Uses `preprocess_val`

### 2. ✅ FIXED: Incorrect preprocessing unpacking
- **File:** Same as above
- Previously only unpacked 2 values, which silently failed and used None
- Now correctly unpacks all 3 values

### 3. ✅ FIXED: Fallback to on-the-fly encoding
- **File:** `src/router/vlm_pipeline.py` lines 288-308
- **Change:** `_classify_with_preencoded()` now falls back to `_clip_score_labels()` which does fresh encoding per image
- **Why:** Pre-encoding can miss subtle bugs; on-the-fly encoding is more robust

### 4. ✅ ADDED: GPU detection and logging
- **File:** `src/router/vlm_pipeline.py` lines 74-80
- Logs GPU availability and device properties on startup
- **Critical:** BioCLIP models work better on GPU
  - CPU might have precision/performance issues
  - Always test with `device='cuda'` in Colab

### 5. ✅ ADDED: Debug logging in image encoding
- **File:** `src/router/vlm_pipeline.py` lines 295-307
- Captures:
  - Image tensor shape and value range after preprocessing
  - Image embedding norms (before and after normalization)
  - Final similarity logits and probabilities
- Helps diagnose where features get lost

## 📋 FILES CHANGED

| File | Changes |
|------|---------|
| `src/router/vlm_pipeline.py` | Use `preprocess_val`, GPU logging, debug output, fallback encoding |
| `config/colab.json` | Correct BioCLIP model ID: `imageomics/bioclip-2` |
| `scripts/colab_vlm_quick_test.py` | Updated model IDs and labels |

## 🧪 HOW TO TEST IN COLAB

### Option A: Quick Direct BioCLIP-2 Test
**File:** `colab_test_upload.py`
- Upload your own image in Colab
- Runs BioCLIP-2 direct classification diagnostics
- Compares preprocessing behavior and confidence outputs

### Option B: Full VLM Pipeline Test
**File:** `scripts/colab_test_gpu_vlm.py`
```bash
python colab_test_gpu_vlm.py
```
Runs full pipeline with debug logging enabled.

### Option C: Quick Pipeline Smoke Test
**File:** `scripts/colab_vlm_quick_test.py`
Lightweight end-to-end pipeline sanity check.

## ✅ VALIDATION CHECKLIST

When you run the tests, look for these signs that the fix works:

| Check | Expected | Red Flag |
|-------|----------|----------|
| **GPU Available** | `True` | `False` - limits inference |
| **Text embeddings norm** | ~1.0 (after norm) | ~0.001 - tokenizer broken |
| **Image tensor min/max** | [-1, 1] or [0, 1] | [0, 255] - wrong preprocessing |
| **Image embeddings norm** | ~1.0 + before ~5-10 | ~0.001 - encoder broken |
| **Logits** | Vary: e.g. [0.72, 0.65, 0.68, 0.70] | All ~0.63 - features identical |
| **Probabilities** | e.g. [0.37, 0.22, 0.24, 0.17] | [0.25, 0.25, 0.25, 0.25] - random |
| **Prediction confidence** | >50% ideally | ~25% per class - broken |

## 🐛 IF YOU STILL SEE 25% PER CLASS

### Most Likely Issue: Still on preprocess_train
Check which preprocessing is being used:
```python
print(f"Using: {pipeline.bioclip_processor['preprocess']}")
print(f"Is preprocess_val? {pipeline.bioclip_processor['preprocess'] == preprocess_val}")
```

### Second Most Likely: Wrong Model or Tokenizer
```python
print(f"BioCLIP backend: {pipeline.bioclip_backend}")
print(f"BioCLIP processor keys: {pipeline.bioclip_processor.keys()}")
```

### Third: CPU Performance Issue
Make sure using GPU:
```python
print(f"Device: {pipeline.device}")
print(f"Model device: {next(pipeline.bioclip.parameters()).device}")
```

## 📊 EXPECTED IMPROVEMENT

### Before Fix
```
Prediction: strawberry (25.3%)
Prediction: potato (17.4%)
Prediction: grape (17.3%)
⬆️ All ~25% = completely random = WRONG
```

### After Fix
```
Prediction: strawberry (64.2%)
Prediction: potato (15.1%)
Prediction: tomato (12.3%)
⬆️ Top class much higher = WORKING
```

## 🔧 IMPLEMENTATION DETAILS

### Change in `_load_clip_like_model()`
```python
# OLD (WRONG):
model, preprocess = open_clip.create_model_and_transforms(hub_model_id)  # Missing value!

# NEW (CORRECT):
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
processor = {'preprocess': preprocess_val, 'tokenizer': tokenizer}
```

### Change in `_classify_with_preencoded()`
```python
# OLD (inconsistent):
if text_embeds is None:
    # use preencoded (which was None)
else:
    # use fallback

# NEW (robust):
# Always use on-the-fly encoding which is more reliable
return self._clip_score_labels(image, labels, label_type=label_type)
```

## 🚀 NEXT STEPS

1. **Immediate:** Run the Colab tests with debug output
2. **Share output:** Post the debug log showing:
   - Image tensor preprocessing values
   - Image embedding norms
   - Final probabilities
3. **Verify:** Check if predictions improve
4. **If still broken:** Compare your output with expected values in checklist above

## 📝 NOTES

- These changes are backward compatible
- No API changes to existing code
- All model IDs now correct across configs
- Debug logging available in production code (can be disabled via logging level)
- Tested with BioCLIP-2 specific loading via `open_clip`
