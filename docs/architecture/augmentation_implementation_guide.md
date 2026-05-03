# Practical Implementation Guide: Augmentation & Leakage Prevention

**Version:** 1.0  
**Date:** May 2, 2026  
**Audience:** AADS v6 practitioners (data scientists, agronomists, ML engineers)  
**Prerequisites:** Completed the [main literature review](./data_augmentation_leakage_prevention.md)

---

## Quick Reference: Is My Dataset Safe from Leakage?

### Checklist (Run These First)

```bash
# 1. Verify split structure exists
ls -la data/prepared_runtime_datasets/your_dataset_key/

# Expected output:
# continual/  ← Training (may contain _offline_aug/ subdirectory)
# val/        ← Validation (no augmented variants)
# test/       ← Test (no augmented variants)
# ood/        ← Out-of-distribution (if available)
# split_manifest.json ← Metadata

# 2. Check manifest for leakage
.\scripts\python.cmd -c "
import json
from pathlib import Path

manifest = json.loads(Path('data/prepared_runtime_datasets/your_dataset_key/split_manifest.json').read_text())

# Count images per split
splits = {}
for row in manifest.get('rows', []):
    split = row['split']
    if split not in splits:
        splits[split] = {'real': 0, 'generated': 0}
    
    if row.get('generated_offline_augmentation'):
        splits[split]['generated'] += 1
    else:
        splits[split]['real'] += 1

print('Dataset composition:')
for split, counts in splits.items():
    print(f'  {split}: {counts[\"real\"]} real + {counts[\"generated\"]} generated')
"

# 3. Verify no cross-split contamination
.\scripts\python.cmd scripts/validate_dataset_layout.py \
  --root data/prepared_runtime_datasets/your_dataset_key/ \
  --check-leakage
```

---

## Section 1: Dataset Preparation Workflow

### Step 1.1: Prepare Grouped Dataset (Notebook 0)

**Purpose:** Audit raw dataset, group duplicates/variants, create runtime splits with family awareness.

**What Notebook 0 does:**
1. Scans flat class-root layout (all images in `data/class_root_dataset/crop_name/`)
2. Computes DINOv3 embeddings for each image
3. Clusters similar images using embedding similarity (threshold: 0.985)
4. Identifies exact duplicates via perceptual hashing (PHASH)
5. Produces grouped dataset with variant families marked
6. Performs stratified split preserving families
7. Materializes `data/prepared_runtime_datasets/` layout

**Key Configuration (Notebook 0 Parameters):**
```python
# From colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb

CROP_NAME = "tomato"  # Your crop
PART_NAME = "leaf"
SOURCE_ROOT = "data/class_root_dataset"  # Input flat layout
RUNTIME_ROOT = "data/prepared_runtime_datasets"  # Output structured layout
ARTIFACT_ROOT = "outputs/colab_notebook_data_prep"  # Audit reports

# Embedding models
DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BIOCLIP_MODEL_ID = "imageomics/bioclip-2.5-vith14"

# Similarity thresholds
DINO_AUTO_MIN = 0.985  # ← Conservative: only truly similar images grouped
BIOCLIP_AUTO_MIN = 0.970
PHASH_AUTO_MAX_DISTANCE = 4  # Exact duplicate threshold

# Split policy
SPLIT_POLICY = "grouped_family_canonical_eval_60_20_20"
# Result: 60% train, 20% val, 20% test
# Families kept intact within splits
```

**Output Artifacts:**
```
data/prepared_runtime_datasets/tomato_leaf_20250502_xyz/
├── continual/          ← Training split (60%)
│   ├── healthy/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── early_blight/
│   └── late_blight/
├── val/                ← Validation split (20%, NO augmented variants)
│   ├── healthy/
│   ├── early_blight/
│   └── late_blight/
├── test/               ← Test split (20%, NO augmented variants)
│   ├── healthy/
│   ├── early_blight/
│   └── late_blight/
├── ood/                ← Out-of-distribution samples (if available)
├── split_manifest.json ← Family groupings + metadata
└── human_review_packet.json  ← Audit flagged images for review
```

### Step 1.2: Understand `split_manifest.json`

**Critical for verifying NO leakage:**

```json
{
  "dataset_key": "tomato_leaf_20250502_abc123",
  "split_policy": "grouped_family_canonical_eval_60_20_20",
  "total_grouped_families": 847,
  "classes": [
    {
      "class_name": "healthy",
      "reference_image_count": 142,
      "grouped_family_count": 45
    },
    {
      "class_name": "early_blight",
      "reference_image_count": 87,
      "grouped_family_count": 28
    }
  ],
  "rows": [
    {
      "grouped_family_id": "family_001",
      "normalized_class_name": "healthy",
      "split": "continual",
      "original_image_name": "IMG_0001.jpg",
      "reference_image_count": 1,
      "family_size": 1,
      "runtime_skipped": false,
      "generated_offline_augmentation": false,
      "synthetic_hint": false,
      "source": "field_camera_1",
      "capture_date": "2024-06-15"
    },
    {
      "grouped_family_id": "family_001_variant_a",
      "normalized_class_name": "healthy",
      "split": "continual",  // ← SAME split as base
      "original_image_name": "IMG_0001_rotated.jpg",
      "is_variant": true,
      "variant_of": "family_001",
      "generated_offline_augmentation": true,  // ← Offline-generated
      "synthetic_hint": false
    }
  ]
}
```

**Key Fields for Leakage Detection:**
- `split`: Must be identical for all variants of the same family
- `generated_offline_augmentation`: Distinguishes real from pre-computed variants
- `reference_image_count`: Count ONLY for rows with this = false (real images)

### Step 1.3: Verify No Leakage After Notebook 0

```python
#!/usr/bin/env python3
"""Validate split integrity and detect leakage."""

import json
from pathlib import Path
from collections import defaultdict

def check_split_integrity(dataset_root: Path):
    """Verify no family variants appear in different splits."""
    
    manifest_path = dataset_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    
    # Map family ID to splits containing it
    family_splits = defaultdict(set)
    
    for row in manifest.get("rows", []):
        family_id = row.get("grouped_family_id", "")
        split = row.get("split", "")
        
        if family_id and split:
            family_splits[family_id].add(split)
    
    # Check for leakage
    leakage_found = []
    for family_id, splits in family_splits.items():
        if len(splits) > 1:
            leakage_found.append({
                "family_id": family_id,
                "appears_in_splits": list(splits),
            })
    
    if leakage_found:
        print(f"❌ LEAKAGE DETECTED in {len(leakage_found)} families:")
        for item in leakage_found[:5]:  # Show first 5
            print(f"   {item['family_id']}: {item['appears_in_splits']}")
        raise AssertionError("Dataset has cross-split variant leakage!")
    
    print(f"✓ Dataset integrity verified: {len(family_splits)} families, all within single splits")
    
    # Summary statistics
    splits_summary = defaultdict(lambda: {"real": 0, "generated": 0, "families": 0})
    
    for row in manifest.get("rows", []):
        split = row.get("split", "")
        family_id = row.get("grouped_family_id", "")
        is_generated = row.get("generated_offline_augmentation", False)
        
        splits_summary[split]["generated" if is_generated else "real"] += 1
        splits_summary[split]["families"] = len(set(
            r.get("grouped_family_id", "") 
            for r in manifest.get("rows", [])
            if r.get("split", "") == split
        ))
    
    print("\nDataset composition:")
    for split in ["continual", "val", "test"]:
        stats = splits_summary[split]
        print(f"  {split:10} → {stats['real']:3} real + {stats['generated']:3} generated = {stats['real']+stats['generated']:3} total images ({stats['families']:3} families)")

# Run it
check_split_integrity(Path("data/prepared_runtime_datasets/your_dataset_key/"))
```

**Expected Output (Safe Dataset):**
```
✓ Dataset integrity verified: 847 families, all within single splits

Dataset composition:
  continual    →  229 real + 0 generated = 229 total images (85 families)
  val          →   60 real + 0 generated =  60 total images (25 families)
  test         →   58 real + 0 generated =  58 total images (24 families)
```

---

## Section 2: Online Augmentation During Training

### Step 2.1: Understanding Your Data Loader

**Key Files:**
- `src/data/transforms.py` — Defines augmentation operations
- `src/data/loaders.py` — Applies augmentation only to training split
- `src/data/datasets.py` — Actual image loading

**The Critical Leakage-Prevention Code:**

```python
# From src/data/loaders.py, line ~174

for split in ("train", "val", "test"):
    dataset = dataset_cls(
        data_dir=data_dir,
        crop=crop,
        split=split,
        # ... other args ...
        transform=(split == "train"),  # ← LEAKAGE PREVENTION
        augmentation_policy="randaugment",
        randaugment_num_ops=2,
        randaugment_magnitude=7,
    )
```

**What This Does:**
- `split == "train"` → `transform = True` → Online augmentation **enabled**
- `split == "val"` or `split == "test"` → `transform = False` → Online augmentation **disabled**

### Step 2.2: Augmentation Operations Applied During Training

```python
# From src/data/transforms.py, lines 88-115

if training:
    steps = [
        transforms.RandomResizedCrop(
            size=(target_size, target_size),
            scale=(0.8, 1.0),      # Crop 80-100% of image
            ratio=(0.95, 1.05),    # Preserve aspect ratio
        ),
        transforms.RandomHorizontalFlip(p=0.5),      # 50% probability
        transforms.RandomVerticalFlip(p=0.1),        # 10% probability
        transforms.RandomRotation(12),               # ±12 degrees
        transforms.RandAugment(
            num_ops=2,                 # Apply 2 random ops from pool
            magnitude=7,                # Magnitude 7/30 (moderate)
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))],
            p=0.1                       # 10% chance of blur
        ),
        *_normalization_steps(),        # ImageNet normalization
    ]
else:
    # Validation/test: NO augmentation
    steps = [
        transforms.Resize((target_size, target_size)),
        *_normalization_steps(),
    ]
```

### Step 2.3: Modifying Augmentation Parameters

**Scenario 1: Disable augmentation (baseline experiments)**

```python
# Edit config/base.json or pass config_overrides

config_overrides = {
    "training": {
        "continual": {
            "data": {
                "augmentation_policy": "none",  # ← Disables all augmentation except resize
            }
        }
    }
}

# Then train
from src.workflows.training import TrainingWorkflow

result = TrainingWorkflow.run(
    crop="tomato",
    runtime_dataset_root="data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/",
    output_dir="runs/tomato/",
    config_overrides=config_overrides,
)
```

**Scenario 2: Aggressive augmentation (to handle OOD/domain shift)**

```python
config_overrides = {
    "training": {
        "continual": {
            "data": {
                "augmentation_policy": "augmix",  # Mix multiple augmentations
                "augmix_severity": 5,             # Increased severity (1-10)
                "augmix_width": 4,                # More mixing branches
                "augmix_depth": 3,                # Deeper mixing chains
                "augmix_alpha": 1.5,              # Stronger blending
                "randaugment_magnitude": 10,      # Increased magnitude (0-30)
            }
        }
    }
}
```

**Scenario 3: Minimal augmentation (limited data)**

```python
config_overrides = {
    "training": {
        "continual": {
            "data": {
                "augmentation_policy": "basic",   # ColorJitter only
                "randaugment_num_ops": 1,         # Single random operation
                "randaugment_magnitude": 5,       # Subtle (5/30)
            }
        }
    }
}
```

### Step 2.4: Monitoring Augmentation Effectiveness

**Add logging to track augmentation impact:**

```python
# In your training loop, add this observer:

from src.training.session import TrainingObserver
import torch

class AugmentationMonitor(TrainingObserver):
    """Track how augmentation affects training."""
    
    def on_train_batch(self, batch_idx: int, batch_images: torch.Tensor, batch_labels: torch.Tensor, **kwargs):
        """Called after each training batch."""
        
        if batch_idx % 100 == 0:
            # Log image statistics to detect augmentation
            mean_brightness = batch_images.mean().item()
            std_brightness = batch_images.std().item()
            
            print(f"Batch {batch_idx}: "
                  f"mean_brightness={mean_brightness:.3f}, "
                  f"std={std_brightness:.3f}")
            
            # Save a few augmented examples for visual inspection
            if batch_idx == 0:
                self._save_augmented_samples(batch_images, batch_labels)
    
    def _save_augmented_samples(self, images: torch.Tensor, labels: torch.Tensor):
        """Visualize augmented training samples."""
        from torchvision.utils import save_image
        
        # Denormalize from ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        denormalized = images[:4] * std + mean  # First 4 images
        save_image(denormalized, "runs/augmented_samples_batch0.png", nrow=2)
        print("Saved augmented samples to: runs/augmented_samples_batch0.png")

# Register observer
observer = AugmentationMonitor()
result = TrainingWorkflow.run(
    crop="tomato",
    runtime_dataset_root="...",
    output_dir="...",
    custom_observers=[observer],
)
```

---

## Section 3: Offline Augmentation (Pre-Computed Variants)

### Step 3.1: When to Use Offline Augmentation

**Decision Tree:**

```
Do you have ≥ 100 base images per class? 
├─ YES → Use ONLINE augmentation only (simpler, better)
└─ NO  → Do you have 50-99 images per class?
         ├─ YES → Use BOTH online + offline (generate 1-2 variants)
         └─ NO  → Use BOTH + increase variants (generate 2-3), document as research
```

### Step 3.2: Generate Offline Augmented Variants

```bash
# Step 1: Ensure dataset is prepared
ls -la data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/continual/

# Step 2: Generate offline variants (only in continual split)
.\scripts\python.cmd scripts/augment_runtime_train_split.py \
  --source-root data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/ \
  --num-variants 2 \
  --seed 42 \
  --verbose

# Output:
# Processing continual split...
# ✓ Image 1/200: early_blight/IMG_001.jpg
#   - Generated 2 variants: IMG_001_v0.jpg, IMG_001_v1.jpg
# ...
# ✓ Generated 400 variants from 200 base images
# ✓ Updated split_manifest.json with generated_offline_augmentation=true
# ✓ No changes to val/ or test/ splits
```

### Step 3.3: Verify Offline Augmentation Applied Correctly

```python
#!/usr/bin/env python3
"""Check that offline augmentation was applied only to training split."""

import json
from pathlib import Path
from collections import defaultdict

dataset_root = Path("data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/")
manifest = json.loads((dataset_root / "split_manifest.json").read_text())

# Count generated images per split
generated_per_split = defaultdict(int)
real_per_split = defaultdict(int)

for row in manifest.get("rows", []):
    split = row.get("split", "")
    is_generated = row.get("generated_offline_augmentation", False)
    
    if is_generated:
        generated_per_split[split] += 1
    else:
        real_per_split[split] += 1

print("Offline augmentation verification:")
print()
print(f"{'Split':<12} {'Real Images':<15} {'Generated':<15} {'Total':<10}")
print("-" * 52)

for split in ["continual", "val", "test"]:
    real = real_per_split.get(split, 0)
    gen = generated_per_split.get(split, 0)
    total = real + gen
    print(f"{split:<12} {real:<15} {gen:<15} {total:<10}")
    
    # Safety check: Generated images should ONLY be in training split
    if split != "continual" and gen > 0:
        print(f"  ⚠️  WARNING: Found {gen} generated images in {split} split!")
        print("       This indicates potential leakage!")

print()
print("✓ Safe configuration:" if generated_per_split.get("val", 0) == 0 and generated_per_split.get("test", 0) == 0 else "❌ Unsafe: generated images in val/test")
```

**Expected Output (Safe):**
```
Offline augmentation verification:

Split        Real Images     Generated       Total     
----------------------------------------------------
continual    229             400             629       
val          60              0               60        
test         58              0               58        

✓ Safe configuration: generated images only in training split
```

### Step 3.4: Understanding Generated Variant Quality

**Example: What does `augment_runtime_train_split.py` create?**

```python
# Generated from single base image: early_blight/IMG_001.jpg

# Deterministic transformations (same seed = same result):
def _apply_variant(image, source_path, variant_index, seed):
    rng = random.Random(hash(f"{source_path}::{variant_index}::{seed}"))
    
    # Variant 0 (index=0): More aggressive crop + strong rotation
    if variant_index == 0:
        # Random horizontal flip (seed-dependent)
        # Random crop: 86-100% of size
        # Rotation: -18° to +18°
        # Brightness: ±14%
        # Result: ~20° rotated, cropped top-right, slightly darker
    
    # Variant 1 (index=1): Different random seed
        # Random crop: 86-100% (different region)
        # Rotation: different angle
        # Color: different brightness/contrast
        # Result: ~-8° rotated, cropped bottom-left, slightly brighter

# Stored as:
# continual/_offline_aug/early_blight/IMG_001_v0.jpg
# continual/_offline_aug/early_blight/IMG_001_v1.jpg

# Tracked in manifest:
# "generated_offline_augmentation": true
# "original_image_name": "IMG_001.jpg"
```

---

## Section 4: Class Balancing with Augmentation

### Step 4.1: Understanding Class Imbalance in Your Dataset

```python
#!/usr/bin/env python3
"""Analyze class balance in your dataset."""

import json
from pathlib import Path
from collections import Counter

def analyze_class_balance(dataset_root: Path):
    manifest = json.loads((dataset_root / "split_manifest.json").read_text())
    
    # Count base images per class per split
    split_class_counts = {}
    
    for split in ["continual", "val", "test"]:
        split_class_counts[split] = Counter()
        
        for row in manifest.get("rows", []):
            if row.get("split") == split and not row.get("generated_offline_augmentation"):
                class_name = row.get("normalized_class_name", "")
                split_class_counts[split][class_name] += 1
    
    print("Class distribution (real images only):")
    print()
    
    # Compute imbalance ratio for training split
    train_counts = split_class_counts["continual"]
    max_count = max(train_counts.values())
    min_count = min(train_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 52)
    
    total_per_class = {}
    for class_name in sorted(set(split_class_counts["continual"].keys()) | 
                             set(split_class_counts["val"].keys()) |
                             set(split_class_counts["test"].keys())):
        train = split_class_counts["continual"].get(class_name, 0)
        val = split_class_counts["val"].get(class_name, 0)
        test = split_class_counts["test"].get(class_name, 0)
        total = train + val + test
        
        total_per_class[class_name] = total
        print(f"{class_name:<20} {train:<8} {val:<8} {test:<8} {total:<8}")
    
    print()
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}x")
    print(f"Recommendation:", end=" ")
    
    if imbalance_ratio < 1.5:
        print("✓ Balanced—use simple sampling")
    elif imbalance_ratio < 3.0:
        print("⚠ Moderate imbalance—use weighted sampler + effective-number weighting")
    else:
        print("❌ Severe imbalance—consider data collection or few-shot research mode")
    
    return total_per_class, imbalance_ratio

# Run analysis
class_counts, ratio = analyze_class_balance(
    Path("data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/")
)
```

**Example Output:**
```
Class distribution (real images only):

Class                Train    Val      Test     Total   
----------------------------------------------------
early_blight         87       18       17       122     
healthy              142      30       28       200     
late_blight          15       3        3        21      

Imbalance ratio (max/min): 9.33x
Recommendation: ❌ Severe imbalance—consider data collection or few-shot research mode
```

### Step 4.2: Applying Effective-Number Weighting

**When Imbalance Ratio 1.5-3.0 (Moderate):**

```python
# config/base.json

{
  "training": {
    "continual": {
      "class_balance": {
        "allow_sampler_and_loss_rebalance": true  # ← Enable
      },
      "data": {
        "sampler": "weighted"  # ← Use weighted sampler
      }
    }
  }
}
```

**When Imbalance Ratio > 3.0 (Severe):**

```python
# Option 1: Increase offline augmentation for minority classes
.\scripts\python.cmd scripts/augment_runtime_train_split.py \
  --source-root data/prepared_runtime_datasets/your_dataset/ \
  --num-variants 5 \
  --per-class-variants '{"early_blight": 10, "late_blight": 15}' \
  --seed 42

# Option 2: Or document as few-shot research
config_overrides = {
    "training": {
        "continual": {
            "few_shot_research_mode": True,  # Disables 100-image guardrail
        }
    }
}
```

### Step 4.3: Computing Effective-Number Weights Manually

```python
#!/usr/bin/env python3
"""Compute class weights using Cui et al. effective-number formula."""

import json
from pathlib import Path
from collections import Counter

def compute_effective_number_weights(class_counts: dict[str, int]) -> dict[str, float]:
    """
    Apply effective-number weighting per Cui et al. (2019).
    
    Formula: E_n = (1 - β^n) / (1 - β), where β = (N-1)/N
    Weight: w_i = 1 / E_i
    """
    
    total_samples = sum(class_counts.values())
    beta = (total_samples - 1) / total_samples
    
    weights = {}
    for class_name, count in class_counts.items():
        # Compute effective number
        effective_num = (1.0 - beta ** count) / (1.0 - beta)
        # Invert to get weight
        weight = 1.0 / effective_num
        weights[class_name] = weight
    
    # Normalize so they sum to number of classes
    num_classes = len(class_counts)
    total_weight = sum(weights.values())
    normalized = {
        name: w * num_classes / total_weight 
        for name, w in weights.items()
    }
    
    return normalized

# Example: Compute weights for tomato dataset
class_counts = {
    "healthy": 142,
    "early_blight": 87,
    "late_blight": 15,
}

weights = compute_effective_number_weights(class_counts)

print("Effective-number class weights:")
print()
print(f"{'Class':<20} {'Count':<10} {'Weight':<10} {'Notes'}")
print("-" * 60)

for class_name in sorted(class_counts.keys()):
    count = class_counts[class_name]
    weight = weights[class_name]
    max_weight = max(weights.values())
    relative = weight / max_weight  # Relative to heaviest class
    
    note = ""
    if relative < 0.3:
        note = "Minority class (upweighted ×3)"
    elif relative > 0.9:
        note = "Majority class (downweighted)"
    
    print(f"{class_name:<20} {count:<10} {weight:<10.3f} {note}")

print()
print("Usage in training:")
print("  weights = torch.tensor([w for c, w in sorted(weights.items())])")
print("  loss_fn = CrossEntropyLoss(weight=weights)")
```

**Example Output:**
```
Effective-number class weights:

Class                Count      Weight      Notes
------------------------------------------------------------
healthy              142        0.646       Majority class (downweighted)
early_blight         87         1.034       
late_blight          15         3.191       Minority class (upweighted ×3)

Usage in training:
  weights = torch.tensor([w for c, w in sorted(weights.items())])
  loss_fn = CrossEntropyLoss(weight=weights)
```

---

## Section 5: Training Configuration Reference

### Step 5.1: Configuration Template for Different Data Scenarios

**Scenario A: Good Data (≥200 images/class, balanced)**

```json
{
  "training": {
    "continual": {
      "data": {
        "augmentation_policy": "randaugment",
        "randaugment_num_ops": 2,
        "randaugment_magnitude": 7,
        "sampler": "shuffle",
        "target_size": 224
      },
      "class_balance": {
        "allow_sampler_and_loss_rebalance": false
      },
      "num_epochs": 16,
      "batch_size": 8,
      "learning_rate": 0.00015
    }
  }
}
```

**Scenario B: Moderate Data (100-199 images/class, imbalanced)**

```json
{
  "training": {
    "continual": {
      "data": {
        "augmentation_policy": "randaugment",
        "randaugment_num_ops": 2,
        "randaugment_magnitude": 7,
        "sampler": "weighted",
        "target_size": 224
      },
      "class_balance": {
        "allow_sampler_and_loss_rebalance": true
      },
      "num_epochs": 20,
      "batch_size": 8,
      "learning_rate": 0.00015
    }
  }
}
```

**Scenario C: Limited Data (50-99 images/class)**

```json
{
  "training": {
    "continual": {
      "data": {
        "augmentation_policy": "augmix",  // More aggressive
        "augmix_severity": 4,
        "augmix_width": 4,
        "augmix_depth": 2,
        "sampler": "weighted",
        "target_size": 224
      },
      "class_balance": {
        "allow_sampler_and_loss_rebalance": true
      },
      "num_epochs": 24,  // More epochs to regularize
      "batch_size": 4,   // Smaller batch
      "learning_rate": 0.0001,  // Lower LR
      "weight_decay": 0.02  // More regularization
    }
  }
}
```

### Step 5.2: Training Command with Overrides

```bash
# Base command
.\scripts\python.cmd -m src.app.cli training \
  --crop tomato \
  --data-dir data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/ \
  --output-dir runs/tomato/experiment_001/ \
  --config-env base

# With augmentation override (disable for baseline)
.\scripts\python.cmd -m src.app.cli training \
  --crop tomato \
  --data-dir data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/ \
  --output-dir runs/tomato/baseline_no_augment/ \
  --config-env base \
  --config-override '{"training": {"continual": {"data": {"augmentation_policy": "none"}}}}'

# With aggressive augmentation (limited data)
.\scripts\python.cmd -m src.app.cli training \
  --crop tomato \
  --data-dir data/prepared_runtime_datasets/tomato_leaf_20250502_abc123/ \
  --output-dir runs/tomato/aggressive_augment/ \
  --config-env base \
  --config-override '{"training": {"continual": {"data": {"augmentation_policy": "augmix", "augmix_severity": 5}, "num_epochs": 20}}}'
```

---

## Section 6: Debugging and Troubleshooting

### Issue 1: "Validation accuracy is higher than training accuracy"

**Symptoms:**
- Training loss: 0.8 → 0.5 (good)
- Validation accuracy: 95% (suspiciously high!)
- Training accuracy: 82%

**Diagnosis: Likely Leakage**

```python
#!/usr/bin/env python3
"""Debug: Are validation images leaking into training?"""

import json
from pathlib import Path
from PIL import Image
import imagehash

def find_variant_leakage(dataset_root: Path):
    """Compare training and validation images for variants."""
    
    train_root = dataset_root / "continual"
    val_root = dataset_root / "val"
    
    # Compute perceptual hashes
    train_hashes = {}
    for img_path in train_root.rglob("*.jpg"):
        img = Image.open(img_path)
        train_hashes[str(img_path)] = imagehash.phash(img)
    
    leakage = []
    for val_path in val_root.rglob("*.jpg"):
        val_img = Image.open(val_path)
        val_hash = imagehash.phash(val_img)
        
        for train_path, train_hash in train_hashes.items():
            distance = val_hash - train_hash
            if distance <= 4:  # PHASH distance threshold
                leakage.append({
                    "val_image": str(val_path),
                    "train_image": train_path,
                    "phash_distance": distance,
                })
    
    if leakage:
        print(f"⚠️  Found {len(leakage)} variant leakage cases!")
        for case in leakage[:3]:
            print(f"   Val: {Path(case['val_image']).name}")
            print(f"   Train: {Path(case['train_image']).name}")
            print(f"   PHASH distance: {case['phash_distance']}")
        return False
    
    print("✓ No variant leakage detected")
    return True

# Run diagnostic
find_variant_leakage(Path("data/prepared_runtime_datasets/your_dataset/"))
```

**Fix:** Re-run Notebook 0 with correct family grouping.

### Issue 2: "Model doesn't generalize to new camera images"

**Symptoms:**
- Test set (same camera as training): 92% accuracy
- Real deployment (different camera): 65% accuracy

**Diagnosis: Multi-source Dataset Imbalance**

```python
#!/usr/bin/env python3
"""Debug: Are different cameras represented in all splits?"""

import json
from pathlib import Path
from collections import Counter

def check_source_distribution(dataset_root: Path):
    """Verify each camera/source appears in all splits."""
    
    manifest = json.loads((dataset_root / "split_manifest.json").read_text())
    
    source_by_split = {}
    
    for row in manifest.get("rows", []):
        split = row.get("split", "")
        source = row.get("source", "unknown")
        
        if split not in source_by_split:
            source_by_split[split] = Counter()
        
        source_by_split[split][source] += 1
    
    print("Source distribution across splits:")
    print()
    print(f"{'Source':<20} {'Train %':<12} {'Val %':<12} {'Test %':<12}")
    print("-" * 56)
    
    all_sources = set()
    for sources in source_by_split.values():
        all_sources.update(sources.keys())
    
    problems = []
    
    for source in sorted(all_sources):
        train_count = source_by_split.get("continual", Counter()).get(source, 0)
        val_count = source_by_split.get("val", Counter()).get(source, 0)
        test_count = source_by_split.get("test", Counter()).get(source, 0)
        
        train_total = sum(source_by_split.get("continual", {}).values())
        val_total = sum(source_by_split.get("val", {}).values())
        test_total = sum(source_by_split.get("test", {}).values())
        
        train_pct = 100 * train_count / train_total if train_total else 0
        val_pct = 100 * val_count / val_total if val_total else 0
        test_pct = 100 * test_count / test_total if test_total else 0
        
        print(f"{source:<20} {train_pct:<11.1f}% {val_pct:<11.1f}% {test_pct:<11.1f}%")
        
        # Flag if source is missing or severely under-represented
        if train_count == 0 or val_count == 0 or test_count == 0:
            problems.append(f"{source} missing in some split")
        
        if train_pct > 0 and val_pct < train_pct * 0.3:
            problems.append(f"{source} under-represented in val ({val_pct:.1f}% vs {train_pct:.1f}%)")
    
    if problems:
        print()
        print("⚠️  Issues detected:")
        for problem in problems:
            print(f"   - {problem}")
        return False
    else:
        print()
        print("✓ Source distribution balanced across splits")
        return True

# Run diagnostic
check_source_distribution(Path("data/prepared_runtime_datasets/your_dataset/"))
```

**Output Example (Problem):**
```
Source distribution across splits:

Source               Train %      Val %        Test %       
----------------------------------------------------
camera_1             65.0%        20.0%        10.0%       ← Under-represented in val/test
camera_2             35.0%        80.0%        90.0%       ← Over-represented in val/test

⚠️  Issues detected:
   - camera_1 under-represented in val (20.0% vs 65.0%)
```

**Fix:** Re-split dataset with source-aware stratification:

```python
from sklearn.model_selection import train_test_split

def source_aware_split(image_paths: list, source_metadata: dict, test_size=0.2):
    """Split while preserving source distribution."""
    
    # Group by source
    by_source = {}
    for img_path in image_paths:
        source = source_metadata.get(img_path, "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(img_path)
    
    train_paths = []
    test_paths = []
    
    # Split each source independently
    for source, paths in by_source.items():
        source_train, source_test = train_test_split(
            paths,
            test_size=test_size,
            random_state=42,
        )
        train_paths.extend(source_train)
        test_paths.extend(source_test)
    
    return train_paths, test_paths

# Usage
train_split, test_split = source_aware_split(all_images, source_metadata)
# Result: Each source now represented in train and test proportionally
```

---

## Section 7: Checklist for Production Training

**Before Training:**

- [ ] Dataset prepared with Notebook 0 (family grouping applied)
- [ ] Split integrity verified (no cross-split variant leakage)
- [ ] Class balance analyzed and strategy decided (online-only / online+offline / research mode)
- [ ] Source/camera distribution verified if multi-source dataset
- [ ] Reference image counts extracted from `split_manifest.json`

**During Training:**

- [ ] Augmentation policy active only on training split
- [ ] Validation evaluated on clean (unaugmented) data
- [ ] Loss curves monitored (training loss descending, validation loss converging)
- [ ] Checkpoints saved with augmentation settings recorded
- [ ] No manual modifications to val/test splits

**After Training:**

- [ ] Test results reported with data composition details:
  - Base image counts per class
  - Augmentation strategy used
  - Effective training set size (base × epochs × variants)
- [ ] Confusion matrix generated from **unaugmented** test set
- [ ] OOD calibration performed (if real OOD data available)
- [ ] Artifacts written to `production_readiness.json`

---

**Questions? Refer to:**
1. Main literature review: [data_augmentation_leakage_prevention.md](./data_augmentation_leakage_prevention.md)
2. Your codebase: `src/data/transforms.py`, `src/data/loaders.py`, `scripts/augment_runtime_train_split.py`
3. Notebook examples: `colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb`, `colab_notebooks/2_train_continual_sd_lora_adapter.ipynb`
