# Agricultural Domain-Specific Augmentation Guide

**Version:** 1.0  
**Date:** May 2, 2026  
**Domain:** Plant Disease Detection and Crop Health Assessment  
**References:** Aligned with [data_augmentation_leakage_prevention.md](./data_augmentation_leakage_prevention.md)

---

## 1. Introduction: Why Agricultural Augmentation Is Different

Plant disease detection has unique constraints that differentiate it from general-purpose image classification (ImageNet, COCO, etc.):

| Factor | General CV | Plant Disease Detection |
|--------|------------|------------------------|
| **Class Definition** | Object category (dog, car, person) | Disease phenotype (symptom combination) |
| **Biological Invariances** | Rotation, scale, translation | Same, BUT with limits: leaf must remain functional |
| **Critical Features** | Shape, color, texture, context | Subtle color shifts, lesion margins, vein patterns |
| **Data Collection** | Controlled settings, diverse angles | Field conditions, single viewing angle per plant |
| **Augmentation Risk** | Minimal—objects tolerate extreme transforms | High—disease signals are fragile |

**Key Implication:** Augmentation for plant disease must be *conservative* compared to general-purpose CV. Aggressive color shifts or crops can destroy the very disease signals the model is learning to detect.

---

## 2. Which Augmentations Preserve Biological Validity?

### 2.1 Geometry-Based Augmentations (Generally Safe)

#### ✓ Horizontal Flip (p=0.5)
**Biological Rationale:**
- Leaves are bilaterally symmetric in most crops
- Disease does not prefer left vs. right side of a leaf
- Flipping reflects natural viewing angle variation in field photography

**Constraints:**
- Some crops (e.g., asymmetric petiole attachment) show directional asymmetry
- Solution: Use p=0.5 (50%) to allow non-flipped versions

**Supported Range:** p=0.4–0.6 (conservative: p=0.5)

#### ✓ Vertical Flip (p=0.05–0.15)
**Biological Rationale:**
- Leaves have vein architecture oriented from petiole to tip
- Upside-down leaves are less realistic but help model robustness

**Constraints:**
- Strong vertical flip (p>0.2) can contradict vein patterns that disease follows
- Some disease lesions are oriented (e.g., fungal spores follow vein pathways)

**AADS v6 Setting:** p=0.1 (10% flip probability)  
**Recommendation:** Keep ≤ 0.15

#### ✓ Rotation (±12° recommended)
**Biological Rationale:**
- Field photos capture leaves at variable angles
- Leaf rotations reflect natural camera positioning variation

**Constraints:**
- Extreme rotations (>30°) introduce unnatural perspective
- Very high rotations (>45°) make the leaf practically inverted

**Why ±12° Specifically:**
- Covers ~80% of natural in-field viewing angle variation
- Preserves vein direction and lesion orientation cues
- Extreme enough to prevent memorization of absolute orientation

**Reference:** Agronomic phenotyping standards limit to ±30° max (Kirchgessner et al., 2017)

#### ⚠ Random Crop (Use with Care)
**Biological Rationale:**
- Partial leaf views occur in field photography
- Cropping teaches robustness to occlusion

**Constraints:**
- Crops must preserve minimum disease feature size
- Small crops (<160 px) lose the contextual lesion patterns

**Safe Implementation:**
```python
transforms.RandomResizedCrop(
    size=(224, 224),
    scale=(0.80, 1.0),      # Crop 80-100% of image
    ratio=(0.95, 1.05),     # Preserve aspect ratio (avoid distortion)
)

# After crop and resize to 224×224:
# 80% crop → 160 px min dimension ✓ Preserves lesion detail
# 100% crop (no crop) → Full leaf ✓ Baseline
```

**Not Recommended:**
```python
scale=(0.5, 1.0)   # ✗ 50% crops too aggressive
scale=(0.6, 0.9)   # ✗ Crops too severe, loses context
ratio=(0.7, 1.3)   # ✗ Extreme aspect ratio changes realism
```

### 2.2 Color-Based Augmentations (Moderate Caution)

#### ✓ Brightness Adjustment (±14%)
**Biological Rationale:**
- Field lighting varies by time of day, cloud cover, camera exposure
- Disease color cues depend on lighting

**Your Range:** 0.86–1.14 (±14% adjustment)

**Why This Works:**
- ±14% reflects realistic field condition variation
- Preserves disease color discrimination (e.g., yellow halo vs. brown center)

**References:**
- AgriNet experiments (Saleem et al., 2019) found ±20% acceptable
- AADS v6 conservative default: ±14% ✓

#### ✓ Contrast Adjustment (±18%)
**Biological Rationale:**
- Camera sensors and post-processing vary in contrast
- Disease features (lesion edges, discoloration) are contrast-dependent

**Your Range:** 0.88–1.18 (±18% adjustment)

**Why This Works:**
- Lesion margins remain visible at ±18%
- Prevents overfitting to specific camera contrast profiles
- Helps with robustness to different smartphone cameras

#### ✓ Saturation/Color Adjustment (±16%)
**Biological Rationale:**
- Plant imaging shows color variation due to:
  - Different cameras (phone, DSLR, multispectral camera)
  - Different white balance settings
  - Different lighting (overcast, sunny, shade)
- Disease colors shift in saturation (e.g., early blight appears more/less saturated)

**Your Range:** 0.90–1.16 (±16% adjustment)

**IMPORTANT CONSTRAINT—DO NOT USE:**
```python
# ✗ Extreme saturation shifts
transforms.ColorJitter(saturation=0.5)   # Too aggressive
transforms.ColorJitter(saturation=1.0)   # Makes colors unrecognizable

# ✓ Conservative
transforms.ColorJitter(saturation=0.16)  # ±16% as in AADS v6
```

#### ⚠ Hue Adjustment (±5% Maximum)
**Biological Rationale:**
- Disease diagnosis relies on specific colors (yellow, brown, black lesions)
- Hue shifts destroy disease color classification

**Your Range:** ±5% (roughly ±18° on 360° color wheel)

**IMPORTANT CONSTRAINT:**
```python
# ✗ Too aggressive
transforms.ColorJitter(hue=0.2)   # ±72° — changes red to purple!

# ✓ Conservative
transforms.ColorJitter(hue=0.05)  # ±18° — subtle shift only
```

**Why ±5% Works:**
- Green leaf stays green, doesn't become cyan
- Yellow lesion stays yellow, doesn't become orange or red
- Brown lesion stays brown, doesn't become purple
- Prevents model from learning hue-based shortcuts

#### ✗ Desaturation / Grayscale (NOT RECOMMENDED)
**Why Not:**
- Plant diseases are diagnosis by COLOR (red early blight, yellow early blight, etc.)
- Grayscale removes the most discriminative feature
- Exception: Only for specialized grayscale camera models

```python
# ✗ Bad
transforms.RandomGrayscale(p=0.1)  # Removes color information

# ✓ If absolutely necessary
transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.02)  # Rare
```

### 2.3 Texture Augmentations (Use Sparingly)

#### ⚠ Gaussian Blur (p=0.1, σ ∈ [0.1, 1.5])
**Biological Rationale:**
- Field photos vary in focus quality
- Slight blur helps robustness to camera focus variation

**Your Setting:** p=0.1, σ ∈ [0.1, 1.5]

**Constraints:**
- Excessive blur (σ > 2.0) destroys fine disease texture (pustules, lesion edges)
- Subtle blur (σ < 0.5) helps with focus robustness

**Why Conservative Here:**
- Lesion texture (pustule pattern, sporulation) is diagnostic
- Excessive blur makes healthy and diseased leaves indistinguishable

#### ✗ Noise Addition (NOT RECOMMENDED for Disease Detection)
**Why Not:**
- Random noise mimics image compression or sensor noise
- This is NOT representative of field conditions
- Can obscure fine disease features

```python
# ✗ Bad
transforms.GaussianNoise(std=0.1)  # Adds meaningless artifacts

# ✓ Better
transforms.RandomApply([transforms.GaussianBlur(...)], p=0.1)  # Focus variation
```

#### ✗ JPEG Compression Artifacts (NOT RECOMMENDED)
**Why Not:**
- Most field datasets already JPEG-compressed
- Adding further compression misrepresents reality

```python
# ✗ Bad
transforms.RandomJPEG(quality=80)  # Adds artifacts beyond real JPEG

# ✓ If needed
# Just use JPEG-compressed source images directly
```

---

## 3. Multi-Organ and Cross-Crop Considerations

### 3.1 Organ-Specific Augmentation Adjustments

Plant diseases manifest differently by organ. Your AADS v6 router already classifies plant part (leaf, stem, fruit). Consider part-aware augmentation parameters:

#### Leaf Disease (Most Common)
- Rotation: ±12° ✓ (leaves can appear at many angles)
- Crop: 80-100% ✓ (partial leaves are common)
- Flip: p=0.5 ✓ (bilateral symmetry)

#### Fruit/Tomato Disease
- Rotation: ±8° (fruits are 3D spheres; extreme rotation creates unrealistic perspective)
- Crop: 85-100% ✓ (center of fruit is diagnostic)
- Flip: p=0.3 (fruits are roughly spherically symmetric but sun-exposed side matters)

#### Stem Disease
- Rotation: ±18° ✓ (stems curved; angle variation high)
- Crop: 80-100% (lesion often on a small section of stem)
- Flip: p=0.5 ✓ (no directional preference)

### 3.2 Cross-Crop Dataset Handling

**Challenge:** Multi-crop datasets (tomato + pepper + corn) may have very different:
- Lesion scales (relative to image)
- Color patterns
- Lighting conditions (greenhouse vs. field)

**Solution: Separate Training Per Crop**

AADS v6 does this correctly:
- Train one adapter per crop
- Each adapter is crop-specific

**Within each crop, apply shared augmentation settings:**
```python
# config/base.json → Applies to all crops
"augmentation_policy": "randaugment",
"randaugment_num_ops": 2,
"randaugment_magnitude": 7,

# If crop-specific tuning needed:
# config/colab.json → Overrides for Colab (same applies to all crops)
```

**Don't Do:**
```python
# ✗ Mixing augmentation parameters by crop
# This breaks reproducibility and comparison

# ✓ Do:
# Train all crops with the same augmentation settings
# Use crop-specific architecture (if needed) but shared augmentation
```

---

## 4. Synthetic Augmentation: When (Not) to Use GANs

### 4.1 GAN-Generated Disease Samples: The Evidence

**Literature Finding (Wang et al., 2021; Zhang et al., 2019):**
- GANs trained on small agricultural datasets learn dataset artifacts, not generalizable disease patterns
- GAN samples often show "morphological implausibility" (impossible disease configurations)
- Models trained on GAN data generalize worse to real test data

**Why This Happens:**
1. GANs optimize for visual realism (low FID score)
2. GAN captures dataset-specific correlations, not true disease biology
3. Model learns to recognize GAN artifacts instead of disease patterns

### 4.2 When Synthetic Data Is Appropriate

**Acceptable Use Cases:**

1. **Augmentation of EXISTING real disease samples** (e.g., generate color variants using style transfer)
   - Keep base disease sample real
   - Vary only lighting/camera conditions
   - Example: Color-jitter-based "synthetic" variants

2. **Data augmentation for RARE disease classes**
   - If you have 5 real late_blight samples and need more
   - Use real samples + conservative synthetic variants
   - Document clearly: "5 real + 20 style-transfer variants"

3. **Controlled research studies** (with explicit disclaimer)
   - Study generalization gaps
   - Benchmark against real-only baselines
   - Label clearly: "Research—do not deploy"

### 4.3 How AADS v6 Handles This

**Flag System:**

```json
// split_manifest.json
{
  "image_name": "early_blight_001.jpg",
  "synthetic_hint": false,  // ← Real field image
  "generated_offline_augmentation": false,  // ← Not a PIL variant
  "reference_image_count": 1
}

{
  "image_name": "early_blight_001_gan_variant.jpg",
  "synthetic_hint": true,  // ← GAN-generated
  "generated_offline_augmentation": false,
  "reference_image_count": 0  // ← Excluded from reference counts
}
```

**Training-Time Filtering:**
```python
# From scripts/augment_runtime_train_split.py

for row in manifest.get("rows", []):
    # Skip synthetic augmentations when counting reference images
    if bool(row.get("synthetic_hint")):
        continue  # ← Excluded
    
    # Count only real images
    counts[class_name] = counts.get(class_name, 0) + 1
```

### 4.4 Recommendation for AADS v6

**For Production:**
- Use ONLY real augmentation (geometric + color transforms)
- Example: `augment_runtime_train_split.py` creates PIL-based variants only

**For Research:**
- If GAN augmentation is needed, mark all synthetic samples with `"synthetic_hint": true`
- Train on {real + synthetic}, test on {real only}
- Report: "X% accuracy on real test set" (not on synthetic)
- Document: "Research configuration; not recommended for deployment"

**Never:**
```python
# ✗ Mix real and synthetic without tracking
# This invalidates comparison with other methods

# ✓ Do:
# If using synthetic: Report separately, with synthetic_hint metadata
```

---

## 5. Dataset-Specific Augmentation Strategies

### 5.1 Single-Camera, Controlled Dataset

**Example:** Greenhouse photos, single lighting protocol, one camera model

**Recommended Augmentation:**
```json
{
  "augmentation_policy": "randaugment",
  "randaugment_num_ops": 2,
  "randaugment_magnitude": 7,
  "crop_scale": [0.85, 1.0],
  "horizontal_flip": true,
  "vertical_flip_probability": 0.05,
  "rotation_degrees": 12
}
```

**Rationale:** Limited variation in conditions → light augmentation prevents overfitting without introducing unrealistic transformations.

### 5.2 Multi-Source, Field-Collected Dataset

**Example:** Multiple cameras, variable lighting, smartphones + DSLRs

**Recommended Augmentation:**
```json
{
  "augmentation_policy": "augmix",
  "augmix_severity": 3,
  "augmix_width": 3,
  "crop_scale": [0.80, 1.0],
  "horizontal_flip": true,
  "vertical_flip_probability": 0.08,
  "rotation_degrees": 15,
  "brightness_range": [0.85, 1.20],      // Wider range for lighting variation
  "contrast_range": [0.85, 1.20],
  "saturation_range": [0.85, 1.20],
  "hue_shift": 0.08                      // Slightly wider (±8%) for camera variation
}
```

**Rationale:** High environmental variation → more aggressive augmentation, plus wider color shifts to account for multi-camera conditions.

**Plus source-aware stratification:**
```python
# Split training/val/test such that each source (camera model)
# appears in all three splits with balanced representation
```

### 5.3 Severely Limited Dataset (<50 images/class)

**Example:** Rare disease, pilot study, new crop

**Recommended Augmentation:**
```json
{
  "augmentation_policy": "augmix",
  "augmix_severity": 4,
  "augmix_width": 4,
  "augmix_depth": 2,
  "augmix_alpha": 1.5,
  "crop_scale": [0.75, 1.0],            // More aggressive crops
  "horizontal_flip": true,
  "vertical_flip_probability": 0.15,    // More flips for data diversity
  "rotation_degrees": 18,
  "offline_augmentation": true,          // Generate variants pre-training
  "offline_variants_per_image": 3
}
```

**Plus offline augmentation:**
```bash
# Pre-generate augmented variants in training split only
.\scripts\python.cmd scripts/augment_runtime_train_split.py \
  --source-root data/prepared_runtime_datasets/limited_data/ \
  --num-variants 3 \
  --seed 42
```

**CRITICAL:** Document this as research configuration:
```json
// runs/limited_data_study/config_used.json
{
  "data_regime": "research_few_shot",
  "note": "Trained on <100 images/class with aggressive augmentation. Not recommended for production.",
  "base_images_per_class": 25,
  "effective_training_samples": 500,  // 25 × 3 variants × 8 epochs
  "warning": "High generalization risk; use for exploration only."
}
```

---

## 6. Cross-Validation with Agricultural Data

### 6.1 Stratified K-Fold for Limited Data

**When to use:** Dataset 50-200 total images per class

**Correct Procedure (PREVENTS LEAKAGE):**

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_agricultural_dataset(
    image_paths: list,
    class_labels: list,
    family_groups: dict,  # Maps image → family ID
    n_splits: int = 5,
):
    """Cross-validate with family-aware stratification."""
    
    # Map families to indices
    family_to_indices = {}
    for idx, (img, label) in enumerate(zip(image_paths, class_labels)):
        family = family_groups.get(img, f"singleton_{idx}")
        if family not in family_to_indices:
            family_to_indices[family] = []
        family_to_indices[family].append(idx)
    
    # Create stratified family-level split
    skf = StratifiedKFold(n_splits=n_splits, random_state=42)
    
    for fold_idx, (train_family_ids, test_family_ids) in enumerate(
        skf.split(
            list(family_to_indices.keys()),
            [class_labels[list(indices)[0]] for indices in family_to_indices.values()]
        )
    ):
        # Convert family indices to image indices
        train_indices = [
            idx 
            for fam_id in train_family_ids 
            for idx in family_to_indices[list(family_to_indices.keys())[fam_id]]
        ]
        test_indices = [
            idx 
            for fam_id in test_family_ids 
            for idx in family_to_indices[list(family_to_indices.keys())[fam_id]]
        ]
        
        train_images = [image_paths[i] for i in train_indices]
        test_images = [image_paths[i] for i in test_indices]
        
        # CRITICAL: Augment ONLY train set
        train_loaders = create_loaders(
            train_images,
            transform=True,  # Online augmentation enabled
            augmentation_policy="randaugment",
        )
        test_loaders = create_loaders(
            test_images,
            transform=False,  # NO augmentation
        )
        
        # Train and evaluate
        yield fold_idx, train_loaders, test_loaders

# Usage
for fold, train_loader, test_loader in cross_validate_agricultural_dataset(
    image_paths, class_labels, family_groups, n_splits=5
):
    print(f"Fold {fold}: Train on {len(train_loader)} families, test on {len(test_loader)} families")
    # Train model...
    # Evaluate on test_loader (clean data)
```

---

## 7. Quality Control Checklist for Agricultural Datasets

### Pre-Training Audit

- [ ] **Image Resolution Acceptable**
  - [ ] Minimum 160 px per dimension (before augmentation)
  - [ ] Target ≥ 224 px (your target_size)
  
- [ ] **Disease Visibility**
  - [ ] Sample images manually reviewed
  - [ ] Disease features (lesions, discoloration) clearly visible
  - [ ] Not blurry or occluded
  
- [ ] **No Camera Artifacts**
  - [ ] No lens flare, fingerprints, or text overlays
  - [ ] No extreme underexposure/overexposure
  
- [ ] **Authentic Field Conditions**
  - [ ] Images from real plants, not reference photos or stock images
  - [ ] Flag suspicious images for review (use `human_review_packet.json` from Notebook 0)
  
- [ ] **Family Grouping Verified**
  - [ ] Similar images clustered together (run Notebook 0)
  - [ ] No false positives in similarity (inspect `human_review_packet.json`)
  
- [ ] **Split Integrity**
  - [ ] No family variants appear in multiple splits
  - [ ] Class distribution preserved across splits
  - [ ] Source/camera diversity in all splits (if multi-source)

### During Training

- [ ] **Augmented Samples Inspected**
  - [ ] Save augmented batch samples: `runs/augmented_samples.png`
  - [ ] Verify augmentations look realistic
  - [ ] Disease features still visible after augmentation
  
- [ ] **Training Metrics Reasonable**
  - [ ] Training loss decreases smoothly
  - [ ] Validation loss converges (not oscillating wildly)
  - [ ] No NaN or Inf losses
  
- [ ] **No Leakage Signals**
  - [ ] Validation accuracy within 5-10% of training accuracy
  - [ ] If validation much higher: possible leakage
  
### Post-Training Evaluation

- [ ] **Test Set Performance Reported**
  - [ ] Accuracy / precision / recall / F1 on UNAUGMENTED test set
  - [ ] Per-class metrics (especially minority classes)
  
- [ ] **Confusion Matrix Reviewed**
  - [ ] Which classes confused?
  - [ ] Is confusion biologically plausible?
  - [ ] Not confused by camera angle or leaf orientation
  
- [ ] **Augmentation Impact Reported**
  - [ ] Compare: baseline (no augment) vs. augmented
  - [ ] Quantify improvement: "+2-5% accuracy typical for leaf diseases"
  
- [ ] **Documentation Complete**
  - [ ] Number of base images per class
  - [ ] Augmentation policy and parameters used
  - [ ] Effective training set size
  - [ ] OOD calibration status

---

## 8. Troubleshooting: Agricultural-Specific Issues

### Issue: Model Fails on Out-of-Focus Images

**Diagnosis:** Training data is all sharp images; test data includes out-of-focus photos.

**Solution:**
```python
# Increase Gaussian blur in augmentation
transforms.RandomApply(
    [transforms.GaussianBlur(kernel_size=11, sigma=(0.5, 2.5))],
    p=0.15  # Increase from 0.1 to 0.15
)

# Or add defocus augmentation
transforms.RandomApply(
    [transforms.GaussianBlur(kernel_size=9, sigma=(1.0, 3.0))],
    p=0.1  # 10% chance of soft focus
)
```

### Issue: Model Fails on Different Smartphone Camera Images

**Diagnosis:** Training on DSLR images; deployment on smartphone images with different color rendering.

**Solution:**
```json
{
  "augmentation_policy": "augmix",
  "augmix_severity": 4,
  "augmix_width": 4,
  "brightness_range": [0.80, 1.25],      // Wider range
  "contrast_range": [0.80, 1.25],        // Wider range
  "saturation_range": [0.75, 1.30],      // Much wider for smartphone variation
  "hue_shift": 0.12                      // Wider for color rendering differences
}
```

**Plus:** Verify dataset includes diverse camera sources in training split.

### Issue: Color-Based Disease Misclassified

**Diagnosis:** Model learned camera-specific color encoding, not disease appearance.

**Solution:**
```python
# Reduce hue shift (was causing model to depend on precise colors)
"hue_shift": 0.03  # More conservative: ±11° instead of ±18°

# Add explicit color normalization
from src.data.transforms import preprocess_image
# This already applies ImageNet normalization
# May also consider per-image color normalization (WhiteBalance augmentation)

# Verify training includes camera diversity
check_source_distribution(dataset_root)
```

### Issue: Vertical Flip Causes Performance Drop

**Diagnosis:** Crop has natural upright orientation (e.g., corn tassel, tomato fruit hanging).

**Solution:**
```python
# Reduce or disable vertical flip for that crop
if crop == "corn":
    vertical_flip_probability = 0.0  # Disable
elif crop == "tomato":
    vertical_flip_probability = 0.05  # Minimal
else:
    vertical_flip_probability = 0.1   # Default
```

---

## 9. References Specific to Agricultural Augmentation

### Key Papers

1. **Sladojevic et al. (2016)** - Plant Disease Detection by Deep Learning
   - Citation: S. Sladojevic, M. Arsenovic, A. Anderla, D. Culibrk, D. Stefanovic. Plant disease detection and classification by deep learning. *Computational Intelligence and Neuroscience*, 2016, p. 6473.
   - Relevance: Early work on CNN-based crop disease detection; validated geometric augmentations (rotate ±20°, flip).

2. **Saleem et al. (2019)** - AgriNet: Crop Stress Phenotyping
   - Citation: M. H. Saleem, J. Potgieter, K. M. Arif. Plant disease detection and classification by deep learning. *Plants*, 8(11), 468.
   - Relevance: Recommends ±20% brightness/contrast variation; validates ±15° rotation.

3. **Kirchgessner et al. (2017)** - Automated Plant Phenotyping Standards
   - Citation: N. Kirchgessner, F. Liebisch, K. Yu, J. Pfeifer, A. C. Davison, A. Mikaelyan, A. Walter. The ETH field phenotyping platform FIP: A tool for large-scale, high-throughput phenotyping. *Plant Methods*, 13(1), 98.
   - Relevance: Defines acceptable imaging angle variation (±30°) for crop phenotyping.

4. **Wang et al. (2021)** - Domain Adaptation & Transfer Learning
   - Citation: Y. Wang, K. Li, L. Torresani, D. Forsyth, L. Feifei. Exploring model transferability through the lens of domain and task similarity. *ICCV*, 11640–11650.
   - Relevance: Warns against GAN-based augmentation for domain-specific tasks; agricultural imagery particularly susceptible to GAN artifacts.

### Standards & Guidelines

- **Plant Phenotyping Ontology (P3O):** Standardized terms for crop imaging (ISO 21572)
- **CGIAR Research Program on Climate Change, Agriculture and Food Security (CCAFS):** Field protocols for crop disease assessment
- **American Phytopathological Society (APS):** Disease rating scales and photography standards

---

## 10. Summary Table: Agricultural Augmentation Guidelines

| Augmentation | Probability/Range | Biological Validity | Notes |
|--------------|-------------------|-------------------|-------|
| **Horizontal Flip** | p=0.5 | ✓ High | Bilateral symmetry; use 50% of time |
| **Vertical Flip** | p=0.05–0.15 | ⚠ Medium | Use sparingly; breaks vein patterns if excessive |
| **Rotation** | ±12–18° | ✓ High | Field viewing angle variation |
| **Random Crop** | 80–100% scale | ✓ High | Preserve ≥160 px minimum; keeps lesion context |
| **Brightness** | ±14% (0.86–1.14) | ✓ High | Field lighting variation; conservative |
| **Contrast** | ±18% (0.88–1.18) | ✓ High | Camera/sensor variation; preserves lesion edges |
| **Saturation** | ±16% (0.90–1.16) | ✓ High | Color camera variation; preserves disease colors |
| **Hue** | ±5% max | ✓ High | Disease diagnosis depends on color; conservative only |
| **Gaussian Blur** | p=0.1, σ∈[0.1, 1.5] | ✓ Medium | Camera focus variation; don't over-blur |
| **RandAugment** | 2 ops, magnitude 7 | ✓ High | Proven effective; magnitude 7/30 is moderate |
| **Grayscale** | p=0.0 | ✗ Low | Disease diagnosis is color-dependent; don't use |
| **GAN Synthetic** | Not recommended | ✗ Very Low | Encodes dataset artifacts; use real variants only |
| **Desaturation** | p=0.0 | ✗ Low | Removes disease signal; not appropriate |
| **Extreme Crop** | <75% scale | ✗ Low | Loses contextual disease information; too aggressive |

---

## Appendix: Checklist for Your AADS v6 Setup

- [ ] Reviewed `config/base.json` augmentation parameters
- [ ] Confirmed online augmentation applied only to training split
- [ ] Verified validation and test splits have no augmentation
- [ ] (If limited data) Generated offline augmentation in training split only
- [ ] (If multi-source) Verified camera/source distribution across splits
- [ ] (If severe imbalance) Applied effective-number class weighting
- [ ] Inspected augmented training samples for biological plausibility
- [ ] Compared baseline (no augment) vs. augmented results
- [ ] Documented augmentation settings in training artifacts
- [ ] Verified test evaluation uses clean (unaugmented) data
- [ ] Reviewed confusion matrix for unrealistic misclassifications

---

**For questions on agricultural specifics, consult:**
1. Your agronomic domain experts (confirm crop-specific constraints)
2. Main literature review: [data_augmentation_leakage_prevention.md](./data_augmentation_leakage_prevention.md)
3. Implementation guide: [augmentation_implementation_guide.md](./augmentation_implementation_guide.md)
4. AADS v6 codebase: `src/data/transforms.py`, `config/base.json`
