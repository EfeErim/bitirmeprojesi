# OOD Phase 1 Data Improvement Plan

This note captures the Phase 1 recommendation for improving the OOD evidence pool for each adapter in AADS v6. The goal is to make readiness gating and OOD evaluation more realistic before changing model architecture or score selection.

## Core Principle

Phase 1 is about improving the data surface first.

A good OOD pool should be:

- representative of deployment failures
- rich in same-crop unknowns
- rich in realistic failure cases such as blur, occlusion, damaged tissue, clutter, and unusual viewpoints
- not dominated by easy off-crop negatives
- kept disjoint from any auxiliary OE training pool

If the pool is too easy, the model may look better than it is.
If the pool is too small, the metrics are unstable.
If the pool is dominated by off-crop images, it can hide the real failure mode.

## Current Evidence Snapshot

From `data/ood_dataset/final/final_ood_manifest.json`:

- `strawberry__leaf_ood_final`: 292 images
- `strawberry__fruit_ood_final`: 292 images
- `tomato__leaf_ood_final`: 62 images
- `tomato__fruit_ood_final`: 1297 images
- `grape__leaf_ood_final`: 249 images
- `grape__fruit_ood_final`: 249 images

The main issue is not just size. It is class balance and slice quality.

## 1. Strawberry / Leaf

Supported classes in the prepared runtime dataset:

- healthy
- strawberry_leaf_scorch_leaf
- strawberry_leaf_spot_leaf
- strawberry_powdery_mildew_leaf

Current OOD mix:

- off_crop_secondary: 159
- other_crop_disease: 106
- non_plant_misc: 27

### Recommendation

The pool is usable, but it is overweighted toward easy negatives.

- Reduce `off_crop_secondary` substantially. Keep only a small validation slice of easy off-crop negatives.
- Remove or minimize `non_plant_misc` unless it is being used to test a specific deployment failure mode.
- Increase same-crop unknowns and realistic strawberry failure cases.

### Target mix

A better Phase 1 balance would look like:

- same-crop unsupported or unusual strawberry disease states: 150 to 170
- failure cases such as blur, occlusion, damaged leaves, harsh lighting, unusual view angle: 80 to 100
- off-crop validation negatives: 20 to 30
- non-plant misc: 0 to a very small token set if needed

### Why this matters

This adapter already has very strong classification metrics, so the remaining gap is mostly about OOD rejection. A pool dominated by easy off-crop images does not stress the detector enough.

## 2. Strawberry / Fruit

The fruit adapter uses the same 292-image strawberry OOD final pool, but the visual failure modes should be fruit-specific.

### Recommendation

- Rebuild the pool around fruit-specific unknowns instead of reusing a leaf-oriented mix.
- Add fruit damage, ripeness anomalies, rot patterns, sun damage, bruising, and partial occlusion.
- Keep off-crop negatives as a small minority.

### Target mix

- fruit-specific unknowns and unsupported disease states: 150 to 170
- fruit failure cases: 80 to 100
- off-crop validation negatives: 20 to 30
- non-plant misc: only if needed for a specific gate

### Why this matters

Fruit rejection is usually about texture, color progression, decay, and partial visibility. If the OOD pool is leaf-shaped, the evaluation can miss the actual fruit failure mode.

## 3. Tomato / Leaf

Current OOD mix:

- unsupported_tomato_unknowns: 13
- tomato_failure_cases: 6
- tomato_off_coverage_secondary: 17
- off_crop_secondary: 13
- scene_context_leak_check: 13

This is the weakest OOD pool in the repo.

### Recommendation

This one needs the biggest Phase 1 expansion.

- Grow the pool from 62 to at least 600 images.
- Add many more same-crop unknowns.
- Add realistic tomato leaf failure cases.
- Keep off-crop images only as a small validation slice.

### What to add

#### Same-crop unsupported tomato unknowns

Examples:

- tomato diseases not in the supported label set
- nutrient deficiency patterns
- herbicide damage
- insect damage
- wilt progression
- mildew-like symptoms not covered by training labels
- unusual stress patterns that resemble disease but are outside the supported set

#### Failure cases

Examples:

- blur and motion blur
- occlusion by hands, stems, soil, or other leaves
- torn or damaged leaves
- backlighting, shadow, and extreme exposure
- unusual camera angles
- small or distant leaves

#### Off-coverage same-crop negatives

Examples:

- seedlings
- senescent leaves
- visually stressed but not labeled disease cases
- plants outside the usual field framing

### Target mix

- same-crop unsupported tomato unknowns: 250 to 300
- failure cases: 180 to 200
- off-coverage same-crop: 100 to 120
- off-crop validation: around 10 to 20

### Why this matters

This adapter has enough classification support, but its real OOD pool is too small to make the false-positive rate meaningful. This is the highest-priority adapter for Phase 1.

## 4. Tomato / Fruit

Current OOD mix:

- unsupported_same_crop: 1273
- blur_or_occlusion: 24

### Recommendation

This pool is large, but it is too dominated by one broad category.

- Audit the `unsupported_same_crop` slice for duplicates and near-duplicates.
- Rebalance toward failure cases and diverse hard negatives.
- Increase blur, occlusion, damaged fruit, ripeness anomalies, and lighting variations.

### Target mix

- real same-crop unsupported diseases and similar hard negatives: about 800
- failure cases: 300 to 350
- off-crop validation: about 30

### Why this matters

A huge single slice can make the evaluation look broad while still being narrow in practice. The detector should face varied tomato fruit failure modes, not just one dominant unknown bucket.

## 5. Grape / Leaf

Current OOD mix:

- off_crop_or_root_disease: 148
- off_crop_secondary: 55
- scene_context_leak_check: 46

### Recommendation

This pool is closer to strawberry leaf than tomato leaf: it has useful coverage, but it is still too easy in parts.

- Reduce off-crop-heavy slices.
- Increase grape-specific same-crop unknowns.
- Add realistic grape failure cases.

### Target mix

- grape-specific unknowns: 120 to 140
- failure cases: 60 to 80
- off-crop negatives: about 25

### Good additions

- unsupported grape disease progressions
- root-disease-adjacent visual cases only if they realistically appear in leaf imagery
- unusual discoloration, vein artifacts, and leaf deformation
- blur and occlusion
- unusual angle and canopy clutter

## 6. Grape / Fruit

The fruit adapter should follow the same principle as grape leaf, but with fruit-specific failure modes.

### Recommendation

- Reduce off-crop-heavy validation slices.
- Add more fruit damage and ripeness anomalies.
- Keep only a small number of scene-context checks.

### Target mix

- grape fruit unknowns: 120 to 140
- fruit failure cases: 60 to 80
- off-crop negatives: about 25

## Prioritization Order

If only one adapter can be improved first, do this order:

1. `tomato__leaf`
2. `strawberry__leaf`
3. `tomato__fruit`
4. `strawberry__fruit`
5. `grape__leaf`
6. `grape__fruit`

## Practical Rule For Phase 1

For each adapter, aim for this rough ratio:

- 50 to 60 percent same-crop unknowns
- 25 to 35 percent realistic failure cases
- 5 to 15 percent off-crop validation negatives
- near-zero non-plant misc unless it is explicitly needed

That ratio is not a law, but it is a better starting point than a pool dominated by easy off-crop examples.

## After Phase 1

Once the OOD pools are improved, then Phase 2 should compare score methods on the same evidence surface:

- current ensemble
- energy-based score
- kNN-style score

Only after the data surface is strong should training-side regularization or OE-style exposure be considered.
