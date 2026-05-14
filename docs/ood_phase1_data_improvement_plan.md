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

## Split Policy For OOD Evidence

The cleanest evaluation setup is to keep the unknown pool separate from both training and calibration decisions.

Recommended split roles:

- `ood_dev`: a small calibration subset used only for threshold selection, score comparison, and readiness tuning
- `ood_test`: the final held-out OOD evidence that is never used for tuning
- `ood_sanity_check`: a separate mixed set with a small amount of ID inserted into OOD so we can measure false positives and contamination sensitivity

This follows the literature and avoids evaluation leakage:

- Outlier Exposure explicitly separates auxiliary outliers from test outliers and does not tune on the test distribution. Source: [Hendrycks et al., 2018](https://openreview.net/forum?id=HyxCxhRcY7)
- OOD benchmark quality matters, and contaminated test OOD sets can distort conclusions. Source: [Bitterwolf et al., 2023](https://proceedings.mlr.press/v202/bitterwolf23a.html)
- The amount of auxiliary outlier data can be small, but training outliers and final OOD evidence still must be disjoint. Source: [Liznerski et al., 2022](https://openreview.net/forum?id=3v78awEzyB)

Practical rule for this repo:

- keep `val/` for in-distribution calibration and model selection
- keep `ood/` as the source of final unknown evidence
- if the pool is large enough, split `ood/` into `ood_dev` and `ood_test`
- if the pool is too small, use `val/` for calibration and reserve `ood/` for final readiness evidence only
- do not mix the exact same images between `oe/` and final `ood/` evidence

The `ood_sanity_check` set is optional but useful when you want to answer a specific question: "does the adapter still call real ID images OOD if a few are mixed into a harder unknown set?"

It is a diagnostic stress set, not the canonical final OOD benchmark.

## Current Evidence Snapshot

From `data/ood_dataset/final/final_ood_manifest.json` after the 2026-05-14 targeted OOD/OE takviye:

- `apricot__fruit_ood_final`: 175 images
- `apricot__leaf_ood_final`: 195 images
- `grape__fruit_ood_final`: 126 images
- `grape__leaf_ood_final`: 343 images
- `strawberry__fruit_ood_final`: 95 images
- `strawberry__leaf_ood_final`: 213 images
- `tomato__fruit_ood_final`: 352 images
- `tomato__leaf_ood_final`: 370 images

The main issue is not just size. It is class balance and slice quality. The 2026-05-14 pass added apricot coverage, kept tomato leaf above the 300-image stability floor, and refreshed tomato fruit around same-crop unsupported unknowns plus failure cases. The weakest current pools are strawberry fruit and grape fruit; strawberry fruit still needs more true fruit-specific failure cases before it should be treated as complete.

## 1. Apricot / Leaf

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- same_crop_unsupported_unknowns: 90
- other_crop_disease: 61
- off_crop_secondary: 35
- non_plant_misc: 9

The apricot leaf pool exists, but it is still small and currently leans toward other-crop and off-crop evidence rather than same-crop unknowns.

### Recommendation

- Increase apricot-specific unsupported disease states.
- Keep off-crop and non-plant slices as secondary evidence.
- Add more apricot leaf failure cases such as blur, occlusion, and canopy clutter.

### Target mix

- same-crop unsupported or unusual apricot disease states: 120 to 140
- failure cases: 60 to 80
- off-crop validation negatives: 20 to 30
- non-plant misc: a very small token set if needed

### Why this matters

This pool is useful, but not yet rich enough to stress rejection on apricot-specific unknowns.

## 2. Apricot / Fruit

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- same_crop_unsupported_unknowns: 51
- other_crop_disease: 50
- scene_context_leak_check: 20
- off_crop_secondary: 35
- non_plant_misc: 19

The apricot fruit pool is even more mixed than the leaf pool and still needs fruit-specific hard negatives.

### Recommendation

- Rebuild around fruit-specific unknowns instead of relying on other-crop disease.
- Add fruit damage, ripeness anomalies, bruising, rot, sun damage, and partial occlusion.
- Keep off-crop and non-plant images as minority slices.

### Target mix

- fruit-specific unknowns and unsupported disease states: 120 to 140
- fruit failure cases: 60 to 80
- off-crop validation negatives: 20 to 30
- non-plant misc: only if needed for a specific gate

### Why this matters

Apricot fruit rejection should be decided by fruit-like failure modes, not by off-crop shortcuts.

## 3. Strawberry / Leaf

Supported classes in the prepared runtime dataset:

- healthy
- strawberry_leaf_scorch_leaf
- strawberry_leaf_spot_leaf
- strawberry_powdery_mildew_leaf

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- same_crop_unsupported_unknowns: 165
- strawberry_failure_cases: 13
- off_crop_secondary: 25
- other_crop_disease: 5
- non_plant_misc: 5

### Recommendation

The pool is no longer dominated by easy negatives, but realistic failure cases are still under target.

- Keep `off_crop_secondary` and `non_plant_misc` small.
- Keep same-crop unknowns as the dominant slice.
- Add more realistic strawberry leaf failure cases.

### Target mix

A better Phase 1 balance would look like:

- same-crop unsupported or unusual strawberry disease states: 150 to 170
- failure cases such as blur, occlusion, damaged leaves, harsh lighting, unusual view angle: 80 to 100
- off-crop validation negatives: 20 to 30
- non-plant misc: 0 to a very small token set if needed

### Why this matters

This adapter already has very strong classification metrics, so the remaining gap is mostly about OOD rejection. A pool dominated by easy off-crop images does not stress the detector enough.

## 4. Strawberry / Fruit

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- same_crop_unsupported_unknowns: 42
- fruit_specific_unknowns: 8
- fruit_failure_cases: 11
- off_crop_secondary: 6
- other_crop_disease: 0
- non_plant_misc: 28

The fruit adapter is now much smaller than the leaf adapter, and the current mix is still not fruit-specific enough.

### Recommendation

- Continue rebuilding the pool around fruit-specific unknowns instead of leaf-oriented same-crop negatives.
- Add fruit damage, ripeness anomalies, rot patterns, sun damage, bruising, and partial occlusion.
- Keep off-crop negatives as a small minority.

### Target mix

- fruit-specific unknowns and unsupported disease states: 150 to 170
- fruit failure cases: 80 to 100
- off-crop validation negatives: 20 to 30
- non-plant misc: only if needed for a specific gate

### Why this matters

Fruit rejection is usually about texture, color progression, decay, and partial visibility. If the OOD pool is leaf-shaped, the evaluation can miss the actual fruit failure mode.

## 5. Tomato / Leaf

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- unsupported_tomato_unknowns: 260
- tomato_failure_cases: 11
- tomato_off_coverage_secondary: 21
- off_crop_secondary: 40
- scene_context_leak_check: 38

This pool is large enough for a stable FPR estimate, but the failure-case slice still needs more blur, occlusion, partial-leaf, and mixed-background examples.

### Recommendation

This one was the biggest Phase 1 expansion.

- Keep the pool above 300 images before using FPR as a readiness signal.
- Add more realistic tomato leaf failure cases.
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

This adapter has enough classification support, but its real OOD pool is still not the most weakly covered surface in the manifest. FPR is meaningful here, but the failure slice should still be expanded.

## 6. Tomato / Fruit

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- unsupported_same_crop: 208
- fruit_failure_cases: 80
- blur_or_occlusion: 24
- off_crop_secondary: 20
- non_plant_misc: 20

### Recommendation

This pool is fruit-specific and no longer dominated by leaf-oriented or off-crop evidence.

- Audit the `unsupported_same_crop` slice for duplicates and near-duplicates.
- Rebalance toward failure cases and diverse hard negatives.
- Increase blur, occlusion, damaged fruit, ripeness anomalies, and lighting variations.

### Target mix

- real same-crop unsupported diseases and similar hard negatives: about 800
- failure cases: 300 to 350
- off-crop validation: about 30

### Why this matters

A huge single slice can make the evaluation look broad while still being narrow in practice. The detector should face varied tomato fruit failure modes, not just one dominant unknown bucket.

## 7. Grape / Leaf

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- off_crop_or_root_disease: 148
- off_crop_secondary: 90
- same_crop_unsupported_unknowns: 35
- scene_context_leak_check: 70

### Recommendation

This pool has useful coverage, but it is still too easy in parts and still leans heavily on off-crop evidence.

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

## 8. Grape / Fruit

Current OOD mix after the 2026-05-14 targeted OOD/OE takviye:

- off_crop_or_root_disease: 14
- off_crop_secondary: 55
- same_crop_unsupported_unknowns: 30
- scene_context_leak_check: 27

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

If only one adapter can be improved first, do this order based on current pool weakness and OOD risk:

1. `strawberry__fruit`
2. `grape__fruit`
3. `apricot__fruit`
4. `apricot__leaf`
5. `strawberry__leaf`
6. `grape__leaf`
7. `tomato__fruit`
8. `tomato__leaf`

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

## References

- Hendrycks, D., Mazeika, M., & Dietterich, T. (2018). Deep Anomaly Detection with Outlier Exposure. arXiv. https://arxiv.org/abs/1812.04606
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv. https://arxiv.org/abs/2107.07511
- Naqvi, A., et al. (2022). Uncovering bias in the PlantVillage dataset: A comparison of diseased plant leaves in isolation and within canopies. arXiv. https://doi.org/10.48550/arXiv.2206.04374
