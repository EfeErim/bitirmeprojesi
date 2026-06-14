# Notebook 16 Multi-Target Failure Prioritization

Date: 2026-06-13

Source report: `docs/ablation_results/dual_view_inference/multi_target_report.json`
Calibration artifact: `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`

## Executive Decision

The next work should not be a single-target `strawberry__fruit` fix. `strawberry__fruit` is the largest accuracy outlier, but the system-level blocker is broader:

- review gate misses too many wrong full-image decisions;
- global aggressive evidence gates are unsafe because they over-review correct predictions;
- target-specific behavior differs enough that one policy is not credible;
- adapter/data investigation is needed, but only after the failure priorities are separated by target and failure type.

Keep production behavior unchanged: full-image adapter prediction remains final, ROI/bbox evidence remains advisory, and calibration artifacts stay report-only until a separate runtime integration is validated.

## Global Snapshot

| Metric | Value |
| --- | ---: |
| Samples | `2946` |
| Accuracy | `0.8836` |
| Macro-F1 | `0.8563` |
| Requires-review rate | `0.0859` |
| ROI conflict rate | `0.1388` |
| Review capture on wrong predictions | `0.3557` |
| False-positive review rate on correct predictions | `0.0503` |

Failure-analysis bucket counts:

| Bucket | Count |
| --- | ---: |
| Router | `0` |
| BBox / ROI evidence | `754` |
| Adapter | `681` |
| Confidence / OOD | `413` |
| Review gate | `352` |

Interpretation:

- Router handoff is not the current primary blocker in this report.
- The review gate catches only about one third of wrong full-image decisions.
- BBox/ROI signals are active, but they do not justify hard ROI override because prior ROI scoring underperformed full-image scoring.
- Adapter/data quality is uneven by target and should be prioritized by error volume plus review-gate miss rate, not by one headline outlier.

## Target Priority Table

| Priority | Target | Samples | Errors | Accuracy | Review capture | Missed wrong reviews | False-positive review | Calibration status | Why it matters |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `tomato__leaf` | `1204` | `121` | `0.8995` | `0.3058` | `84` | `0.0416` | `no_eligible_policy` | Largest sample volume and tied for most missed wrong reviews; improving this affects the most production-like traffic. |
| 2 | `strawberry__fruit` | `204` | `112` | `0.4510` | `0.2500` | `84` | `0.3152` | `no_eligible_policy` | Worst accuracy and very high false-positive review rate; this is a data/label/adapter outlier, not the whole system. |
| 3 | `grape__leaf` | `324` | `39` | `0.8796` | `0.5897` | `16` | `0.0632` | `target_specific` | Only target with an eligible calibration policy, but holdout metrics weakened; useful as a calibration pilot, not runtime-ready. |
| 4 | `apricot__leaf` | `150` | `28` | `0.8133` | `0.5357` | `13` | `0.1066` | `no_eligible_policy` | Moderate accuracy problem with non-trivial review burden; good secondary calibration/data target. |
| 5 | `grape__fruit` | `174` | `16` | `0.9080` | `0.6250` | `6` | `0.0823` | `too_few_errors` | Error count is low; monitor, but do not prioritize ahead of higher-volume failures. |
| 6 | `apricot__fruit` | `280` | `14` | `0.9500` | `0.1429` | `12` | `0.0038` | `too_few_errors` | Accuracy is strong; review capture is weak, but error volume is too low for immediate target-specific tuning. |
| 7 | `tomato__fruit` | `119` | `12` | `0.8992` | `0.5833` | `5` | `0.1121` | `too_few_errors` | Small sample and low error count; keep as monitor target. |
| 8 | `strawberry__leaf` | `491` | `1` | `0.9980` | `0.0000` | `1` | `0.0000` | `too_few_errors` | Not a current failure priority despite high sample count. |

## Confusion Hotspots

### High-Volume Adapter/Data Issues

`tomato__leaf` has the largest operational footprint:

- `domates_late_blight_yaprak -> domates_early_blight_yaprak`: `18`
- `domates_early_blight_yaprak -> domates_late_blight_yaprak`: `15`
- `domates_bacterial_spot_and_speck_yaprak -> domates_septoria_leaf_spot_yaprak`: `11`
- several additional cross-disease confusions spread across early blight, late blight, septoria, powdery mildew, and bacterial spot/speck

This is a broad disease-family confusion problem, not one bad class pair.

`strawberry__fruit` is the largest accuracy outlier:

- `strawberry_healthy_fruit -> strawberry_unripe_fruit`: `79`
- `strawberry_gray_mold_fruit -> strawberry_powdery_mildew_fruit`: `13`
- `strawberry_gray_mold_fruit -> strawberry_anthracnose_fruit`: `9`
- `strawberry_powdery_mildew_fruit -> strawberry_anthracnose_fruit`: `6`
- `strawberry_powdery_mildew_fruit -> strawberry_gray_mold_fruit`: `5`

The healthy-vs-unripe confusion dominates, so the first question is whether the labels and class definitions are visually separable enough for the current adapter.

### Moderate Confusion Targets

`grape__leaf` has distributed disease confusions:

- anthracnose, downy mildew, powdery mildew, esca, fanleaf virus, and healthy classes cross-confuse.
- Review capture is relatively better than most targets, and this is the only target where v1 calibration found an eligible target-specific policy.

`apricot__leaf` mostly confuses healthy, shot-hole, and sharka-virus leaf classes:

- `kays_saglkl_yaprak_302 -> kays_yaprak_delen_cil_hastalg_yaprak_300`: `9`
- `kays_saglkl_yaprak_302 -> kays_sarka_virusu_yaprak_206`: `8`
- `kays_yaprak_delen_cil_hastalg_yaprak_300 -> kays_sarka_virusu_yaprak_206`: `7`

This suggests class-boundary and visual-similarity review before retraining.

## Review Gate Diagnosis

The review gate is too conservative for wrong predictions, but a global aggressive gate is not safe.

Missed wrong reviews by target:

| Target | Errors | Missed wrong reviews | Review capture |
| --- | ---: | ---: | ---: |
| `tomato__leaf` | `121` | `84` | `0.3058` |
| `strawberry__fruit` | `112` | `84` | `0.2500` |
| `grape__leaf` | `39` | `16` | `0.5897` |
| `apricot__leaf` | `28` | `13` | `0.5357` |
| `apricot__fruit` | `14` | `12` | `0.1429` |
| `grape__fruit` | `16` | `6` | `0.6250` |
| `tomato__fruit` | `12` | `5` | `0.5833` |
| `strawberry__leaf` | `1` | `1` | `0.0000` |

False-positive review pressure by target:

| Target | False-positive review rate |
| --- | ---: |
| `strawberry__fruit` | `0.3152` |
| `tomato__fruit` | `0.1121` |
| `apricot__leaf` | `0.1066` |
| `grape__fruit` | `0.0823` |
| `grape__leaf` | `0.0632` |
| `tomato__leaf` | `0.0416` |
| `apricot__fruit` | `0.0038` |
| `strawberry__leaf` | `0.0000` |

Interpretation:

- `tomato__leaf` is a high-leverage review-gate target because it misses many errors while keeping false-positive review relatively low.
- `strawberry__fruit` is not a clean review-gate target because it has both many missed errors and many false-positive reviews.
- `grape__leaf` is the best calibration pilot, but its holdout result is not strong enough to promote into runtime.

## ROI / BBox Evidence Diagnosis

ROI evidence is informative but inconsistent:

- The report has no ROI missing or fallback problem.
- Error rows often have `supports_full`, meaning ROI evidence can agree with a wrong full-image decision.
- ROI quality issues such as `roi_too_large` and `roi_too_small` are common enough to report, but a global ROI-quality review rule created too many false positives in calibration.

Notable error-row ROI evidence:

| Target | Main error-row ROI evidence pattern |
| --- | --- |
| `tomato__leaf` | `supports_full=45`, `roi_too_large=37`, `conflicts_with_full=25`, `roi_too_small=14` |
| `strawberry__fruit` | `supports_full=52`, `roi_too_large=48`, `conflicts_with_full=11`, `roi_too_small=1` |
| `grape__leaf` | `conflicts_with_full=16`, `roi_too_large=13`, `supports_full=7`, `roi_too_small=3` |
| `apricot__leaf` | `supports_full=13`, `roi_too_large=8`, `conflicts_with_full=5`, `roi_too_small=2` |

Decision:

- Do not use ROI conflict or ROI quality as a global runtime gate.
- Continue target-specific calibration and report-only analysis.
- Treat ROI evidence as a diagnostic signal until retraining aligns training and inference views.

## Recommended Work Order

1. **Build the multi-target failure analysis surface into a repeatable report command.**
   - Current report was produced from `multi_target_report.json`.
   - The next code step should add a small script that regenerates this prioritization table whenever Notebook 16 produces a new report.

2. **Target review-gate improvement where error volume and false-positive risk are both favorable.**
   - Start with `tomato__leaf`, not `strawberry__fruit`.
   - It has `121` errors, `84` missed wrong reviews, and only `0.0416` false-positive review rate.

3. **Use `grape__leaf` as the calibration pilot, not as a runtime change.**
   - It is the only target with an eligible v1 target-specific policy.
   - Holdout capture fell below the default target and false-positive review rose above the default cap, so it remains advisory.

4. **Run a separate data/label audit for `strawberry__fruit`.**
   - This is a real problem, but it is not the whole system.
   - The healthy-vs-unripe confusion should be checked before retraining or gate tuning.

5. **Keep router work in monitor mode.**
   - `router=0` in the latest failure buckets does not prove the router is permanently solved.
   - It does mean router rewrite is not the next evidence-backed move from this report.

## Non-Actions

Do not do these next:

- do not rewrite the router based on this Notebook 16 report;
- do not promote the global evidence-gate policy into runtime;
- do not make hard ROI crop inference the final decision path;
- do not start broad adapter retraining before separating label/data, review-gate, OOD/confidence, and ROI-evidence failure modes;
- do not treat `strawberry__fruit` as the single system blocker.

## Next Concrete Task

Add a repeatable report generator for this analysis:

```text
scripts/analyze_notebook16_failures.py
```

Default input:

```text
docs/ablation_results/dual_view_inference/multi_target_report.json
```

Default output:

```text
docs/ablation_results/dual_view_inference/multi_target_failure_prioritization.md
```

This keeps future Notebook 16 runs from turning into ad hoc manual interpretation.
