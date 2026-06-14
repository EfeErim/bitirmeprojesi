# Review Gate Failure Analysis and Literature Plan

Date: 2026-06-13

Primary evidence:

- `docs/ablation_results/dual_view_inference/multi_target_report.json`
- `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`
- `scripts/colab_roi_ablation.py`
- `src/pipeline/evidence_gate_calibration.py`

## Question

Why does the current Notebook 16 review gate miss wrong full-image decisions, and what solution family is supported by the current data plus the literature?

## Short Answer

The current gate misses wrong decisions because it is mostly a low-confidence gate. It reviews a row when full-image confidence is below the configured threshold, and only then attaches ROI quality/conflict details. But many wrong predictions are high-confidence mistakes. ROI evidence and OOD flags are not reliable enough as global standalone review triggers: they also fire on many correct predictions, and ROI often supports the wrong full-image class.

The solution should not be one global threshold. The next credible direction is target-conditional risk-control:

- calibrate per target when enough errors exist;
- use global or clustered fallback when target evidence is small;
- optimize a risk-coverage objective, not accuracy alone;
- keep ROI evidence as features for calibration/reporting, not hard override;
- separately audit high-confusion data/class definitions where calibration cannot rescue the target.

## Current Gate Logic

Notebook 16 uses `full_image_primary_with_roi_evidence`.

In `scripts/colab_roi_ablation.py`, the review logic is:

```text
review if target detection missing
review if Grounding DINO errored
review if full_confidence < full_confidence_review_threshold
    then attach ROI quality, semantic mismatch, ROI conflict, and ROI confidence lead reasons
```

Important consequence:

- `roi_conflict`, `roi_too_large`, `roi_too_small`, and `roi_confidence_leads` do not independently trigger review.
- A high-confidence wrong full-image prediction will pass unless target detection is missing or Grounding DINO errors.
- The default `full_confidence_review_threshold` is `0.70`.

This conservative design was intentional because hard ROI rules previously created too much noise. The latest report confirms that the conservative gate is safe but misses many errors.

## Data Findings

Global row counts:

| Group | Count |
| --- | ---: |
| Total comparable rows | `2946` |
| Wrong full-image predictions | `343` |
| Wrong predictions reviewed | `122` |
| Wrong predictions missed | `221` |
| Correct predictions reviewed | `131` |

The missed wrong predictions are not low-confidence:

| Group | Full-confidence median | Full-confidence p10 | Full-confidence p90 | `<0.70` | `<0.80` | `<0.90` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Missed wrong | `0.9665` | `0.7715` | `0.9964` | `0` | `28` | `68` |
| Reviewed wrong | `0.5504` | `0.3817` | `0.6689` | `122` | `122` | `122` |
| Correct reviewed | `0.5638` | `0.4538` | `0.6678` | `131` | `131` | `131` |
| Correct not reviewed | `0.9998` | `0.9330` | `1.0000` | `0` | `71` | `183` |

This explains the failure directly: the current review gate is confidence-threshold dominated, but `221` missed errors have confidence at least `0.7017`, and the median missed error confidence is `0.9665`.

## Why Simple Global Thresholding Fails

Raising the global confidence threshold improves capture but quickly reviews too many correct predictions:

| Global threshold | Review rate | Wrong capture | False-positive review |
| ---: | ---: | ---: | ---: |
| `0.70` | `0.0859` | `0.3557` | `0.0503` |
| `0.80` | `0.1195` | `0.4373` | `0.0776` |
| `0.90` | `0.1711` | `0.5539` | `0.1206` |
| `0.95` | `0.2251` | `0.6443` | `0.1698` |
| `0.98` | `0.2946` | `0.7201` | `0.2386` |
| `0.99` | `0.3479` | `0.8192` | `0.2858` |

If the target is `>=0.70` wrong capture and `<=0.15` false-positive review, no global threshold is acceptable. This matches the v1 calibration artifact, where the best rejected global policy captured many wrong predictions but had false-positive review rate `0.4416`.

## Why Target-Conditional Thresholding Looks Better

The same threshold has very different behavior by target.

For `tomato__leaf`:

| Threshold | Review rate | Wrong capture | False-positive review |
| ---: | ---: | ---: | ---: |
| `0.70` | `0.0681` | `0.3058` | `0.0416` |
| `0.90` | `0.1321` | `0.5868` | `0.0813` |
| `0.95` | `0.1761` | `0.7107` | `0.1163` |
| `0.98` | `0.2326` | `0.8099` | `0.1681` |

For `strawberry__fruit`:

| Threshold | Review rate | Wrong capture | False-positive review |
| ---: | ---: | ---: | ---: |
| `0.70` | `0.2794` | `0.2500` | `0.3152` |
| `0.90` | `0.4951` | `0.3125` | `0.7174` |
| `0.95` | `0.5686` | `0.3661` | `0.8152` |
| `0.98` | `0.6471` | `0.4375` | `0.9022` |

Interpretation:

- `tomato__leaf` is a plausible target-specific review-gate improvement candidate: `0.95` meets the default `0.70` capture and `0.15` false-positive constraints on this full report slice.
- `strawberry__fruit` cannot be solved by confidence thresholding. Raising the threshold mostly reviews correct predictions too. This target needs data/label/class-boundary analysis.
- A global rule hides these differences and creates the wrong operational tradeoff.

## ROI and OOD Signal Findings

Missed wrong predictions had these ROI evidence statuses:

| ROI evidence status | Missed wrong count |
| --- | ---: |
| `supports_full` | `91` |
| `roi_too_large` | `89` |
| `conflicts_with_full` | `29` |
| `roi_too_small` | `12` |

This matters:

- `supports_full` means ROI agrees with the wrong full-image decision. ROI cannot rescue those rows.
- `roi_too_large` is frequent, but previous global ROI-quality policies over-reviewed correct predictions.
- `conflicts_with_full` catches some errors, but it also appears in correct rows; it is not safe as a global standalone trigger.

OOD flags are also too weak as a global rescue signal:

- among missed wrong predictions, `full_ood_is_ood=True` only for `7/221`;
- `roi_ood_is_ood=True` only for `9/221`;
- many ROI OOD fields are unavailable when ROI prediction was not run because ROI quality failed.

## Root Causes

1. **The gate assumes low confidence is the main error signature.**
   - Current data says many errors are high-confidence.

2. **The model is overconfident on some class confusions.**
   - This is classic calibration failure: confidence is not consistently representative of correctness.

3. **ROI evidence is not monotonic with correctness.**
   - It can support wrong predictions, conflict with correct predictions, or fail quality checks.

4. **Target behavior is heterogeneous.**
   - `tomato__leaf` benefits from a more aggressive target threshold.
   - `strawberry__fruit` does not; it needs data/class-boundary work.

5. **The current grid is too small and globally constrained.**
   - It tests a useful v1 surface, but it cannot express target-specific shrinkage, confusion-pair risk, or calibrated probability of error.

## Literature Findings

### Selective Classification / Reject Option

Geifman and El-Yaniv frame the problem as selective classification: trade coverage for risk by rejecting uncertain or risky predictions. Their 2017 work constructs a selective classifier for a trained neural network and lets the user set a desired risk level, with rejection used to satisfy that risk target. Source: [Selective Classification for Deep Neural Networks](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).

Relevance to this repo:

- Notebook 16 review is exactly a reject-option surface.
- The repo should measure risk-coverage curves per target, not only accuracy.
- Review should be selected to meet an explicit risk target under a review budget.

### Integrated Reject Heads

SelectiveNet extends selective prediction by training a network with an integrated reject option instead of only thresholding a pretrained classifier confidence. Source: [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).

Relevance:

- This is useful later if adapters are retrained.
- It is not the immediate fix because current adapters are already trained and Notebook 16 is a post-hoc evidence surface.
- A future multi-view retrain could include a learned reject/evidence head, but v1 should stay post-hoc.

### Neural Network Calibration

Guo et al. show that modern neural networks can be poorly calibrated and that temperature scaling is a simple post-hoc calibration method. Source: [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

Relevance:

- The missed wrong rows have high full-image confidence, so raw confidence is not a calibrated risk score.
- If logits are available from adapters, temperature scaling per target or per adapter should be tested.
- If only final confidence is available, fit a post-hoc confidence-to-error calibrator on held-out rows and report calibration error.

### Conformal Risk Control

Conformal Risk Control generalizes conformal prediction to control expected monotone losses, including task-specific risks such as false negative rate. Source: [Conformal Risk Control](https://arxiv.org/pdf/2208.02814).

Relevance:

- This repo has explicit losses: missed wrong prediction and false-positive review.
- Review policies can be selected using calibration rows to control missed-error risk subject to review burden.
- The method assumes a monotone policy parameter; for this repo, confidence threshold is monotone, but arbitrary combinations of ROI features may not be.

### Learn-Then-Test

Learn-Then-Test calibrates predictive algorithms to satisfy explicit finite-sample guarantees without assuming model internals. Source: [Learn then Test](https://www.gsb.stanford.edu/faculty-research/working-papers/learn-then-test-calibrating-predictive-algorithms-achieve-risk).

Relevance:

- This fits the repo's deterministic policy grid.
- Candidate review policies can be tested against constraints such as capture `>=0.70` and false-positive review `<=0.15`.
- It supports rejecting unsafe policies instead of silently falling back.

### Group-Conditional / Mondrian Calibration

Mondrian conformal prediction calibrates separately by groups to obtain group-conditional guarantees. A practical description is that data are partitioned by group and split conformal is applied per group. Source: [MAPIE Mondrian conformal documentation](https://mapie.readthedocs.io/en/v0.9.0/theoretical_description_mondrian.html).

Relevance:

- Adapter target is the natural group: `tomato__leaf`, `grape__leaf`, etc.
- The latest data proves group heterogeneity, so global calibration is not enough.
- Small-error targets need fallback or shrinkage because per-target calibration is unstable when errors are few.

### Adaptive Prediction Sets / RAPS

Adaptive Prediction Sets and RAPS produce set-valued predictions for classifiers and can adapt set size to uncertainty while offering finite-sample coverage. Sources: [Classification with Valid and Adaptive Coverage](https://arxiv.org/abs/2006.02544) and [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://openreview.net/forum?id=eNdiU_DbM9).

Relevance:

- If adapter top-k probabilities are available, a prediction-set surface can review rows where the set has multiple plausible disease classes.
- This is better aligned with disease-family confusions than binary top-1 confidence alone.
- It is especially relevant for `tomato__leaf` and `grape__leaf`, where several disease classes cross-confuse.

## Proposed Solution

### Phase 1: Expand The Calibration Report, Not Runtime

Add a repeatable analysis command that produces:

- per-target risk-coverage curves;
- missed-error and false-positive review counts by threshold;
- confusion-pair summaries;
- ROI/OOD signal distributions over missed errors;
- recommended target-specific candidate policies, clearly marked advisory.

Candidate file:

```text
scripts/analyze_notebook16_failures.py
```

This is the next immediate implementation step because the current report was manually generated.

### Phase 2: Add Target-Conditional Policy Search

Extend `src/pipeline/evidence_gate_calibration.py` beyond the current small grid:

- allow target-specific confidence thresholds beyond `[0.50, 0.60, 0.70, 0.80, 0.90]`, at least `0.95` and `0.98`;
- report risk-coverage curves, not just one winner;
- use target policies only when both calibration and holdout are stable;
- add shrinkage or global fallback for small-error targets;
- keep `no_eligible_policy` explicit.

Do not apply any policy to runtime yet.

### Phase 3: Calibrate Confidence Properly

If adapter logits or class probability vectors are available:

- fit temperature scaling per adapter/target on validation rows;
- compute expected calibration error and reliability curves;
- re-run review thresholding on calibrated confidence.

If only scalar confidence is available:

- fit a simple held-out isotonic or logistic error calibrator per target where sample size allows;
- fallback to grouped or global calibration where target rows are sparse.

### Phase 4: Add Prediction-Set Evidence When Top-K Scores Exist

For targets with disease-family confusions:

- compute APS/RAPS-style prediction sets from top-k class probabilities;
- review if set size is greater than one, or if a known high-risk confusion pair is active;
- evaluate missed-error capture and false-positive review per target.

This is most relevant for:

- `tomato__leaf`;
- `grape__leaf`;
- `apricot__leaf`.

### Phase 5: Separate Data/Class-Boundary Audits

Do not use calibration to hide data problems.

Run targeted data audits for:

- `strawberry__fruit`: healthy vs unripe class boundary;
- `tomato__leaf`: early blight vs late blight and bacterial spot/speck vs septoria;
- `apricot__leaf`: healthy vs shot-hole vs sharka.

The output should decide whether the issue is:

- label noise;
- visually ambiguous class boundary;
- insufficient examples;
- missing context;
- adapter underfit;
- unsuitable class taxonomy.

## Recommended Next Implementation

Implement `scripts/analyze_notebook16_failures.py` first.

Acceptance criteria:

- reads `docs/ablation_results/dual_view_inference/multi_target_report.json`;
- optionally reads `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`;
- writes `docs/ablation_results/dual_view_inference/multi_target_failure_prioritization.md`;
- includes global and per-target threshold sweeps;
- includes missed wrong review profiles;
- includes target confusion hotspots;
- includes ROI/OOD signal distributions over missed errors;
- does not change Notebook 16 or runtime inference behavior.

Then extend calibration search with target-specific threshold candidates `0.95` and `0.98`, but still keep the artifact advisory until a separate validation pass says otherwise.

## Final Recommendation

The review gate fails because it treats confidence below `0.70` as the primary review trigger while the dominant missed errors are high-confidence mistakes. The fix is not a single stronger threshold. The fix is a target-conditional selective-prediction workflow:

```text
per-target failure report
-> per-target risk-coverage curves
-> calibrated confidence or calibrated risk score
-> target/group conditional policy selection
-> holdout validation
-> report-only artifact
-> runtime integration only after validation
```

For immediate engineering work, automate the failure report first. For model-quality work, treat `tomato__leaf` as the best review-gate candidate and `strawberry__fruit` as a separate data/class-boundary audit.
