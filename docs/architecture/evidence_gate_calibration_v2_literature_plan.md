# Evidence Gate Calibration V2 Literature Plan

Date: 2026-06-13

Purpose: define how to evolve the current advisory v1 evidence-gate calibration into an automated v2 system without manually tuning every adapter.

Current v1 surfaces:

- `src/pipeline/evidence_gate_calibration.py`
- `scripts/calibrate_evidence_gate.py`
- `docs/architecture/evidence_gate_calibration_plan.md`
- `docs/architecture/review_gate_failure_analysis_and_literature.md`
- `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`
- `docs/ablation_results/dual_view_inference/multi_target_failure_prioritization.md`

## V2 Conclusion

Do not move from one global threshold to hand-tuned per-adapter thresholds. Move to a hierarchical, automated calibration system:

```text
Notebook 16 rows
-> global risk-coverage baseline
-> group policies by part/crop/failure profile
-> target-specific policies only when evidence is sufficient
-> clustered fallback when per-target evidence is sparse
-> explicit no_eligible_policy when constraints fail
-> advisory artifact first
-> runtime integration only after holdout validation
```

The literature supports this direction:

- selective classification gives the risk-coverage framing;
- Learn-Then-Test and Conformal Risk Control give explicit risk-control calibration frameworks;
- Mondrian/group-conditional conformal methods justify target/group-conditioned calibration;
- clustered conformal prediction addresses limited per-class/per-target calibration data;
- temperature scaling and related probability calibration methods address overconfident neural outputs;
- APS/RAPS prediction sets become useful later if top-k class probabilities are exported.

## Literature Map

### 1. Selective Classification: The Correct Problem Framing

Source: Geifman and El-Yaniv, 2017, "Selective Classification for Deep Neural Networks"  
URL: https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

Key idea:

- A classifier can abstain/reject some predictions to satisfy a desired risk level.
- The operational metric is risk-coverage, not just accuracy.

Repo implication:

- `requires_review=True` is a reject/abstain decision.
- V2 should output risk-coverage curves and selected policies under explicit constraints.
- Evaluation should state both accepted coverage and accepted-risk, plus missed wrong decisions and false-positive reviews.

Recommended V2 adoption:

- Keep the policy as a post-hoc selector over current model outputs.
- Make `coverage`, `wrong_capture_rate`, `false_positive_review_rate`, `accepted_error_rate`, and `review_rate` first-class metrics.

### 2. SelectiveNet: Useful Later, Not V2 Immediate

Source: Geifman and El-Yaniv, 2019, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option"  
URL: https://proceedings.mlr.press/v97/geifman19a.html

Key idea:

- Instead of thresholding a pretrained classifier, train an integrated reject head end-to-end.

Repo implication:

- This is not the next step because current adapters are already trained and Notebook 16 is an advisory analysis surface.
- It becomes relevant only when retraining adapters or adding a multi-view/evidence head.

Recommended V2 adoption:

- Do not implement SelectiveNet in v2.
- Record it as a v3/v4 retraining option if multi-view adapter retraining becomes planned.

### 3. Learn-Then-Test: Policy Grid With Statistical Rejection

Source: Angelopoulos et al., 2021, "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control"  
URL: https://arxiv.org/abs/2110.01052

Key idea:

- Calibrate predictive algorithms to satisfy explicit finite-sample risk constraints without refitting the model.
- Treat candidate policies as hypotheses and reject unsafe ones.

Repo implication:

- This matches the current v1 grid-search shape.
- V2 should keep enumerating candidate policies, but the candidate space and reporting need to expand.
- The system must continue to emit `no_eligible_policy` when no candidate meets constraints.

Recommended V2 adoption:

- Keep deterministic candidate policies.
- Add p-value or confidence-bound style reporting later, but first preserve a simple held-out constraint check.
- Make constraints explicit:
  - `min_wrong_capture`
  - `max_false_positive_review`
  - `max_review_rate`
  - optional `max_accepted_error_rate`

### 4. Conformal Risk Control: Monotone Risk-Control Policies

Source: Angelopoulos et al., 2024, "Conformal Risk Control"  
URL: https://arxiv.org/abs/2208.02814

Key idea:

- Generalizes split conformal prediction to control the expected value of monotone losses.
- Examples include controlling false negative rate in computer vision/NLP settings.

Repo implication:

- Full-confidence thresholding is monotone: raising the threshold reviews more rows.
- Missed wrong prediction rate and false-positive review burden can be treated as risk criteria.
- Arbitrary combinations of ROI features may not be monotone, so V2 should separate monotone threshold sweeps from non-monotone feature-grid experiments.

Recommended V2 adoption:

- Add monotone risk-coverage curves per target/group for confidence threshold.
- Treat ROI/OOD feature policies as auditable grid candidates, but do not claim conformal risk guarantees unless the monotonicity assumptions are satisfied.
- Store selected threshold and metrics with calibration and holdout partitions.

### 5. Neural Network Calibration: Confidence Is Not Probability

Source: Guo et al., 2017, "On Calibration of Modern Neural Networks"  
URL: https://arxiv.org/abs/1706.04599

Key idea:

- Modern neural networks can be overconfident.
- Temperature scaling is a simple post-hoc calibration method that keeps predictions unchanged while softening confidence.
- ECE/reliability diagrams quantify confidence-vs-accuracy gaps.

Repo implication:

- Current missed errors have median full confidence `0.9665`, so raw confidence is not enough.
- If adapter logits or class probabilities are available, V2 should compute calibration diagnostics and try temperature scaling.
- If only scalar confidence is available, V2 can still fit scalar risk curves, but it cannot properly recalibrate the full class distribution.

Recommended V2 adoption:

- Add calibration diagnostics:
  - ECE or binned reliability by target/group;
  - high-confidence error rate;
  - confidence histograms for correct vs incorrect rows.
- Add an optional `logits/probabilities available` branch:
  - temperature scaling per adapter/target where enough validation rows exist;
  - global or group temperature fallback otherwise.

### 6. Dirichlet / Multiclass Calibration: Later If Probabilities Exist

Source: Kull et al., 2019, "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration"  
URL: https://arxiv.org/abs/1910.12656

Key idea:

- Multiclass classifiers can need multiclass-specific calibration beyond a single scalar temperature.

Repo implication:

- Useful if Notebook 16 exports full probability vectors.
- Too much for the immediate v2 if only top-1 label and confidence are available.

Recommended V2 adoption:

- Do not implement first.
- Add schema support for `topk_predictions` or `class_probabilities`.
- Revisit when adapter inference payloads expose top-k probabilities.

### 7. APS / Adaptive Prediction Sets

Source: Romano, Sesia, and Candes, 2020, "Classification with Valid and Adaptive Coverage"  
URL: https://arxiv.org/abs/2006.02544

Key idea:

- Build prediction sets for classification with guaranteed marginal coverage and adaptive behavior.

Repo implication:

- Prediction sets are a better representation than top-1 confidence for disease-family confusions.
- If `tomato__leaf` has several plausible classes, a prediction set of size greater than one is an interpretable review signal.

Recommended V2 adoption:

- Add as optional v2.5/v3 feature if top-k class probabilities are available.
- Use set size and true-label containment as metrics.

### 8. RAPS For Image Classifiers

Source: Angelopoulos et al., 2021, "Uncertainty Sets for Image Classifiers using Conformal Prediction"  
URL: https://arxiv.org/abs/2009.14193

Key idea:

- RAPS regularizes APS for image classifiers, often yielding smaller and more stable prediction sets.

Repo implication:

- Relevant because this is an image-classification pipeline.
- Requires class probability vectors, not just scalar confidence.

Recommended V2 adoption:

- Not required for v2 core.
- Add a future `prediction_set_policy` extension once top-k/full probabilities are exported.

### 9. Mondrian / Group-Conditional Conformal Prediction

Source: MAPIE Mondrian conformal documentation  
URL: https://mapie.readthedocs.io/en/v0.9.0/theoretical_description_mondrian.html

Key idea:

- Split calibration by group to obtain group-conditional coverage.

Repo implication:

- The natural groups are:
  - target: `tomato__leaf`, `grape__leaf`, etc.;
  - part: `leaf`, `fruit`;
  - crop: `tomato`, `grape`, `strawberry`, `apricot`;
  - failure-profile clusters.
- Current evidence already proves target heterogeneity.

Recommended V2 adoption:

- Add group fallback hierarchy:

```text
target-specific
-> crop+part cluster if target evidence too small
-> part group
-> crop group
-> global fallback
-> no_eligible_policy if fallback is unsafe
```

Do not silently use global fallback when it fails risk constraints.

### 10. Clustered Conformal For Limited Target Data

Source: "Class-Conditional Conformal Prediction with Many Classes"  
URL: https://www.stat.berkeley.edu/~ryantibs/papers/classconf.pdf

Key idea:

- Standard conformal is stable but too coarse.
- Classwise conformal is granular but high variance when each class has limited data.
- Clustered conformal groups similar classes to balance granularity and variance.

Repo implication:

- This is directly relevant because some targets have too few wrong examples:
  - `strawberry__leaf`: `1`
  - `tomato__fruit`: `12`
  - `apricot__fruit`: `14`
  - `grape__fruit`: `16`
- V2 should not force per-target policy where evidence is weak.

Recommended V2 adoption:

- Add clustered fallback:
  - first simple manually defined clusters: `leaf_targets`, `fruit_targets`;
  - then data-driven clusters by risk-coverage curve similarity, confusion profile, and ROI evidence distribution.
- Require minimum calibration and holdout error counts before target-specific policy can win.

### 11. Equalized Coverage For Adaptively Selected Groups

Source: Zhou and Sesia, 2024, "Conformal Classification with Equalized Coverage for Adaptively Selected Groups"  
URL: https://arxiv.org/abs/2405.15106

Key idea:

- Select groups that reflect model limitations or biases, then provide coverage conditional on those groups.

Repo implication:

- V2 can start with fixed groups; later it can identify groups from failure analysis:
  - high-confidence wrong;
  - ROI supports wrong full-image;
  - leaf disease-family confusion;
  - fruit maturity/healthy confusion.

Recommended V2 adoption:

- Do not implement adaptive selected groups immediately.
- Make the report expose candidate group definitions so future runs can adopt them safely.

## V2 Design Requirements

### Inputs

Required:

- Notebook 16 `multi_target_report.json`
- row fields:
  - `target_id`
  - `expected_label`
  - `diagnosis`
  - `full_confidence`
  - `roi_evidence_status`
  - `roi_quality_status`
  - `full_ood_is_ood`
  - `roi_ood_is_ood`
  - `requires_review`

Optional but strongly recommended:

- top-k predictions;
- class probability vector;
- logits;
- split indicator;
- adapter metadata and readiness/calibration metadata.

### Candidate Policy Families

V2 should evaluate policy families separately:

1. **Confidence threshold only**

```text
review = full_confidence < threshold
```

Use a wider threshold grid:

```text
[0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
```

2. **Confidence + ROI/OOD feature gates**

```text
review =
    full_confidence < threshold
    OR gated_roi_conflict
    OR gated_roi_quality_bad
    OR gated_full_ood
    OR gated_roi_ood
```

Report as grid-search evidence, not conformal monotone guarantee unless monotonicity is preserved.

3. **Calibrated-risk score**

If enough rows exist, fit:

```text
P(error | features)
```

Potential features:

- full confidence;
- target id/group;
- ROI conflict;
- ROI quality status;
- full OOD flag;
- ROI OOD flag;
- confidence delta;
- detection source.

Keep this report-only until validation is strong. Start with logistic calibration or isotonic calibration; avoid complex learned models in v2.

4. **Prediction-set policy**

Only when top-k or full class probabilities exist:

```text
review = prediction_set_size > 1
```

or:

```text
review = top2_margin < margin_threshold
```

### Group Hierarchy

Use this fallback order:

```text
target_id
-> target_cluster
-> part
-> crop
-> global
-> no_eligible_policy
```

Initial groups:

- `leaf_targets`
- `fruit_targets`
- `tomato_targets`
- `grape_targets`
- `strawberry_targets`
- `apricot_targets`

Later groups can be generated from:

- confusion profile similarity;
- risk-coverage curve similarity;
- ROI evidence profile;
- OOD/confidence profile.

### Selection Constraints

Keep the current core constraints:

- `min_capture`, default `0.70`
- `max_false_positive_rate`, default `0.15`
- `min_target_errors`, default `20`

Add v2 constraints:

- `max_review_rate`, default maybe `0.25`
- `min_calibration_errors`, default `10`
- `min_holdout_errors`, default `5`
- `max_holdout_capture_drop`, default `0.15`
- `max_holdout_fp_increase`, default `0.10`

Rationale:

- A policy that passes calibration but collapses on holdout should not be promoted.
- A policy that captures errors by reviewing half the dataset is not useful.

### Selection Objective

Select eligible policies in this order:

1. pass calibration constraints;
2. pass holdout stability constraints;
3. maximize accepted coverage;
4. maximize wrong capture;
5. minimize false-positive review;
6. minimize review rate;
7. prefer simpler policies;
8. prefer broader group policy if target-specific and group policy are statistically similar.

The last rule avoids unnecessary per-target fragmentation.

## V2 Output Schema

Use a new schema version:

```json
{
  "schema_version": "v2_evidence_gate_calibration",
  "source_report": "docs/ablation_results/dual_view_inference/multi_target_report.json",
  "constraints": {},
  "policy_families": {},
  "global_policy": {},
  "group_policies": {},
  "target_policies": {},
  "risk_coverage_curves": {},
  "calibration_diagnostics": {},
  "audit_queue": [],
  "runtime_recommendation": {
    "status": "advisory_only",
    "reason": "not_runtime_validated"
  }
}
```

### Audit Queue

Targets should be queued for data/class-boundary audit when:

- no calibration policy meets constraints;
- false-positive review remains too high at useful capture levels;
- one confusion pair dominates errors;
- high-confidence errors dominate;
- target sample or error count is too small for reliable policy selection.

Example:

```json
{
  "target_id": "strawberry__fruit",
  "reason": "confidence_thresholding_cannot_improve_capture_without_extreme_false_positive_review",
  "top_confusion": "strawberry_healthy_fruit -> strawberry_unripe_fruit"
}
```

## Concrete Implementation Plan

### Step 1: Automate Failure Prioritization

Add:

```text
scripts/analyze_notebook16_failures.py
```

It should regenerate:

```text
docs/ablation_results/dual_view_inference/multi_target_failure_prioritization.md
```

This prevents manual interpretation drift.

### Step 2: Expand V1 Calibration Into V2 Report Mode

Modify:

```text
src/pipeline/evidence_gate_calibration.py
scripts/calibrate_evidence_gate.py
```

Add:

- `--schema-version v1|v2`
- wider threshold grid;
- risk-coverage curve output;
- group policy evaluation;
- holdout stability checks;
- audit queue output.

Keep runtime unchanged.

### Step 3: Add Tests

Add tests for:

- threshold grid includes `0.95`, `0.98`, `0.99`;
- group fallback wins when target evidence is sparse;
- target-specific wins only when calibration and holdout pass;
- no eligible policy stays explicit;
- audit queue marks unsalvageable targets;
- risk-coverage curves are deterministic.

### Step 4: Optional Probability Export

Before implementing temperature scaling or APS/RAPS, verify whether adapter inference can expose:

- logits;
- top-k labels;
- top-k probabilities;
- full class probability vector.

If not available, V2 should stay threshold/risk-curve based.

## What V2 Should Not Do

- Do not hand-tune every adapter.
- Do not promote policies into runtime automatically.
- Do not use a global ROI conflict or ROI quality rule as a shortcut.
- Do not pretend target-specific calibration is reliable when target errors are sparse.
- Do not use calibration to hide data/class-boundary problems.
- Do not implement SelectiveNet or retraining in V2.

## Recommended V2 Scope

V2 should be:

```text
automated, hierarchical, report-only calibration
```

Not:

```text
runtime behavior change
```

Minimum useful V2:

1. wider thresholds;
2. risk-coverage curves;
3. target/group/global fallback hierarchy;
4. holdout stability checks;
5. audit queue;
6. deterministic markdown + JSON outputs.

## Final Recommendation

The current v1 calibration system is the right base. V2 should not replace it with a learned model or manual per-adapter rules. It should extend v1 into a hierarchical selective-prediction calibrator:

```text
selective classification framing
+ Learn-Then-Test style candidate rejection
+ conformal risk-control style monotone threshold curves
+ Mondrian/group-conditional fallback
+ clustered fallback for low-data targets
+ optional confidence calibration when logits/probabilities exist
```

The first engineering step is still to automate the multi-target failure report. The second step is to add v2 report-mode calibration with group fallback and wider threshold curves. Runtime integration should remain a separate later decision.
