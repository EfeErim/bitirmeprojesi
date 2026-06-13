# Dynamic Evidence Gate Calibration Plan

Date: 2026-06-13

## Summary

Manual per-adapter threshold tuning will be replaced with an automated, target-aware calibration surface. It will consume Notebook 16 `multi_target_report.json`, search a small auditable policy grid, and emit a versioned calibration JSON that chooses the highest-coverage review policy satisfying risk constraints.

The framing follows selective classification and risk-control literature: accept predictions when estimated risk is acceptable; otherwise send to review. Supporting literature is summarized in `docs/architecture/evidence_gate_calibration_literature_review.md`.

## Implementation Status

V1 is implemented in `src/pipeline/evidence_gate_calibration.py` and exposed through `scripts/calibrate_evidence_gate.py`.

Current default output:

```text
docs/ablation_results/dual_view_inference/evidence_gate_calibration.json
```

The first default run (`min_capture=0.70`, `max_false_positive_rate=0.15`) did not find an eligible global policy. It found a target-specific policy for `grape__leaf`; all other targets are explicitly marked `no_eligible_policy`, either because target evidence is too small or target-specific selection failed the risk constraints. Runtime inference behavior remains unchanged.

## Current Evidence

Latest Notebook 16 multi-adapter report:

- Source: `docs/ablation_results/dual_view_inference/multi_target_report.json`
- Sample count: `2946`
- Accuracy: `0.8836`
- Macro-F1: `0.8563`
- Failure buckets:
  - `router`: `0`
  - `bbox`: `754`
  - `adapter`: `681`
  - `confidence_ood`: `413`
  - `review_gate`: `352`
- Review capture on wrong predictions: `0.3557`
- False-positive review rate on correct predictions: `0.0503`

Interpretation:

- Router handoff is not the current main failure source.
- Adapter/data quality is a later investigation, especially `strawberry__fruit`.
- The immediate scalable work is automated, target-aware evidence gate calibration.

## Goal

Create an automated evidence-gate calibration workflow that:

- avoids manual per-adapter threshold selection;
- calibrates by target when enough target evidence exists;
- falls back to a global policy when target evidence is too small;
- emits recommendation artifacts only in the first version;
- does not change Notebook 16 inference behavior or production runtime behavior yet.

## Key Changes

### Reusable Calibration Logic

Add reusable calibration logic under `src/pipeline/`.

Responsibilities:

- Load Notebook 16 rows.
- Derive target key from `target_id`, `dataset_key`, or `crop__part`.
- Determine whether each row is correct.
- Evaluate candidate review policies.
- Split rows deterministically into calibration and holdout partitions.
- Select the highest-coverage eligible policy.
- Produce per-target and global calibration summaries.

### CLI Script

Add a script:

```text
scripts/calibrate_evidence_gate.py
```

Default input:

```text
docs/ablation_results/dual_view_inference/multi_target_report.json
```

Default output:

```text
docs/ablation_results/dual_view_inference/evidence_gate_calibration.json
```

The script should support:

- `--input`
- `--output`
- `--min-capture`
- `--max-false-positive-rate`
- `--min-target-errors`
- `--holdout-ratio`
- `--seed`
- `--include-samples` for debugging only

Default values:

- `--min-capture 0.70`
- `--max-false-positive-rate 0.15`
- `--min-target-errors 20`
- `--holdout-ratio 0.30`
- deterministic seed, for example `20260613`

## Candidate Policy Grid

Keep the first policy grid small and auditable:

- `full_confidence_threshold`: `[0.50, 0.60, 0.70, 0.80, 0.90]`
- `review_on_roi_conflict`: `[false, true]`
- `review_on_roi_quality_bad`: `[false, true]`
- `review_on_full_ood`: `[false, true]`
- `review_on_roi_ood`: `[false, true]`

Policy decision:

```text
review =
    full_confidence < full_confidence_threshold
    OR (review_on_roi_conflict AND roi_evidence_status == "conflicts_with_full")
    OR (review_on_roi_quality_bad AND roi_quality_status in {"roi_too_large", "roi_too_small"})
    OR (review_on_full_ood AND full_ood_is_ood)
    OR (review_on_roi_ood AND roi_ood_is_ood)
```

Do not add learned models in v1. This should remain a deterministic post-processing calibration surface.

## Selection Objective

For each candidate policy, compute:

- `review_rate`
- `coverage`
- `wrong_capture_rate`
- `false_positive_review_rate`
- `wrong_missed_count`
- `correct_reviewed_count`
- `sample_count`
- `wrong_count`
- `correct_count`

Eligibility:

- `wrong_capture_rate >= min_capture`
- `false_positive_review_rate <= max_false_positive_rate`

Selection:

- choose the eligible policy with maximum `coverage`;
- tie-break by higher `wrong_capture_rate`;
- then lower `false_positive_review_rate`;
- then simpler policy, meaning fewer enabled boolean review signals;
- then lower `full_confidence_threshold`.

If no policy is eligible:

- emit `status: "no_eligible_policy"`;
- include `best_rejected`;
- do not silently fall back to an unsafe policy.

## Target-Specific vs Global Policy

Use target-specific policy when:

- target has at least `min_target_errors` wrong examples across all rows;
- both calibration and holdout partitions contain at least one wrong example.

Use global fallback when:

- target sample is too small;
- target has too few wrong examples;
- target-specific selection has no eligible policy.

The output should clearly record whether each target uses:

- `target_specific`
- `global_fallback`
- `no_eligible_policy`

## Validation Protocol

Split rows deterministically:

- Use a stable hash of `image_path` plus target key.
- Assign approximately `70%` to calibration and `30%` to holdout.
- Select policy only on calibration rows.
- Report both calibration and holdout metrics.

Do not tune on final hidden test evidence if the project later adds a separate hidden test split. For the current repo state, Notebook 16 rows are an analysis/calibration artifact, not a production-readiness claim.

## Output Schema

Write:

```text
docs/ablation_results/dual_view_inference/evidence_gate_calibration.json
```

Required top-level shape:

```json
{
  "schema_version": "v1_evidence_gate_calibration",
  "source_report": "docs/ablation_results/dual_view_inference/multi_target_report.json",
  "constraints": {
    "min_capture": 0.7,
    "max_false_positive_rate": 0.15,
    "min_target_errors": 20,
    "holdout_ratio": 0.3,
    "seed": 20260613
  },
  "global_policy": {
    "status": "eligible",
    "policy": {},
    "calibration_metrics": {},
    "holdout_metrics": {}
  },
  "target_policies": {
    "tomato__leaf": {
      "status": "target_specific",
      "policy": {},
      "calibration_metrics": {},
      "holdout_metrics": {},
      "fallback_reason": ""
    }
  }
}
```

Optional debug fields:

- `variants`
- `eligible_variants`
- `rejected_variants`
- `samples`

Only include sample-level rows when `--include-samples` is explicitly set.

## Runtime Integration Boundary

Do not change production inference in v1.

This calibration artifact is advisory. It can later be integrated into:

- Notebook 16 report interpretation;
- canonical runtime evidence gate config;
- model/card handoff docs;
- OOD/readiness guard reporting.

Runtime use requires a separate implementation step after the calibration artifact is reviewed.

## Test Plan

Add unit tests for calibration logic:

- computes review metrics correctly;
- chooses highest-coverage eligible policy;
- rejects policies that miss capture or false-positive constraints;
- emits `best_rejected` when no policy is eligible;
- uses target-specific policy when target evidence is sufficient;
- uses global fallback when target evidence is insufficient;
- deterministic split is stable across repeated runs.

Add CLI tests:

- reads a minimal Notebook 16-style report;
- writes valid `evidence_gate_calibration.json`;
- supports custom constraints;
- omits sample rows by default;
- includes sample rows with `--include-samples`.

Recommended checks:

```powershell
./scripts/python.cmd -m pytest tests/unit/pipeline/test_evidence_gate_calibration.py tests/unit/scripts/test_calibrate_evidence_gate.py -q
./scripts/python.cmd -m ruff check src/pipeline scripts tests
./scripts/python.cmd scripts/calibrate_evidence_gate.py
```

## Acceptance Criteria

Implementation is complete when:

- `scripts/calibrate_evidence_gate.py` runs on the current Notebook 16 report.
- `evidence_gate_calibration.json` is written under `docs/ablation_results/dual_view_inference/`.
- Output includes global and per-target policies.
- Targets with weak evidence are explicitly marked as using global fallback or no eligible policy.
- Unit and CLI tests pass.
- Notebook 16 inference behavior remains unchanged.
- `PROJECT_STATE.md` is updated with the new supported calibration command and current result summary.

## Assumptions

- Adapter retraining is out of scope.
- Dataset relabeling and `strawberry__fruit` performance diagnosis are out of scope for this calibration step.
- Current default risk constraints are `min_capture=0.70` and `max_false_positive_rate=0.15`.
- Target-specific calibration is preferred when evidence is sufficient.
- Calibration is advisory first; runtime behavior will not change until a separate integration step.
- Existing docs/state notes should be committed with this work unless split commits are requested.
