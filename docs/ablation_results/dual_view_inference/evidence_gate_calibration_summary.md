# Evidence Gate Calibration Summary

Date: 2026-06-13

Source report: `docs/ablation_results/dual_view_inference/multi_target_report.json`
Calibration artifact: `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`

## Result

The calibration artifact now uses `v2_evidence_gate_calibration`. V2 keeps the calibration advisory-only and does not change Notebook 16 final decisions or production runtime inference behavior.

The v2 pass still did not find a safe global evidence-gate policy under the default constraints:

- minimum wrong-prediction capture: `0.70`
- maximum false-positive review rate: `0.15`
- maximum review rate: `0.25`
- minimum target errors for target-specific selection: `20`
- minimum calibration errors: `10`
- minimum holdout errors: `5`

This means the repo should still not switch to a global aggressive ROI-quality, ROI-conflict, OOD, or confidence review rule.

## Selected Policies

V2 evaluates a wider confidence-threshold grid (`0.50`, `0.60`, `0.70`, `0.80`, `0.90`, `0.95`, `0.98`, `0.99`) plus ROI/OOD feature gates, then applies target/group/global fallback with holdout stability checks.

Target-specific policies:

- `grape__leaf`: `full_confidence_threshold=0.80`; calibration capture `0.7143`, calibration false-positive review `0.1294`, holdout capture `0.6364`, holdout false-positive review `0.1786`.
- `tomato__leaf`: `full_confidence_threshold=0.95`; calibration capture `0.7356`, calibration false-positive review `0.1076`, holdout capture `0.6471`, holdout false-positive review `0.1364`.

Group fallback policies:

- `apricot_targets`: `full_confidence_threshold=0.90`; calibration capture `0.7037`, calibration false-positive review `0.1236`, holdout capture `0.8667`, holdout false-positive review `0.0992`.
- `leaf_targets`: `full_confidence_threshold=0.95`; calibration capture `0.7803`, calibration false-positive review `0.1395`, holdout capture `0.7544`, holdout false-positive review `0.1424`.
- `tomato_targets`: `full_confidence_threshold=0.95`; calibration capture `0.7292`, calibration false-positive review `0.1338`, holdout capture `0.6757`, holdout false-positive review `0.1549`.

Target outcomes:

- `apricot__fruit`: `group_fallback` via `apricot_targets`; target errors are sparse (`14 < 20`).
- `apricot__leaf`: `group_fallback` via `leaf_targets`; target-specific selection was not safe/stable.
- `grape__fruit`: `no_eligible_policy`; target errors are sparse (`16 < 20`) and no safe group/global fallback exists.
- `grape__leaf`: `target_specific`.
- `strawberry__fruit`: `no_eligible_policy`; target-specific selection was not safe/stable and no safe group/global fallback exists.
- `strawberry__leaf`: `group_fallback` via `leaf_targets`; target errors are sparse (`1 < 20`).
- `tomato__fruit`: `group_fallback` via `tomato_targets`; target errors are sparse (`12 < 20`).
- `tomato__leaf`: `target_specific`.

## Audit Queue

V2 queues five targets for audit:

- `apricot__fruit`: sparse target errors and high-confidence errors dominate.
- `grape__fruit`: no eligible policy and sparse target errors.
- `strawberry__fruit`: no eligible policy and high-confidence errors dominate.
- `strawberry__leaf`: sparse target errors.
- `tomato__fruit`: sparse target errors.

## Interpretation

The v2 result strengthens the previous conclusion: calibration should stay target/group conditional and advisory. It finds useful report-only policies for leaf/tomato/apricot groups plus `grape__leaf` and `tomato__leaf`, but it still rejects a global runtime gate and explicitly leaves `grape__fruit` and `strawberry__fruit` as `no_eligible_policy`.

The next adapter/data investigation should stay separate from this calibration step. `strawberry__fruit` remains the clearest data/class-boundary issue because it has many errors and no safe calibration policy.

## Decision

Keep v2 calibration advisory only:

- do not change Notebook 16 final decision behavior;
- do not change production runtime behavior;
- keep full-image adapter prediction as the final decision;
- use ROI/bbox evidence and v2 calibration for reporting, review analysis, and audit prioritization only;
- continue target/group-specific calibration experiments instead of applying a global aggressive gate.
