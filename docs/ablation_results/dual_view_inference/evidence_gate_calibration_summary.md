# Evidence Gate Calibration Summary

Date: 2026-06-13

Source report: `docs/ablation_results/dual_view_inference/multi_target_report.json`
Calibration artifact: `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`

## Result

The v1 calibration pass did not find a safe global evidence-gate policy under the default constraints:

- minimum wrong-prediction capture: `0.70`
- maximum false-positive review rate: `0.15`
- minimum target errors for target-specific selection: `20`

The best rejected global policy captured many wrong predictions, but it would review too many correct predictions:

- wrong capture rate: `0.8136`
- false-positive review rate: `0.4416`
- coverage: `0.5154`

This means the repo should not switch to a global aggressive ROI-quality or ROI-conflict review rule.

## Target Policies

Only `grape__leaf` produced an eligible target-specific calibration policy on the calibration split:

- policy: `full_confidence_threshold=0.80`
- ROI conflict review: disabled
- ROI quality review: disabled
- full OOD review: disabled
- ROI OOD review: disabled
- calibration wrong capture rate: `0.7143`
- calibration false-positive review rate: `0.1294`
- holdout wrong capture rate: `0.6364`
- holdout false-positive review rate: `0.1786`

The holdout metrics are weaker than the calibration metrics, so this remains advisory evidence. It should not be promoted into runtime behavior without a separate validation step.

All other targets are explicitly `no_eligible_policy`:

- `apricot__fruit`: too few target errors (`14 < 20`)
- `apricot__leaf`: target-specific search failed the risk constraints
- `grape__fruit`: too few target errors (`16 < 20`)
- `strawberry__fruit`: target-specific search failed the risk constraints
- `strawberry__leaf`: too few target errors (`1 < 20`)
- `tomato__fruit`: too few target errors (`12 < 20`)
- `tomato__leaf`: target-specific search failed the risk constraints

## Interpretation

The current Notebook 16 result does not justify a router rewrite. The multi-target failure analysis reported `router=0`, while adapter/data, bbox/evidence, confidence/OOD, and review-gate buckets remain active.

The calibration artifact also does not justify a global runtime evidence-gate change. The global candidate that captures enough wrong predictions creates an unacceptable review burden on correct predictions.

The next adapter/data investigation should stay separate from this calibration step. The obvious later TODO is `strawberry__fruit`, where the latest Notebook 16 summary recorded `204` samples, `112` errors, and frequent confusion between healthy strawberry fruit and `strawberry_unripe_fruit`.

## Decision

Keep v1 calibration advisory only:

- do not change Notebook 16 final decision behavior;
- do not change production runtime behavior;
- keep full-image adapter prediction as the final decision;
- use ROI/bbox evidence for reporting and review analysis only;
- continue target-specific calibration experiments instead of applying a global aggressive gate.
