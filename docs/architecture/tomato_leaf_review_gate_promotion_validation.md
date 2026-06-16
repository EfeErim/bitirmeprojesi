# Tomato Leaf Review-Gate Promotion Validation

Date: 2026-06-14

## Decision

`tomato__leaf` is the current report-only review-gate pilot candidate. Do not promote the `0.95` full-confidence threshold into Notebook 16 final-decision behavior or production runtime until the validation gates below pass.

The current evidence comes from:

- `docs/ablation_results/dual_view_inference/notebook16_failure_analysis.md`
- `docs/ablation_results/dual_view_inference/evidence_gate_policy_recommendations.md`
- `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`

Current pilot evidence:

- `tomato__leaf` calibration status: `target_specific`
- selected threshold: `full_confidence_threshold=0.95`
- missed wrong predictions before threshold simulation: `84`
- threshold simulation review capture: `0.7107`
- threshold simulation false-positive review: `0.1163`
- v2 holdout capture: `0.6471`
- v2 holdout false-positive review: `0.1364`
- full missed-wrong audit: `docs/ablation_results/dual_view_inference/tomato_leaf_missed_wrong_audit.csv`
- audit rows: `84` missed-wrong predictions, with `84/84` local files available
- grouped review packets: `.runtime_tmp/tomato_leaf_missed_wrong_packets/` currently has `27` confusion-pair packets plus `index.html`

## Promotion Gates

A runtime promotion is allowed only if all gates pass on a refreshed Notebook 16 multi-target report:

- Global policy remains rejected or advisory; no global gate is promoted through this target pilot.
- `tomato__leaf` remains `target_specific` in v2 calibration.
- `tomato__leaf` holdout wrong-capture rate is at least `0.70`.
- `tomato__leaf` holdout false-positive review rate is at most `0.15`.
- `tomato__leaf` total review rate is at most `0.25`.
- `tomato__leaf` calibration and holdout partitions each contain at least `5` wrong predictions.
- Dataset integrity checks do not report exact image-hash overlap for `tomato__leaf`.
- The promotion candidate is regenerated from the latest Notebook 16 report, not hand-edited in docs.

If any gate fails, keep the policy report-only and record the failure as validation evidence. Do not relax thresholds manually unless a new calibration plan explicitly changes the constraints.

## Validation Workflow

Run this sequence after a fresh Notebook 16 multi-target artifact is available:

```powershell
.\scripts\python.cmd scripts\summarize_large_report.py docs\ablation_results\dual_view_inference\multi_target_report.json
.\scripts\python.cmd scripts\calibrate_evidence_gate.py --schema-version v2
.\scripts\python.cmd scripts\analyze_notebook16_failures.py
.\scripts\python.cmd scripts\recommend_evidence_gate_policies.py
.\scripts\python.cmd scripts\export_notebook16_target_audit.py --target-id tomato__leaf
.\scripts\python.cmd scripts\apply_notebook16_target_audit_decisions.py --packet-dir .runtime_tmp\tomato_leaf_missed_wrong_packets
.\scripts\python.cmd scripts\monitor_dataset_integrity.py
```

Review the generated artifacts in this order:

1. `evidence_gate_calibration.json`: confirm target policy status and holdout metrics.
2. `notebook16_failure_analysis.md`: confirm `0.95` threshold simulation metrics and missed-wrong drilldown.
3. `evidence_gate_policy_recommendations.md`: confirm `tomato__leaf` remains report-only unless all gates pass.
4. `tomato_leaf_missed_wrong_audit.csv` or `.runtime_tmp/tomato_leaf_missed_wrong_packets/index.html`: review the full missed-wrong queue before changing data or thresholds.
5. `apply_notebook16_target_audit_decisions.py`: dry-run reviewed CSV decisions before using `--apply`; pass `--packet-dir .runtime_tmp\tomato_leaf_missed_wrong_packets` if decisions were entered in packet CSVs. Removals are quarantined, not deleted.
6. Dataset integrity output: confirm no exact image-hash overlap for `tomato__leaf`.

## Runtime Promotion Boundary

This document does not implement runtime behavior. The first runtime implementation, if the gates pass later, must be a separate change that:

- keeps full-image adapter prediction as the final diagnosis;
- only changes the `requires_review` decision for `tomato__leaf`;
- records the applied policy and source calibration artifact in the inference payload or report metadata;
- includes tests proving non-`tomato__leaf` targets are unaffected;
- leaves `strawberry__fruit` as a separate data/label audit target.

## Current Status

The current `tomato__leaf` pilot is not runtime-ready because the v2 holdout capture is `0.6471`, below the `0.70` promotion gate. The correct next state is report-only validation, not runtime promotion.

The immediate data action is to audit the full missed-wrong CSV or grouped packet sheets. Prioritize the high-count confusions before rerunning Notebook 16, especially `domates_early_blight_yaprak -> domates_late_blight_yaprak`, `domates_late_blight_yaprak -> domates_early_blight_yaprak`, and `domates_bacterial_spot_and_speck_yaprak -> domates_septoria_leaf_spot_yaprak`.
