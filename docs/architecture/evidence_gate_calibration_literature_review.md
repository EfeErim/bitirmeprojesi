# Evidence Gate Calibration Literature Review

Date: 2026-06-13

## Problem

Notebook 16 shows that manual per-adapter review thresholds do not scale. The latest dual-view report has 2,946 samples, 0.8836 accuracy, and 0.8563 macro-F1, but the current review gate captures only 0.3557 of wrong full-image decisions. Router failures were not observed in that run (`router=0`), so the next automation surface is adapter-aware evidence gate calibration rather than router rewrite.

## Literature Takeaways

1. **Selective classification is the right framing.** Geifman and El-Yaniv frame the problem as accepting predictions only when the expected selective risk stays below a target, maximizing coverage under a risk constraint. This maps directly to "answer automatically or send to review" decisions. Source: [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500).

2. **Confidence should be calibrated before thresholding.** Guo et al. show modern neural networks are often miscalibrated and that temperature scaling is a simple post-processing baseline. This supports per-adapter confidence calibration before using confidence as a review signal. Source: [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).

3. **Risk-controlling prediction sets generalize fixed thresholds.** Bates et al. use a holdout set to choose prediction sets that control expected loss at a user-specified level. For this project, the same idea can control "missed wrong prediction" or "review false positive" instead of hand-picking a threshold. Source: [Distribution-Free, Risk-Controlling Prediction Sets](https://arxiv.org/abs/2101.02703).

4. **Conformal Risk Control and Learn-then-Test fit multi-objective gates.** Conformal Risk Control supports monotone losses, and Learn-then-Test calibrates algorithms with finite-sample risk guarantees using held-out data. This fits a grid of candidate gate policies, then accepting only policies that satisfy review-capture and false-positive constraints. Sources: [Conformal Risk Control](https://arxiv.org/abs/2208.02814), [Learn then Test](https://arxiv.org/abs/2110.01052).

5. **Group-conditional calibration is needed for adapter scale.** A single global threshold is weak because Notebook 16 targets behave differently (`strawberry__fruit` is the obvious outlier). Mondrian/group-conditional conformal methods motivate calibrating by group, where group means adapter target such as `tomato__leaf` or `grape__fruit`. Sources: [Mondrian conformal prediction overview](https://mapie.readthedocs.io/en/v0.9.0/theoretical_description_mondrian.html), [Kandinsky Conformal Prediction](https://proceedings.mlr.press/v267/bairaktari25a.html).

6. **If deployment distribution drifts, recalibration must be repeatable.** Adaptive conformal inference treats calibration as an ongoing update problem under distribution shift. This supports rerunning calibration whenever new Notebook 16 reports or new adapter exports land. Source: [Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).

## Recommended Automation Design

Build an `evidence_gate_calibration` surface that consumes Notebook 16 rows and produces a versioned JSON policy per adapter target.

Inputs:

- `docs/ablation_results/dual_view_inference/multi_target_report.json`
- target key: `crop__part`
- row fields: expected label, full diagnosis/confidence, ROI evidence status, ROI quality, OOD flags, review flag, failure-analysis reasons

Candidate policy family:

- full-confidence threshold
- ROI/full conflict as optional review signal
- ROI quality flags as optional review signal
- full/ROI OOD flags as optional review signal
- target-specific constraints and fallback to global pooled policy when sample count is too low

Optimization target:

- maximize coverage, meaning fewer unnecessary reviews
- subject to minimum wrong-prediction capture, for example `>= 0.70`
- subject to maximum false-positive review rate, for example `<= 0.15`
- report no eligible policy if constraints cannot be met

Validation protocol:

- split each target's Notebook 16 rows into calibration and holdout partitions
- select policy on calibration only
- report holdout capture, false-positive review rate, coverage, and sample count
- never tune on final hidden test evidence when it exists

Output:

```text
docs/ablation_results/dual_view_inference/evidence_gate_calibration.json
```

Expected schema:

```json
{
  "schema_version": "v1_evidence_gate_calibration",
  "source_report": "docs/ablation_results/dual_view_inference/multi_target_report.json",
  "global_policy": {},
  "target_policies": {
    "tomato__leaf": {
      "status": "eligible",
      "policy": {},
      "calibration_metrics": {},
      "holdout_metrics": {}
    }
  }
}
```

## Decision

Do not manually tune each adapter. Implement an automated, target-aware calibration script that searches a small, auditable policy grid and selects the highest-coverage policy satisfying risk constraints. Use selective classification/risk-control as the conceptual basis, temperature scaling as optional confidence preprocessing, and group-conditional calibration as the scaling rule.
