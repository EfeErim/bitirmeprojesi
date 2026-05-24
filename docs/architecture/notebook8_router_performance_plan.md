# Notebook 8 Router Performance Improvement Plan

## Summary

Improve Notebook 8 by making router handoff safer and measurable before changing defaults. Optimize first for crop/part correctness, low false adapter loads, and useful abstention (`router_uncertain`, `part=unknown`), with latency tracked as a guardrail rather than the primary goal.

## Key Changes

- Build a representative `data/router_eval/` set using the existing contract:
  - `id/<crop>/<part>/*` for valid crop-part inputs.
  - `negatives/off_crop/<label>/*` and `negatives/non_plant/<label>/*` for false-accept pressure.
  - `ambiguous/<label>/*` for cases expected to abstain.
  - `wrong_part/<crop>/<unsupported_part>/*` for crop-correct but part-unsupported cases.
- Extend router diagnostics output, if needed, so every evaluated sample records top-k crop candidates, crop margin, predicted part, raw part label, part unknown confidence, rejection reason, detection count, and latency.
- Keep diagnostic data in eval JSON outputs, not notebook-only print text.
- Use the existing calibration flow:
  - Run `scripts/evaluate_router_surface.py` for baseline metrics.
  - Run `scripts/calibrate_router_surface.py` with the `quick` preset first.
  - Use the `docs` preset only after the eval set is large enough to justify broader sweeps.
- Accept a new router config only if it improves or preserves negative false-accept rate, crop ID accuracy, part non-unknown precision and recall, wrong-part rejection, risk-coverage, and p95 latency within a measured tolerance.

## Implementation Details

- Do not restrict router crop labels to only supported adapters. The router should still reason over the broader taxonomy.
- Do not add one-off prompts for individual failures. Prompt changes must be generic by crop/part family and validated by the router eval set.
- Keep Notebook 8 as a thin wrapper. Improvements should happen in router runtime, evaluation scripts, config calibration, or diagnostics.
- Use Notebook 5 as the user-facing calibration surface; Notebook 8 should consume the calibrated router behavior.
- If benchmark evidence shows prompt-only routing remains weak, add a second phase with optional BioCLIP prototype evidence:
  - Use clean reference images per crop/part.
  - Fuse prototype similarity with prompt scores and global/ROI evidence.
  - Keep this optional until it beats calibrated prompt-only routing on the same eval set.

## Test Plan

- Unit tests:
  - Router diagnostics payload preserves top-k crop and part rejection evidence.
  - Router eval sample parsing covers `id`, `off_crop`, `non_plant`, `ambiguous`, and `wrong_part`.
  - Calibration ranking prefers lower false accepts before raw accuracy gains.
- Integration checks:
  - `.\scripts\python.cmd scripts/validate_notebook_imports.py`
  - `.\scripts\python.cmd scripts/validate_config_schema.py`
  - `.\scripts\python.cmd -m pytest tests/unit/router tests/unit/utils/test_evaluate_router_surface.py tests/unit/utils/test_calibrate_router_surface.py -q`
- Benchmark acceptance:
  - Baseline `router_eval.json` exists.
  - Calibration report includes baseline, recommended variant, sample count, and ranked variants.
  - Recommended config is not applied unless it passes the safety-first metric gates.

## Assumptions

- Priority is accuracy and safety over speed.
- The first implementation should use existing scripts and Notebook 5 rather than creating a new calibration workflow.
- The eval set can start small but must include negatives and wrong-part samples; ID-only evaluation is not enough.
- Literature-backed rationale stays in maintained docs if defaults or methodology change.
