# Router Calibration Recovery Plan

## Summary

Notebook 5 now runs, but the latest calibration result did not find an eligible router configuration. The best dev-set candidate still had `negative_false_accept_rate=0.3098` and `part_recall=0.2821`, failing the target `negative_false_accept_rate <= 0.05` and the part-recall regression guard. The next step is not another blind threshold increase; it is measured failure analysis plus an adaptive router-policy search.

Success criteria:

- `data/router_eval` produces an eligible recommendation with `negative_false_accept_rate <= 0.05`.
- `part_recall`, `wrong_part_rejection_rate`, and latency stay within the existing regression guards.
- The accepted dev candidate also passes `data/router_eval_holdout`.
- Notebook 5 publishes results under `runs/_index/router_calibration/<timestamp>/`.

## Key Changes

- Add calibration failure analysis to `scripts/calibrate_router_surface.py`.
  - Report false accepts by `off_crop`, `non_plant`, `ambiguous`, and `wrong_part`.
  - Include top predicted crops/parts, confidence/margin distributions, input-guard score distributions, and the hardest false-accept examples.
  - Mark explicit failure causes such as `input_guard_not_separating_negatives` or `guard_too_aggressive`.

- Keep config preview conservative.
  - Show only `eligible=true` recommendations.
  - If no eligible candidate exists, print the best rejected variant, its reasons, and the failure-analysis summary.

- Add adaptive router-policy search.
  - Keep cache/replay for safe gates: `router_min_confidence`, `router_min_margin`, and input-guard thresholds.
  - Use successive halving for heavy router-policy parameters: `vlm_confidence_threshold`, `global_crop_context_weight`, `sam3_mask_threshold`, `part_open_set_min_confidence`, and `part_open_set_margin`.
  - Evaluate heavy candidates on a balanced subset first, then promote only the best candidates to full `data/router_eval`, then validate eligible candidates on holdout.

- Make Notebook 5 results durable.
  - Keep `git add -f` for `runs/_index/router_calibration/<timestamp>/`.
  - Commit/push both accepted and rejected calibration summaries so failed searches still leave useful evidence.

## Test Plan

- Unit-test failure-analysis summaries for `negative_false_accept_rate_above_target` and `part_recall_drop`.
- Unit-test config preview behavior so ineligible variants are never shown as applyable config.
- Unit-test successive-halving promotion so only top candidates reach full eval.
- Run:

```powershell
.\scripts\python.cmd -m pytest tests/unit/utils/test_calibrate_router_surface.py -q
.\scripts\python.cmd scripts/validate_notebook_imports.py
.\scripts\python.cmd scripts/validate_config_schema.py
.\scripts\python.cmd -m py_compile scripts/calibrate_router_surface.py scripts/notebook_cells/nb5_cell*.py
```

## Assumptions

- Keep the target `negative_false_accept_rate <= 0.05`.
- Do not use holdout for optimization; use it only for final validation.
- Do not apply any config when no eligible candidate exists.
- Keep Notebook 5 as the calibration surface; Notebook 1 and Notebook 8 should consume calibrated behavior rather than fork calibration logic.
