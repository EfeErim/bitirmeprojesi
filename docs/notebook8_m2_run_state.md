# Notebook 8 M2 Run State

This file exists so run-mode settings do not live only in chat. After every pushed Notebook 8 M2 result, update this note and `PROJECT_STATE.md` if the next run mode changes.

Machine-readable source of truth: `docs/notebook8_m2_run_state.json`. Notebook 8 applies that JSON automatically when `M2_AUTO_APPLY_RUN_STATE = True`, but explicit visible-cell M2 settings are treated as operator overrides. The JSON only fills run-state fields that the visible cell did not define, so manual Colab edits such as `M2_RUN_PROBLEM_ONLY_DEMO`, `M2_REFRESH_HANDOFF_CACHE`, `M2_BATCH_SIZE`, `M2_ADAPTER_BATCH_SIZE`, manifests, and comparison baselines are preserved in the run artifacts.

## Current Next Run

- Mode: full active-manifest run
- `M2_RUN_PROBLEM_ONLY_DEMO = False`
- `M2_REFRESH_HANDOFF_CACHE = True`
- `M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION = True`
- `M2_BATCH_SIZE = 6`
- `M2_ADAPTER_BATCH_SIZE = 12`
- Baseline for full-run acceptance: `docs/demo_results/m2/20260629T124253Z/summary.json`
- Open-world router production gate: `M2_RUN_OPEN_WORLD_ROUTER_VALIDATION = True`
- Open-world baseline summary: `docs/demo_results/m2/20260629T124253Z/summary.json`
- Prototype curation root: `docs/demo_assets/prototype_curation/20260629T124253Z_router_refinement`
- Active full manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Open-world manifest: `docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv`
- Problem-only manifest retained for diagnostics: `docs/demo_assets/m2_problem_only_manifests/20260628T113313Z_router_failures.csv`

Reason: the `20260629T124253Z` full run improved router/prototype quality but left `51` answered-wrong disease/class rows and `31` router/prototype failures. The next gate is a full rerun after the reviewed router-only residue was merged into `docs/demo_assets/prototype_curation/20260629T124253Z_router_refinement`, followed by the open-world router readiness gate over the balanced supported manifest and the `306`-row fresh negative manifest. Keep handoff refresh enabled and use the lower `6/12` batch sizes for this first post-curation/open-world run.

## Automatic Post-Run Adjustment Rule

After every new `docs/demo_results/m2/<timestamp>/` folder is pushed:

1. Read the newest compact artifacts in this order:
   - `summary.json`
   - `m2_result_comparison.md` when present
   - `analysis_summary.md` or `m2_problem_only_analysis_summary.md`
   - `m2_demo_checklist_run.md` or `m2_problem_only_demo_checklist_run.md`
2. Check safety first:
   - Keep or restore `M2_REFRESH_HANDOFF_CACHE = True` if negative false accepts increase, opposite-part disease labels reopen, manifest/prototype/calibration/code changed, or cache provenance is unclear.
   - Do not switch to a full-manifest acceptance run when the problem-only run has new safety regressions.
3. Check whether the latest run was only a diagnostic:
   - If problem-only metrics improved and safety stayed clean, run one more problem-only diagnostic only when another code or curation change is planned.
   - If problem-only metrics plateau and safety stayed clean, switch the next run to full-manifest mode.
4. Decide cache mode:
   - Use `M2_REFRESH_HANDOFF_CACHE = True` for the first run after any behavior-changing edit, curation-root change, prototype-bank rebuild, calibration-constraint change, manifest change, or baseline change.
   - Use `M2_REFRESH_HANDOFF_CACHE = False` only for repeated same-manifest, same-prototype, same-calibration, same-code checks after one fresh successful run proves the behavior.
5. Keep calibration reuse enabled unless intentionally invalid:
   - Keep `M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION = True` for same manifest, prototype bank, and constraints.
   - Expect automatic recalibration when hashes or constraints differ.
6. Keep batch sizes stable:
   - Keep `M2_BATCH_SIZE = 6` and `M2_ADAPTER_BATCH_SIZE = 12`.
   - If a Colab operator changes visible-cell M2 settings for memory safety or diagnostics, those explicit values override the JSON for that run.
   - Do not raise router batch size to 16 until a fresh run over all images in the active full manifest proves memory stability and quality.
7. Update this file's "Current Next Run" section and the relevant `PROJECT_STATE.md` next-step bullet whenever the recommended next run mode changes.

## Mode Decision Table

| Situation after latest run | Next mode | Cache refresh |
|---|---|---|
| Code, curation, prototype, calibration, or manifest changed | Problem-only if available; otherwise full | `True` |
| Problem-only improved and safety stayed clean, but another small fix is planned | Problem-only | `True` for first run after fix |
| Problem-only improved, safety stayed clean, and no more small fix is planned | Full active-manifest run | `True` |
| Same artifacts and code, repeat check only | Same previous mode | `False` |
| Negative false accepts or opposite-part labels increased | Problem-only or targeted analysis; no full acceptance | `True` |
| Full active-manifest run passed comparison and no behavior changed afterward | Same-manifest repeat only if needed | `False` |

## Current Caution

Problem-only runs are speed diagnostics, not final evidence. Final acceptance still requires a Notebook 8 run over all images in the active full manifest and comparison against the configured baseline.
The open-world router plan is also not final evidence until the same Colab/GPU run writes a passing `docs/demo_results/router_open_world/<timestamp>/router_open_world_readiness.json` and that folder passes `scripts/validate_router_open_world_result.py`.
