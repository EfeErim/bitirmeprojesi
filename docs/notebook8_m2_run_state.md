# Notebook 8 M2 Run State

This file exists so run-mode settings do not live only in chat. After every pushed Notebook 8 M2 result, update this note and `PROJECT_STATE.md` if the next run mode changes.

## Current Next Run

- Mode: problem-only diagnostic
- `M2_RUN_PROBLEM_ONLY_DEMO = True`
- `M2_REFRESH_HANDOFF_CACHE = True`
- `M2_REUSE_EXISTING_PROTOTYPE_CALIBRATION = True`
- `M2_BATCH_SIZE = 12`
- `M2_ADAPTER_BATCH_SIZE = 32`
- Baseline for full-run acceptance: `docs/demo_results/m2/20260628T113313Z/summary.json`
- Problem-only manifest: `docs/demo_assets/m2_problem_only_manifests/20260628T113313Z_router_failures.csv`

Reason: the latest code changed manifest-exact prototype rescue behavior. The next diagnostic must recompute handoff decisions once before any cache reuse.

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
   - Keep `M2_BATCH_SIZE = 12` and `M2_ADAPTER_BATCH_SIZE = 32`.
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
