# Eight Surface Completion Plan

Last updated: 2026-06-17

This plan supersedes the weaker "defensible delivery only" path for M2/M3 work. The current owner preference is to fix all eight final crop/part surfaces, not merely label weak surfaces as experimental.

## Current Evidence

Latest official Notebook 8 M2 run:

- Report: `docs/demo_results/m2/20260617T080846Z/m2_demo_checklist_run.json`
- Summary: `docs/demo_results/m2/20260617T080846Z/summary.json`
- Total rows: 512
- Passed: 261
- Failed: 251
- Answered: 246
- Abstained or reviewed: 266
- Main failure buckets:
  - router: 223
  - adapter_loading: 26

Per-target result:

| Target | Pass / Total | Current status |
|---|---:|---|
| `grape__leaf` | 68 / 77 | strongest surface |
| `strawberry__leaf` | 43 / 45 | strongest surface |
| `tomato__leaf` | 65 / 108 | usable but router failures remain |
| `grape__fruit` | 31 / 55 | mixed; fruit examples often get leaf labels |
| `strawberry__fruit` | 22 / 47 | mixed; fruit examples often get leaf labels |
| `tomato__fruit` | 12 / 75 | weak; router often predicts eggplant or unknown |
| `apricot__leaf` | 8 / 37 | weak |
| `apricot__fruit` | 0 / 54 | blocker |

Interpretation: the first blocker is router handoff, not merely adapter disease accuracy. Many `adapter_unavailable` rows are caused by router outputs such as `potato leaf`, `apple bud`, `pepper bud`, or `tomato whole plant`, so the runtime tries to load an unsupported adapter path.

## Target Definition Of Done

The eight-surface fix is complete only when a fresh Notebook 8 M2 full run satisfies all gates below:

- Every final target reaches at least 80% pass rate.
- `grape__leaf`, `strawberry__leaf`, and `tomato__leaf` target at least 90%.
- Supported targets have zero wrong-router `adapter_unavailable` failures.
- Fruit targets no longer systematically emit leaf/yaprak disease labels.
- `unknown_crop` and `non_plant` rows pass without forced disease labels.
- The final report is auto-pushed under `docs/demo_results/m2/<timestamp>/`.

## Implementation Plan

### 1. Make M2 Metrics Decision-Grade

- Extend the saved M2 manifest contract with optional `expected_crop`, `expected_part`, and `expected_class` columns.
- Update `scripts/run_demo_checklist.py` to report:
  - router crop correctness,
  - router part correctness,
  - normalized disease-class correctness,
  - opposite-part disease labels,
  - wrong-router adapter-unavailable failures versus real missing adapter failures,
  - per-target answer, abstain, fail, and exact-class counts.
- Write the analysis beside the normal report as `analysis_summary.json` and `analysis_summary.md`.

### 2. Fix Router Handoff For Final Supported Surfaces

- Add a final-demo router policy for Notebook 8 official mode.
- Limit final supported crops to `tomato`, `strawberry`, `grape`, and `apricot`.
- Limit final supported parts to `leaf` and `fruit`.
- If the router emits an unsupported crop, return `unknown_crop` or `router_uncertain` and do not load an adapter.
- If the router emits an unsupported part such as `bud`, `stem`, `whole plant`, `tuber`, `flower`, `root`, or `unknown`, return `router_uncertain` and do not load an adapter.
- Add tests proving unsupported crop/part outputs cannot become adapter-loading attempts.

### 3. Separate Router Failures From Adapter Failures

- Run `adapter-smoke` against the 512-image saved manifest with expected crop/part hints.
- Use the result to identify which surfaces fail even when router is bypassed.
- Treat adapter-smoke failures as adapter/class-index/data problems, not router problems.
- Keep official Notebook 8 run as the final acceptance surface.

### 4. Fix Adapter Class And Part Alignment

- Inspect adapter metadata, class indices, prepared dataset folders, and adapter root resolution for all fruit targets.
- For fruit adapters, fix any class-index or adapter-root issue that allows leaf/yaprak classes to appear as fruit predictions.
- If metadata is correct but behavior remains weak, retrain only the affected adapters in this priority order:
  1. `apricot__fruit`
  2. `tomato__fruit`
  3. `strawberry__fruit`
  4. `grape__fruit`
  5. `apricot__leaf`
- Do not add new model families or a web/mobile app.

### 5. Rerun And Close

- First run: Notebook 8 with `M2_DEMO_LIMIT = 32`.
- Second run: stratified subset with at least 20 rows per target.
- Final run: full 512 rows with `M2_DEMO_LIMIT = None`.
- Pull the auto-pushed result locally.
- Update:
  - `docs/demo_checklist.md`
  - `docs/handoff_guide.md`
  - `docs/final_validation_checklist.md`
  - `docs/presentation_outline.md`
  - `PROJECT_STATE.md`

## Validation

Required local checks after code changes:

- `.\scripts\python.cmd scripts\validate_notebook_imports.py`
- `.\scripts\python.cmd scripts\validate_config_schema.py`
- `.\scripts\python.cmd scripts\audit_code_organization.py`
- `.\scripts\python.cmd -m pytest tests/unit/scripts/test_run_demo_checklist.py -q`
- Targeted unit tests for router handoff and adapter discovery changes
- Targeted Ruff on changed Python files

Required Colab checks:

- Notebook 8 official run with `M2_DEMO_LIMIT = 32`
- Notebook 8 official full run with `M2_DEMO_LIMIT = None`
- Optional adapter-smoke run before the final official run to isolate adapter-only issues

## Assumptions

- Final delivery remains repo + Colab + PowerPoint + handoff docs.
- `runs` remains the Colab adapter root for demo.
- GitHub push branch is `master`.
- Colab has SAM3/Hugging Face access and `GH_TOKEN` or `GITHUB_TOKEN`.
- Notebook 16 remains report-only unless its separate promotion gates pass.
