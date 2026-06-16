# Handoff Guide

Last updated: 2026-06-16

This guide is the company/evaluator handoff surface for the June 2026 finish. It should be completed during M3 after the demo checklist has been run.

## Read First

1. `README.md`
2. `docs/README.md`
3. `docs/project_completion_plan.md`
4. `docs/demo_checklist.md`
5. `docs/final_validation_checklist.md`

## What The Project Does

AADS v6 is a narrow plant-disease demo product. Given a plant photo, it routes the crop/part, loads the matching adapter when available, predicts the disease, and returns unknown/abstain behavior when the input is unsupported or unsafe.

## Final Demo Surface

- Primary: `colab_notebooks/8_auto_router_adapter_prediction.ipynb`
- Current M1 adapter root for demo runs: `ADAPTER_ROOT = ROOT / "runs"`. The default `models/adapters/` path is the deployment layout, but it is empty in this workspace snapshot; generated Colab exports are available under `runs/`.
- Fallback evidence path: `.runtime_tmp/final_demo_fallbacks/`
- Supporting notebooks:
  - Notebook 0: dataset preparation
  - Notebook 1: router crop/part identification
  - Notebook 2: adapter training
  - Notebook 3: direct adapter validation
  - Notebook 5: router calibration
  - Notebook 16: ROI/bbox evidence-gate analysis, report-only unless promotion gates pass

## Supported Targets

Fill this table after running `docs/demo_checklist.md`.

| Target | Final support label | Evidence | Limitation |
|---|---|---|---|
| tomato__fruit | TBD | TBD | TBD |
| tomato__leaf | TBD | TBD | TBD |
| strawberry__fruit | TBD | TBD | TBD |
| strawberry__leaf | TBD | TBD | TBD |
| grape__fruit | TBD | TBD | TBD |
| grape__leaf | TBD | TBD | TBD |
| apricot__fruit | TBD | TBD | TBD |
| apricot__leaf | TBD | TBD | TBD |

Support labels:

- `supported`: acceptable demo behavior with documented checklist evidence
- `low_confidence`: usable only with caution or review
- `experimental`: not presented as final supported behavior

## Demo Run Procedure

1. Open Notebook 8 in Colab.
2. Confirm repo access and dependency setup.
3. Set `ADAPTER_ROOT = ROOT / "runs"` unless final bundles have been copied into `models/adapters/`.
4. Set `ANALYSIS_IMAGE_PATH` for the current checklist image and keep `RETURN_OOD = True`.
5. Run the Notebook 1 router analysis cell, then the Notebook 8 adapter prediction cell.
6. Run the chosen demo images from `docs/demo_checklist.md`.
7. Record output status, predicted crop/part, disease, confidence/OOD evidence, and pass/fail result.
8. Use fallback screenshots/outputs if external access or GPU runtime fails.

## User Photo Guidance

Ask users to provide:

- one main plant or plant part in the image
- visible leaf or fruit surface
- reasonable lighting and focus
- minimal extreme blur, heavy occlusion, or multi-plant clutter
- one of the supported crops when they expect a disease answer: tomato, strawberry, grape, or apricot

Unsupported crops, unclear parts, non-plant images, and diseases outside the supported class set may return unknown/review instead of a disease.

## Generated And Local-Only Paths

Do not treat these as maintained source:

- `runs/`
- `models/adapters/`
- `outputs/`
- `data/prepared_runtime_datasets/`
- `.runtime_tmp/`

Use generated artifacts as evidence, reports, or local runtime assets only.

## Validation Commands

Use `docs/final_validation_checklist.md` as the source of truth. Minimum final handoff validation should include:

```powershell
.\scripts\python.cmd scripts\validate_notebook_imports.py
.\scripts\python.cmd scripts\validate_config_schema.py
.\scripts\python.cmd scripts\audit_code_organization.py
```

Run broader tests only when time and environment allow.

## Known Final Risks

- Some target surfaces may need `low_confidence` or `experimental` labeling after demo evidence.
- Gated external models such as SAM3 can block local calibration or evidence generation.
- Notebook 16 evidence-gate policies are report-only unless promotion gates pass.
- User-like photos can lower raw accuracy; the system should document photo guidance and report abstention coverage.

## Company Handoff Notes

Before handoff, fill in:

- final demo checklist result summary
- final supported target labels
- exact adapter root used for demo
- required Colab/runtime credentials or tokens
- known unsupported inputs
- fallback artifacts for presentation continuity
- final validation command outcomes
