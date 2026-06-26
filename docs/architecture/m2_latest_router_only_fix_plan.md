# M2 Latest Router-Only Fix Plan

## Summary

This plan uses the latest committed M2 run, `docs/demo_results/m2/20260625T224351Z/`, as the source of truth.

Scope is router/prototype failures only: `95` rows out of the `133` total failures. The `38` answered-wrong adapter/class rows are intentionally deferred and must stay out of this pass.

Goal: reduce failures from `prototype_evidence_weak`, `prototype_policy_not_calibrated`, `negative_prototype_too_close`, and `part_conflict` without weakening safety gates.

## Current Failure Targets

Latest run: `20260625T224351Z`

- Total rows: `602`
- Passed: `469`
- Failed: `133`
- Router/prototype-side failures: `95`
- Answered-wrong adapter/class failures deferred: `38`
- Negative false accepts: `0`

Router-only failure reasons:

| Reason | Count |
| --- | ---: |
| `prototype_evidence_weak` | 37 |
| `prototype_policy_not_calibrated` | 21 |
| `negative_prototype_too_close` | 20 |
| `part_conflict` | 17 |

Target priority:

1. `tomato__leaf`
2. `tomato__fruit`
3. `apricot__fruit`
4. `strawberry__fruit`
5. `grape__fruit` only for residual `part_conflict`

## Key Changes

- Generate a new hard-example audit for `20260625T224351Z`, focused on router/prototype failures only.
- Exclude rows where `actual_status == success`; those are adapter/class mistakes and belong to a later adapter-hard-example pass.
- Use review decisions only from this set:
  - `add_prototype_positive`
  - `add_prototype_hard_negative`
  - `exclude_ambiguous`
  - `keep`
- Write reviewed curation outputs under `docs/demo_assets/prototype_curation/20260625T224351Z/`.
- Update Notebook 8's `M2_PROTOTYPE_CURATION_ROOT` to `docs/demo_assets/prototype_curation/20260625T224351Z` only after reviewed manifests exist and the local prototype-bank smoke build passes.

## Implementation Result

- Added router-only export support to `scripts/export_m2_hard_example_audit.py`; sparse result folders can now be read from `HEAD:` when `m2_demo_checklist_run.json` is not materialized.
- Exported `docs/demo_results/m2/20260625T224351Z/hard_example_audit.csv` with exactly `95` failed router/prototype rows and no answered-wrong adapter/class rows.
- Added `scripts/review_m2_router_only_audit.py` for conservative router-only review decisions using only `add_prototype_positive`, `add_prototype_hard_negative`, and `exclude_ambiguous`.
- Applied reviewed decisions to `docs/demo_assets/prototype_curation/20260625T224351Z/`: `71` prototype positives, `8` cross-target hard negatives, `16` ambiguous exclusions, `0` adapter hard examples, `0` relabels, and `pending_review_count=0`.
- Local smoke build passed with `curation_positive_count=71`, `hard_negative_count=8`, `skipped_count=0`, and targets `apricot__fruit`, `apricot__leaf`, `grape__fruit`, `strawberry__fruit`, `tomato__fruit`, and `tomato__leaf`.
- Notebook 8 now points `M2_PROTOTYPE_CURATION_ROOT` and `M2_COMPARISON_BASELINE` at the `20260625T224351Z` router-only pass.

## Implementation Steps

1. Access latest result artifacts for `20260625T224351Z`. The repo may be sparse, so use Git object reads if `docs/demo_results/m2/20260625T224351Z/` is not materialized in the working tree.
2. Export a router-only hard-example audit from `m2_demo_checklist_run.json`, including only failed rows with these `reconcile_reason` values:
   - `prototype_evidence_weak`
   - `prototype_policy_not_calibrated`
   - `negative_prototype_too_close`
   - `part_conflict`
3. Review packet rows with these rules:
   - For `prototype_evidence_weak`, add positives only when the image is visually clear, target-correct, and suitable as a class/target prototype.
   - For `prototype_policy_not_calibrated`, add positives for underrepresented target/class rows; exclude ambiguous rows.
   - For `negative_prototype_too_close`, audit whether the hard negative is same-target; avoid adding same-target disease confusions as hard negatives.
   - For `part_conflict`, add clean fruit positives for `strawberry__fruit` and `grape__fruit` only when the image visibly supports fruit-target routing.
4. Apply reviewed decisions to produce curation manifests in `docs/demo_assets/prototype_curation/20260625T224351Z/`.
5. Build a local prototype-bank smoke artifact with:

```powershell
./scripts/python.cmd scripts/build_router_prototype_bank.py --no-adapter-discovery --output-root .runtime_tmp/router_prototype_curation_latest --run-id 20260625T224351Z-router-only-local --max-images-per-class 50 --curation-root docs/demo_assets/prototype_curation/20260625T224351Z
```

6. If the smoke output has nonzero curated positives and no unexpected skipped images, update Notebook 8, extracted notebook cell script, validation checks, and `PROJECT_STATE.md` to point at the new curation root.

## Test Plan

- Export latest router-only audit and confirm selected candidates are bounded by the `95` router/prototype failures.
- Run the apply script with `--require-reviewed`; it must fail until all selected rows have decisions.
- Run targeted tests:

```powershell
./scripts/python.cmd -m pytest tests/unit/scripts/test_export_m2_hard_example_audit.py tests/unit/scripts/test_apply_m2_hard_example_audit_decisions.py tests/unit/router/test_prototype_bank.py tests/unit/router/test_prototype_reconciler.py -q
```

- Run Notebook 8 validation after any curation-root surface changes:

```powershell
./scripts/python.cmd scripts/validate_notebook_imports.py
```

## Assumptions

- This is router-only by design; adapter/class wrong-answer fixes are deferred.
- No broad threshold loosening is allowed in this pass.
- Safety gates must stay unchanged: `negative_false_accepts` must remain `0`, opposite-part guards must not be weakened, and unsupported/unknown rows must stay protected.
- A later full Notebook 8 Colab/GPU run is still required for final metric proof, but no rerun is needed before this curation work starts.
