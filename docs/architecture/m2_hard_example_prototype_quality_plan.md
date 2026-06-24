# M2 Hard-Example And Prototype-Quality Plan

## Summary

The latest M2 run improved safety but still is not final-pass quality. The next improvement pass should focus on three actions:

1. Stop broad threshold tuning.
2. Build a targeted hard-example audit set.
3. Improve prototype bank quality, not just quantity.

This plan intentionally keeps the current safety posture intact. Global prototype similarity, margin, negative-gap, and false-accept constraints should not be relaxed unless a later comparison proves that safety is preserved.

## Implementation Status

Implemented in this pass:

- `scripts/export_m2_hard_example_audit.py` ranks supported hard examples, writes `hard_example_audit.csv`, and creates per-target review packets plus a static packet index.
- `scripts/apply_m2_hard_example_audit_decisions.py` overlays packet-level review decisions and writes dry-run curation manifests without moving or deleting source images.
- Unit coverage exists in `tests/unit/scripts/test_export_m2_hard_example_audit.py` and `tests/unit/scripts/test_apply_m2_hard_example_audit_decisions.py`.
- The first exported audit is `docs/demo_results/m2/20260624T200654Z/hard_example_audit.csv` with `150` ranked rows and packet outputs under `.runtime_tmp/m2_hard_example_audit/20260624T200654Z/`.
- `--require-reviewed` is expected to return nonzero until every selected row has a reviewer decision.
- The first review pass is complete: packet decisions were filled conservatively, `--require-reviewed` passes with `pending_review_count=0`, and dry-run manifests were written under `docs/demo_assets/prototype_curation/20260624T200654Z/`.
- Manifest counts from that pass are `23` prototype positives, `50` prototype hard negatives, `23` adapter hard examples, `41` ambiguous exclusions, and `0` relabeled rows.
- Prototype-bank curation ingestion is implemented behind the explicit `--curation-root` builder flag. Curated positives are added as `split=curated` prototype samples, while hard negatives are stored separately in `hard_negative_prototypes` and used only by the existing prototype negative-gap abstention gate.
- Notebook 8 now exposes `M2_PROTOTYPE_CURATION_ROOT = 'docs/demo_assets/prototype_curation/20260624T200654Z'` and passes `--curation-root` during auto-build. When this value is set, Notebook 8 skips reuse of older committed prototype artifacts so the curated prototype bank is actually rebuilt.
- Local curated build validation passed with `./scripts/python.cmd scripts/build_router_prototype_bank.py --no-adapter-discovery --output-root .runtime_tmp/router_prototype_curation_local --run-id 20260624T200654Z-curated-local --max-images-per-class 50 --curation-root docs/demo_assets/prototype_curation/20260624T200654Z`: `curation_positive_count=23`, `hard_negative_count=50`, and `skipped_count=0`.

## Current Evidence

Latest analyzed run: `docs/demo_results/m2/20260624T200654Z/`.

- Total rows: `602`
- Passed: `469`
- Failed: `133`
- Answered: `415`
- Abstained or reviewed: `187`
- Negative false accepts: `0`
- Opposite-part disease labels: `8`
- Router/reconciliation failures: `93`
- Answered-wrong adapter/class failures: `40`

The main failing surfaces are:

| Target | Failures |
| --- | ---: |
| `tomato__leaf` | 31 |
| `tomato__fruit` | 23 |
| `apricot__fruit` | 23 |
| `strawberry__fruit` | 23 |

The dominant router/reconciliation reasons are:

| Reason | Count |
| --- | ---: |
| `prototype_evidence_weak` | 38 |
| `prototype_policy_not_calibrated` | 22 |
| `part_conflict` | 17 |
| `negative_prototype_too_close` | 16 |

## Action 1: Stop Broad Threshold Tuning

Do not lower global thresholds as the next fix. The run already has `0` negative false accepts, so global relaxation is more likely to reopen unsafe accepts than solve the remaining class and prototype-quality problems.

Required guardrail:

- Keep current M2 safety defaults unchanged.
- Do not relax global prototype similarity, margin, negative-gap, `max_negative_false_accepts`, or opposite-part guards in this pass.
- Before any future threshold change, compare against `docs/demo_results/m2/20260624T200654Z/summary.json` and require:
  - negative false accepts stay at `0`
  - opposite-part disease labels do not increase
  - failed rows do not increase
  - router failures do not increase
  - focus-target failures improve on at least one of `tomato__leaf`, `tomato__fruit`, `apricot__fruit`, `strawberry__fruit`

Implementation target:

- Add this guardrail to the M2 workflow docs when implementing the active-learning tooling.
- Keep threshold changes out of the first hard-example/prototype-quality patch.

## Action 2: Build A Targeted Hard-Example Audit Set

Add a focused exporter for M2 hard examples.

Proposed script:

```text
scripts/export_m2_hard_example_audit.py
```

Inputs:

- `docs/demo_results/m2/<timestamp>/m2_demo_checklist_run.json`
- default timestamp: newest folder under `docs/demo_results/m2`
- default targets:
  - `tomato__leaf`
  - `tomato__fruit`
  - `apricot__fruit`
  - `strawberry__fruit`

Outputs:

- `docs/demo_results/m2/<timestamp>/hard_example_audit.csv`
- `.runtime_tmp/m2_hard_example_audit/<timestamp>/packet_summary.json`
- `.runtime_tmp/m2_hard_example_audit/<timestamp>/index.html`
- optional contact sheets and per-packet `review_rows.csv` files

Candidate sources:

- answered-wrong rows
- `prototype_evidence_weak`
- `prototype_policy_not_calibrated`
- `part_conflict`
- `negative_prototype_too_close`
- low `prototype_margin`
- high-similarity wrong prototype target/class rows
- prototype-correct-but-abstained rows

Required CSV columns:

```text
rank
priority_score
priority_reasons
image_id
resolved_image
source
expected_target
expected_crop
expected_part
expected_class
actual_status
predicted_crop
predicted_part
predicted_disease
prototype_target
prototype_class_label
prototype_similarity
prototype_margin
reconcile_reason
failure_bucket
pass_fail
review_decision
corrected_crop
corrected_part
corrected_class
prototype_quality
adapter_training_quality
review_notes
```

Ranking policy:

- answered-wrong supported rows rank highest
- target surfaces listed above get a priority boost
- router failures rank by reason severity
- low margin and high wrong-target similarity increase priority
- unsupported/unknown rows are excluded unless explicitly requested

Review decisions:

```text
keep
exclude_ambiguous
relabel:<class>
add_prototype_positive
add_prototype_hard_negative
add_adapter_train
```

## Action 3: Improve Prototype Bank Quality

Add a dry-run apply script that turns reviewed audit rows into curated manifests. This pass should not move or delete dataset files.

Proposed script:

```text
scripts/apply_m2_hard_example_audit_decisions.py
```

Inputs:

- `hard_example_audit.csv`
- optional packet directory containing edited `review_rows.csv`

Outputs:

```text
docs/demo_assets/prototype_curation/<timestamp>/prototype_positive_manifest.csv
docs/demo_assets/prototype_curation/<timestamp>/prototype_hard_negative_manifest.csv
docs/demo_assets/prototype_curation/<timestamp>/adapter_hard_example_manifest.csv
docs/demo_assets/prototype_curation/<timestamp>/excluded_ambiguous_rows.csv
docs/demo_assets/prototype_curation/<timestamp>/curation_summary.json
```

Manifest rules:

- `add_prototype_positive`: use only reviewed, target-correct, visually clear rows.
- `add_prototype_hard_negative`: use visually close wrong-target rows to protect against unsafe overrides.
- `add_adapter_train`: use rows where crop/part are correct but disease class is wrong or underrepresented.
- `exclude_ambiguous`: keep the row out of prototype and retraining manifests.
- `relabel:<class>`: write corrected class to the manifest, but do not mutate the source dataset in this pass.

Prototype-quality rules:

- Favor clear canonical examples over bulk additions.
- Preserve target and class provenance.
- Keep fruit/leaf lookalikes as hard negatives instead of positive examples.
- Keep reviewed examples separate from raw M2/demo assets.
- Do not use classless supported probes as disease-class positives unless a reviewer supplies a corrected class.

## Validation Plan

Add unit tests for:

- candidate ranking
- target filtering
- missing margin/similarity handling
- CSV and packet output shape
- packet decision overlay
- dry-run manifest generation
- unsupported review decisions
- `--require-reviewed` nonzero exit when selected rows remain unreviewed

Expected validation commands:

```powershell
./scripts/python.cmd scripts/export_m2_hard_example_audit.py --run-id 20260624T200654Z --limit 150
./scripts/python.cmd scripts/apply_m2_hard_example_audit_decisions.py --decisions-csv docs/demo_results/m2/20260624T200654Z/hard_example_audit.csv --require-reviewed
./scripts/python.cmd -m pytest tests/unit/scripts/test_export_m2_hard_example_audit.py tests/unit/scripts/test_apply_m2_hard_example_audit_decisions.py -q
./scripts/python.cmd -m ruff check scripts/export_m2_hard_example_audit.py scripts/apply_m2_hard_example_audit_decisions.py tests/unit/scripts/test_export_m2_hard_example_audit.py tests/unit/scripts/test_apply_m2_hard_example_audit_decisions.py
```

The apply command above is a review gate: it writes empty dry-run manifests and returns nonzero while `review_decision` is still blank.

## Acceptance Criteria

- A reviewer can open packet contact sheets and fill `review_decision` without reading raw JSON.
- The apply script produces curated manifests without moving or deleting files.
- The curated manifests separate positive prototype additions, hard negatives, adapter hard examples, and ambiguous exclusions.
- The next prototype-bank rebuild can consume reviewed manifests in a later patch.
- No global threshold or runtime safety setting changes in this pass.

## Next Implementation Step

Run Notebook 8 M2 in Colab/GPU with the current visible defaults so the curated BioCLIP prototype bank is rebuilt, calibration is recomputed against the new prototype-bank hash, and results are published. Then compare the fresh run against `docs/demo_results/m2/20260624T200654Z/summary.json` before promoting any runtime behavior.
