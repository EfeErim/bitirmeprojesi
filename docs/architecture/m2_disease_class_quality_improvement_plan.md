# M2 Disease-Class Quality Improvement Plan

## Summary

The latest full M2 run, `docs/demo_results/m2/20260629T124253Z/`, is a strong router/prototype improvement but not final demo quality.

- Total rows: `602`
- Passed: `520`
- Failed: `82`
- Answered-wrong disease/class rows: `51`
- Router/prototype failures: `31`
- Negative false accepts: `0`
- Opposite-part disease labels: `0`

The next improvement phase should reduce disease-class mistakes while preserving the current safety counters. Do not broadly relax router/prototype thresholds in this phase.

## Implementation Status

Implemented locally:

- `scripts/export_m2_hard_example_audit.py --answered-wrong-only`
- `docs/demo_results/m2/20260629T124253Z/answered_wrong_audit.csv` with `51` rows
- `.runtime_tmp/m2_answered_wrong_audit/20260629T124253Z/` review packets grouped by target and expected/predicted class pair
- `scripts/review_m2_answered_wrong_audit.py` for conservative draft adapter-hard-example decisions
- visual review of all `51` answered-wrong rows
- `docs/demo_assets/prototype_curation/20260629T124253Z/adapter_hard_example_manifest.csv` with `28` adapter hard examples
- `docs/demo_assets/prototype_curation/20260629T124253Z/excluded_ambiguous_rows.csv` with `14` ambiguous or low-signal rows
- `docs/demo_assets/prototype_curation/20260629T124253Z/relabel_manifest.csv` with `9` likely label corrections
- Notebook 8 next-run baseline and batch settings synchronized to `20260629T124253Z`, `M2_BATCH_SIZE = 6`, and `M2_ADAPTER_BATCH_SIZE = 12`

Still external:

- target-adapter retraining/fine-tuning from the reviewed adapter hard examples
- the follow-up full Notebook 8 M2 Colab/GPU rerun and comparison against `20260629T124253Z`

## Goal

Reduce answered-wrong disease/class failures on the same 602-row M2 manifest by auditing, curating, and retraining from reviewed hard examples.

Success criteria for the next full Notebook 8 M2 run:

- `negative_false_accepts = 0`
- `opposite_part_disease_labels = 0`
- answered-wrong disease/class rows decrease below `51`
- total failures decrease below `82`
- comparison against `docs/demo_results/m2/20260629T124253Z/summary.json` passes on the same manifest

## Deliverables And Expected Outcomes

Deliverables:

1. `answered_wrong_audit.csv`
   - Focused CSV for the `51` disease/class wrong rows from `20260629T124253Z`.
   - Grouped and ranked by target, expected disease, predicted disease, prototype evidence, and source image.
2. Review packets
   - Human-reviewable packets under `.runtime_tmp/m2_answered_wrong_audit/20260629T124253Z/`.
   - Used to decide whether each row is a true hard example, ambiguous, mislabeled, or useful for prototype curation.
3. New curation folder
   - `docs/demo_assets/prototype_curation/20260629T124253Z/`
   - Contains adapter hard examples, prototype positives, prototype hard negatives, ambiguous exclusions, relabels, and a curation summary.
4. Target adapter hard-example set
   - Reviewed examples for retraining or fine-tuning weak adapters, especially `strawberry__fruit`, `grape__leaf`, `apricot__fruit`, `grape__fruit`, and `tomato__leaf`.
5. Follow-up full M2 result
   - Same 602-row manifest.
   - Baseline: `docs/demo_results/m2/20260629T124253Z/summary.json`.
   - Run with `M2_BATCH_SIZE = 6` and `M2_ADAPTER_BATCH_SIZE = 12`.

Expected outcomes:

- Main improvement is fewer disease-class mistakes: answered-wrong count should drop below `51`.
- Total failures should drop below `82`.
- Safety must remain unchanged: `negative_false_accepts = 0` and `opposite_part_disease_labels = 0`.
- Router/prototype thresholds should stay stable; gains should come from adapter/data quality, not looser safety gates.
- The review output should clearly separate true adapter weaknesses, ambiguous images, label/manifest issues, adapter-training examples, and prototype-curation examples.

## Current Failure Shape

The remaining failures are no longer mainly router handoff failures.

| Failure group | Count |
| --- | ---: |
| Answered-wrong disease/class | 51 |
| Router/prototype failure | 31 |

Answered-wrong rows by target:

| Target | Count |
| --- | ---: |
| `strawberry__fruit` | 12 |
| `grape__leaf` | 9 |
| `apricot__fruit` | 9 |
| `grape__fruit` | 8 |
| `tomato__leaf` | 7 |
| `tomato__fruit` | 4 |
| `apricot__leaf` | 1 |
| `strawberry__leaf` | 1 |

High-priority class-confusion groups:

- `strawberry__fruit`: healthy/unripe, powdery mildew, gray mold, anthracnose.
- `grape__leaf`: mildew, powdery mildew, anthracnose, esca, leafroll, fanleaf.
- `grape__fruit`: botrytis, mildew, powdery mildew, healthy.
- `apricot__fruit`: sharka, peach scab, shot-hole/leaf-hole fruit symptoms, healthy.
- `tomato__leaf`: bacterial spot/speck, early blight, late blight, septoria, leaf mold.

## Implementation Plan

### 1. Export an answered-wrong audit

Extend `scripts/export_m2_hard_example_audit.py` with an `--answered-wrong-only` mode.

Inputs:

- `docs/demo_results/m2/20260629T124253Z/m2_demo_checklist_run.json`

Filtering rules:

- Include rows with `pass_fail = fail`, no `failure_bucket`, and supported `expected_target`.
- Exclude router-only failures, classless supported probes, unsupported/unknown rows, and non-plant rows.
- Keep all eight supported targets by default.

Outputs:

- `docs/demo_results/m2/20260629T124253Z/answered_wrong_audit.csv`
- `.runtime_tmp/m2_answered_wrong_audit/20260629T124253Z/packet_summary.json`
- `.runtime_tmp/m2_answered_wrong_audit/20260629T124253Z/index.html`
- per-target/class-pair review packet CSVs

Required audit fields:

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
review_decision
corrected_class
prototype_quality
adapter_training_quality
review_notes
```

Ranking rules:

- Prioritize targets with the most answered-wrong rows.
- Prioritize repeated expected/predicted class-pair confusions.
- Prioritize rows where crop/part are correct but disease is wrong.
- Keep visually ambiguous or label-suspicious rows visible, but do not auto-use them for training.

### 2. Review and convert rows into curation manifests

Use the existing `scripts/apply_m2_hard_example_audit_decisions.py` workflow, extending it only where needed for answered-wrong audit outputs.

Each reviewed row must map to exactly one decision:

```text
add_adapter_train
add_prototype_positive
add_prototype_hard_negative
exclude_ambiguous
relabel:<class>
keep
```

Decision policy:

- `add_adapter_train`: clear crop/part-correct disease mistake; main path for the 51 answered-wrong rows.
- `exclude_ambiguous`: visually ambiguous rows or rows whose disease signal is not reliable.
- `relabel:<class>`: manifest label is visibly wrong; write corrected class to curation output but do not mutate source data.
- `add_prototype_positive`: only clear examples where prototype target/class and reviewer agree.
- `add_prototype_hard_negative`: visually close wrong-class or wrong-target rows useful as guards.
- `keep`: row is understood but should not affect training/prototype artifacts.

Expected output root:

```text
docs/demo_assets/prototype_curation/20260629T124253Z/
```

Expected files:

```text
adapter_hard_example_manifest.csv
prototype_positive_manifest.csv
prototype_hard_negative_manifest.csv
excluded_ambiguous_rows.csv
relabel_manifest.csv
curation_summary.json
```

### 3. Retrain or fine-tune target adapters from reviewed hard examples

Prioritize target adapters by answered-wrong count and demo impact:

1. `strawberry__fruit`
2. `grape__leaf`
3. `apricot__fruit`
4. `grape__fruit`
5. `tomato__leaf`

Training rules:

- Use only reviewed `add_adapter_train` rows and accepted relabels.
- Keep ambiguous rows out of training and prototype positives.
- Preserve target-specific adapter boundaries; do not mix fruit/leaf surfaces.
- Keep the current router/prototype safety settings unchanged during adapter improvement.

### 4. Rerun the same full M2 manifest

Run Notebook 8 on the same active manifest:

```python
M2_RUN_FULL_DEMO = True
M2_RUN_PROBLEM_ONLY_DEMO = False
M2_BATCH_SIZE = 6
M2_ADAPTER_BATCH_SIZE = 12
M2_REFRESH_HANDOFF_CACHE = True
M2_COMPARISON_BASELINE = 'docs/demo_results/m2/20260629T124253Z/summary.json'
```

Compare the new result against `20260629T124253Z`.

## Guardrails

- Do not broadly relax global prototype similarity, margin, negative-gap, or false-accept constraints.
- Do not use prototype disease labels as a blanket override for adapter disease labels; prior analysis showed this would break too many currently correct adapter predictions.
- Do not treat classless supported probes as disease-class positives unless a reviewer supplies a corrected class.
- Keep unsupported, unknown-part, unknown-crop, and non-plant rows as safety probes.
- Do not mutate source datasets during audit/apply; write review and curation manifests only.

## Validation Plan

Local validation:

- `./scripts/python.cmd -m py_compile scripts/export_m2_hard_example_audit.py scripts/apply_m2_hard_example_audit_decisions.py`
- targeted unit tests for:
  - answered-wrong filtering
  - class-pair grouping
  - packet CSV generation
  - packet decision overlay
  - curation manifest generation
  - `--require-reviewed` failure on pending rows

Run validation:

- Full M2 run uses the same manifest SHA as `20260629T124253Z`.
- `negative_false_accepts` remains `0`.
- `opposite_part_disease_labels` remains `0`.
- answered-wrong rows decrease below `51`.
- total failures decrease below `82`.
