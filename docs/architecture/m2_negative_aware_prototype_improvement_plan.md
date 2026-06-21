# M2 Negative-Aware Prototype Improvement Plan

## Summary

Latest M2 run `docs/demo_results/m2/20260621T182207Z/` shows that the taxonomy/prototype fix is working but the demo is not final-pass quality yet:

- Pass: `280/512`
- Fail: `232`
- Answered: `301`
- Abstain/review: `211`
- Prototype reconciliation: `accept_router=150`, `use_prototype=151`, `abstain=211`
- Opposite-part disease labels improved from `39` to `7`

The main remaining router/prototype issue is over-conservative margin gating. Of the `211` abstains, `182` have the correct prototype target, and `163` are blocked mainly because `prototype_margin < 0.04`.

The direct threshold candidate is `min_margin=0.02`: calibration on supported rows gives `coverage=0.842`, `precision=0.988`. However, lowering the margin also increases unsupported/unknown false accepts. The next implementation must therefore be negative-aware, not just a looser global threshold.

## Key Changes

### Negative-aware calibration

- Extend `scripts/calibrate_router_prototype_reconciler.py` so calibration separates:
  - positive supported targets: known `crop__leaf` / `crop__fruit` rows
  - negative targets: `unknown_crop`, `non_plant`, `*__unknown_part`, and unsupported/review rows
- Add calibration metrics:
  - `supported_precision`
  - `supported_coverage`
  - `supported_wrong`
  - `negative_false_accept_count`
  - `negative_false_accept_rate`
  - `non_plant_false_accept_count`
- Candidate eligibility:
  - `supported_precision >= 0.985`
  - `supported_coverage >= 0.80`
  - `non_plant_false_accept_count == 0`
- Candidate ranking:
  - highest supported coverage
  - lowest negative false accepts
  - highest supported precision
- Keep safe fallback behavior if no eligible policy is found.

### Negative prototype guard

- Add a reject/negative prototype group to the prototype evidence surface.
- Negative sources should include M2 manifest negatives first: `unknown_crop`, `non_plant`, and `*__unknown_part`.
- Runtime promotion must require:
  - positive prototype passes selected policy
  - positive target is sufficiently separated from nearest negative prototype
- Calibration should select and record a `min_negative_gap` threshold.

### Target-adaptive thresholds

- Emit target-level policy entries for all eight supported surfaces:
  - `tomato__fruit`, `tomato__leaf`
  - `grape__fruit`, `grape__leaf`
  - `strawberry__fruit`, `strawberry__leaf`
  - `apricot__fruit`, `apricot__leaf`
- Runtime applies target policy first, then global fallback.
- The expected first target is a safe move toward `min_margin=0.02` for high-coverage surfaces while keeping stricter gates where negative false accepts rise.

### Wrong-answer guard and analysis

- Add an adapter wrong-answer guard for opposite-part labels:
  - fruit target receiving leaf/yaprak diagnosis becomes review/uncertain
  - leaf target receiving fruit/meyve diagnosis becomes review/uncertain
- Extend `analysis_summary.json` with:
  - `answered_wrong_by_target`
  - `answered_wrong_by_expected_class`
  - `prototype_correct_but_abstained`
  - `negative_false_accepts`
  - `policy_thresholds_by_target`

### Adapter/class follow-up

After router/prototype improvements, generate an adapter-focused report for `success + fail` rows. Prioritize:

- `strawberry_powdery_mildew_fruit -> anthracnose/gray_mold`
- `grape mildew leaf -> powdery/esca`
- `tomato bacterial spot/speck -> septoria/late_blight`
- `apricot fruit -> healthy/leaf/far fruit class`

## Test Plan

- Unit tests:
  - calibration includes negative rows in false-accept metrics
  - coverage-first ranking selects broader safe policies
  - `margin=0.02` can be selected only when negative guard constraints pass
  - old calibration JSON remains readable
  - runtime refuses unsupported/non-plant promotion when negative prototype is too close
  - target policy takes precedence over global policy
- Validation:
  - `scripts/validate_notebook_imports.py`
  - targeted `ruff check`
  - run-demo/checklist unit tests
- Colab verification:
  - first run `M2_DEMO_LIMIT=64`
  - then full `512`
  - expected full-run direction:
    - `abstain < 211`
    - `use_prototype > 151`
    - unknown/non-plant false accepts do not increase
    - opposite-part disease labels near `0`
    - first pass target: `350+`

## Assumptions

- Runtime must not use expected labels or manifest targets to make predictions.
- Unsupported/non-plant safety is more important than inflating pass count.
- Existing prototype artifact reuse remains enabled for speed.
- For a final evidence run, `M2_PROTOTYPE_MAX_IMAGES_PER_CLASS=None` may be used to rebuild a full-quality prototype bank.
