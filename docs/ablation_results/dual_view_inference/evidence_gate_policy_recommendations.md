# Evidence Gate Policy Recommendations

Calibration artifact: `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`
Failure analysis artifact: `docs/ablation_results/dual_view_inference/notebook16_failure_analysis.json`

## Summary

- target count: `8`
- target-specific candidates: `2`
- group fallback candidates: `4`
- global fallback candidates: `0`
- no eligible policy: `2`
- global policy status: `no_eligible_policy`

## Recommended Report-Only Candidates

| Target | Recommendation | Source | Threshold | Calibration capture | Holdout capture | Holdout false-positive | Missed wrong | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tomato__leaf` | `report_only_candidate` | `target_specific` | `0.9500` | `0.7356` | `0.6471` | `0.1364` | `84` | `` |
| `grape__leaf` | `report_only_candidate` | `target_specific` | `0.8000` | `0.7143` | `0.6364` | `0.1786` | `16` | `` |
| `apricot__leaf` | `report_only_candidate` | `group:leaf_targets` | `0.9500` | `0.7803` | `1.0000` | `0.5000` | `13` | `target_selection_no_eligible_or_unstable_policy` |
| `apricot__fruit` | `report_only_candidate` | `group:apricot_targets` | `0.9000` | `0.7037` | `0.6667` | `0.0241` | `12` | `target_wrong_count_below_min_target_errors:14<20` |
| `tomato__fruit` | `report_only_candidate` | `group:tomato_targets` | `0.9500` | `0.7292` | `1.0000` | `0.3158` | `5` | `target_wrong_count_below_min_target_errors:12<20` |
| `strawberry__leaf` | `report_only_candidate` | `group:leaf_targets` | `0.9500` | `0.7803` | `n/a` | `0.0000` | `1` | `target_wrong_count_below_min_target_errors:1<20` |

## Audit Required

| Target | Reason | Wrong | Missed wrong | Calibration note |
| --- | --- | ---: | ---: | --- |
| `strawberry__fruit` | `no_safe_target_group_or_global_policy` | `112` | `84` | `target_selection_no_eligible_or_unstable_policy` |
| `grape__fruit` | `no_safe_target_group_or_global_policy` | `16` | `6` | `target_wrong_count_below_min_target_errors:16<20` |

## Decision

- Treat this as report-only policy guidance.
- Do not hardcode per-adapter policy decisions manually.
- Do not change runtime inference without a separate validation and promotion step.
