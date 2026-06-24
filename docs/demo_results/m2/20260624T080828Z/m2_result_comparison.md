# M2 Demo Result Comparison

- Status: `fail`
- Baseline: `20260622T161859Z`
- Candidate: `20260624T080828Z`
- Baseline manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Candidate manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Baseline manifest SHA-256: `724a9413aa148e07981959722a105eb6010a623fa3ae3b5296ec8e7210b732d6`
- Candidate manifest SHA-256: `724a9413aa148e07981959722a105eb6010a623fa3ae3b5296ec8e7210b732d6`

## Metric Deltas

| Metric | Delta |
| --- | ---: |
| `total_delta` | 10 |
| `passed_delta` | 14 |
| `failed_delta` | -4 |
| `router_failure_delta` | 13 |
| `negative_false_accept_delta` | -5 |
| `opposite_part_delta` | -7 |
| `prototype_correct_but_abstained_delta` | 14 |

## Focus Target Deltas

| Target | Total Delta | Pass Delta | Fail Delta |
| --- | ---: | ---: | ---: |
| `apricot__fruit` | 0 | -4 | 4 |
| `grape__fruit` | 0 | 5 | -5 |
| `tomato__leaf` | 0 | 12 | -12 |

## Checks

| Check | Result |
| --- | --- |
| `manifests_match` | pass |
| `manifest_sha256_match` | pass |
| `totals_match` | fail |
| `focus_target_totals_match` | pass |
| `failed_not_increased` | pass |
| `router_failures_not_increased` | fail |
| `negative_false_accepts_not_increased` | pass |
| `opposite_part_not_increased` | pass |
| `at_least_one_focus_target_improved` | pass |
