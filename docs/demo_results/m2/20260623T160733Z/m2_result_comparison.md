# M2 Demo Result Comparison

- Status: `fail`
- Baseline: `20260622T161859Z`
- Candidate: `20260623T160733Z`
- Baseline manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Candidate manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Baseline manifest SHA-256: `2b889522459d8611d43206636034d380583b71efd3ca13e1ae3be35387eafbef`
- Candidate manifest SHA-256: `2b889522459d8611d43206636034d380583b71efd3ca13e1ae3be35387eafbef`

## Metric Deltas

| Metric | Delta |
| --- | ---: |
| `total_delta` | 0 |
| `passed_delta` | -54 |
| `failed_delta` | 54 |
| `router_failure_delta` | 78 |
| `negative_false_accept_delta` | -5 |
| `opposite_part_delta` | -11 |
| `prototype_correct_but_abstained_delta` | 83 |

## Focus Target Deltas

| Target | Total Delta | Pass Delta | Fail Delta |
| --- | ---: | ---: | ---: |
| `apricot__fruit` | 0 | -35 | 35 |
| `grape__fruit` | 0 | 5 | -5 |
| `tomato__leaf` | 0 | -22 | 22 |

## Checks

| Check | Result |
| --- | --- |
| `manifests_match` | pass |
| `manifest_sha256_match` | pass |
| `totals_match` | pass |
| `focus_target_totals_match` | pass |
| `failed_not_increased` | fail |
| `router_failures_not_increased` | fail |
| `negative_false_accepts_not_increased` | pass |
| `opposite_part_not_increased` | pass |
| `at_least_one_focus_target_improved` | pass |
