# M2 Demo Result Comparison

- Status: `pass`
- Baseline: `20260625T224351Z`
- Candidate: `20260628T113313Z`
- Baseline manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Candidate manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Baseline manifest SHA-256: `a143763fa927e5b6a30c5733eb57951dacbb0f5a73c45c55eb9032ef7ed0e4b5`
- Candidate manifest SHA-256: `a143763fa927e5b6a30c5733eb57951dacbb0f5a73c45c55eb9032ef7ed0e4b5`
- Baseline manifest SHA-256 source: `local_manifest_enriched`
- Candidate manifest SHA-256 source: `local_manifest_enriched`

## Metric Deltas

| Metric | Delta |
| --- | ---: |
| `total_delta` | 0 |
| `passed_delta` | 4 |
| `failed_delta` | -4 |
| `router_failure_delta` | -6 |
| `negative_false_accept_delta` | 0 |
| `opposite_part_delta` | -7 |
| `prototype_correct_but_abstained_delta` | -7 |

## Focus Target Deltas

| Target | Total Delta | Pass Delta | Fail Delta |
| --- | ---: | ---: | ---: |
| `tomato__leaf` | 0 | 1 | -1 |
| `tomato__fruit` | 0 | 0 | 0 |
| `apricot__fruit` | 0 | 1 | -1 |
| `strawberry__fruit` | 0 | 0 | 0 |

## Checks

| Check | Result |
| --- | --- |
| `manifests_match` | pass |
| `manifest_sha256_match` | pass |
| `totals_match` | pass |
| `focus_target_totals_match` | pass |
| `failed_not_increased` | pass |
| `router_failures_not_increased` | pass |
| `negative_false_accepts_not_increased` | pass |
| `opposite_part_not_increased` | pass |
| `at_least_one_focus_target_improved` | pass |

## Warnings

- `baseline_manifest_sha256_enriched_from_local_manifest`
- `candidate_manifest_sha256_enriched_from_local_manifest`
