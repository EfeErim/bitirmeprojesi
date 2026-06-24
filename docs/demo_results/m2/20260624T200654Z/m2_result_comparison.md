# M2 Demo Result Comparison

- Status: `fail`
- Baseline: `20260622T161859Z`
- Candidate: `20260624T200654Z`
- Baseline manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Candidate manifest: `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`
- Baseline manifest SHA-256: `a143763fa927e5b6a30c5733eb57951dacbb0f5a73c45c55eb9032ef7ed0e4b5`
- Candidate manifest SHA-256: `a143763fa927e5b6a30c5733eb57951dacbb0f5a73c45c55eb9032ef7ed0e4b5`
- Baseline manifest SHA-256 source: `local_manifest_enriched`
- Candidate manifest SHA-256 source: `local_manifest_enriched`

## Metric Deltas

| Metric | Delta |
| --- | ---: |
| `total_delta` | 90 |
| `passed_delta` | 112 |
| `failed_delta` | -22 |
| `router_failure_delta` | -9 |
| `negative_false_accept_delta` | -6 |
| `opposite_part_delta` | -7 |
| `prototype_correct_but_abstained_delta` | 31 |

## Focus Target Deltas

| Target | Total Delta | Pass Delta | Fail Delta |
| --- | ---: | ---: | ---: |
| `apricot__fruit` | 10 | 6 | 4 |
| `grape__fruit` | 10 | 28 | -18 |
| `tomato__leaf` | 10 | 26 | -16 |

## Checks

| Check | Result |
| --- | --- |
| `manifests_match` | pass |
| `manifest_sha256_match` | pass |
| `totals_match` | fail |
| `focus_target_totals_match` | fail |
| `failed_not_increased` | pass |
| `router_failures_not_increased` | pass |
| `negative_false_accepts_not_increased` | pass |
| `opposite_part_not_increased` | pass |
| `at_least_one_focus_target_improved` | pass |

## Warnings

- `manifest_hash_matches_but_total_rows_differ`
- `baseline_manifest_sha256_enriched_from_local_manifest`
- `candidate_manifest_sha256_enriched_from_local_manifest`
