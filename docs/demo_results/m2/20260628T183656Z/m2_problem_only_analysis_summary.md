# M2 Demo Analysis Summary

- total: 89
- router_crop_correctness: `{"correct": 54, "incorrect": 35, "not_applicable": 0}`
- router_part_correctness: `{"correct": 52, "incorrect": 37, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 32, "incorrect": 6, "not_applicable": 51}`
- classless_supported_probes: `{"answered": 0, "answered_target_correct": 0, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 0, "total": 0}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 51, "accept_router": 1, "use_prototype": 37}`
- answered_wrong_by_target: `{"apricot__fruit": 1, "strawberry__fruit": 5}`
- prototype_correct_but_abstained: 36 `{"apricot__fruit": 6, "apricot__leaf": 2, "grape__fruit": 3, "grape__leaf": 1, "strawberry__fruit": 2, "tomato__fruit": 5, "tomato__leaf": 17}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 14 | 6 | 8 | 5 | 9 | 5 | 0 |
| apricot__leaf | 6 | 4 | 2 | 4 | 2 | 4 | 0 |
| grape__fruit | 4 | 1 | 3 | 1 | 3 | 1 | 0 |
| grape__leaf | 2 | 0 | 2 | 0 | 2 | 0 | 0 |
| strawberry__fruit | 16 | 14 | 2 | 9 | 7 | 9 | 0 |
| strawberry__leaf | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| tomato__fruit | 19 | 8 | 11 | 8 | 11 | 8 | 0 |
| tomato__leaf | 27 | 5 | 22 | 5 | 22 | 5 | 0 |
