# M2 Demo Analysis Summary

- total: 89
- router_crop_correctness: `{"correct": 70, "incorrect": 19, "not_applicable": 0}`
- router_part_correctness: `{"correct": 67, "incorrect": 22, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 48, "incorrect": 10, "not_applicable": 31}`
- classless_supported_probes: `{"answered": 0, "answered_target_correct": 0, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 0, "total": 0}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 31, "accept_router": 3, "use_prototype": 55}`
- answered_wrong_by_target: `{"apricot__fruit": 1, "strawberry__fruit": 5, "tomato__fruit": 1, "tomato__leaf": 3}`
- prototype_correct_but_abstained: 16 `{"apricot__fruit": 1, "grape__fruit": 3, "grape__leaf": 1, "strawberry__fruit": 2, "tomato__fruit": 2, "tomato__leaf": 7}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 14 | 11 | 3 | 10 | 4 | 10 | 0 |
| apricot__leaf | 6 | 6 | 0 | 6 | 0 | 6 | 0 |
| grape__fruit | 4 | 1 | 3 | 1 | 3 | 1 | 0 |
| grape__leaf | 2 | 0 | 2 | 0 | 2 | 0 | 0 |
| strawberry__fruit | 16 | 14 | 2 | 9 | 7 | 9 | 0 |
| strawberry__leaf | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| tomato__fruit | 19 | 11 | 8 | 10 | 9 | 10 | 0 |
| tomato__leaf | 27 | 15 | 12 | 12 | 15 | 12 | 0 |
