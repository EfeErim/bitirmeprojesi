# M2 Demo Analysis Summary

- total: 602
- router_crop_correctness: `{"correct": 491, "incorrect": 111, "not_applicable": 0}`
- router_part_correctness: `{"correct": 463, "incorrect": 139, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 369, "incorrect": 40, "not_applicable": 193}`
- classless_supported_probes: `{"answered": 9, "answered_target_correct": 9, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 71, "total": 80}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 180, "accept_router": 195, "use_prototype": 227}`
- answered_wrong_by_target: `{"apricot__fruit": 8, "apricot__leaf": 1, "grape__fruit": 7, "grape__leaf": 9, "strawberry__fruit": 7, "strawberry__leaf": 1, "tomato__fruit": 3, "tomato__leaf": 4}`
- prototype_correct_but_abstained: 114 `{"apricot__fruit": 20, "apricot__leaf": 9, "grape__fruit": 6, "grape__leaf": 5, "strawberry__fruit": 23, "strawberry__leaf": 6, "tomato__fruit": 22, "tomato__leaf": 23}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 64 | 40 | 24 | 42 | 22 | 32 | 0 |
| apricot__leaf | 47 | 32 | 15 | 40 | 7 | 30 | 0 |
| grape__fruit | 65 | 53 | 12 | 54 | 11 | 44 | 0 |
| grape__leaf | 87 | 78 | 9 | 76 | 11 | 66 | 0 |
| grape__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 57 | 31 | 26 | 34 | 23 | 24 | 0 |
| strawberry__leaf | 55 | 46 | 9 | 53 | 2 | 43 | 0 |
| tomato__fruit | 85 | 56 | 29 | 63 | 22 | 53 | 0 |
| tomato__leaf | 118 | 82 | 36 | 87 | 31 | 77 | 0 |
| tomato__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| unknown_crop | 14 | 0 | 14 | 14 | 0 | 0 | 0 |
