# M2 Demo Analysis Summary

- total: 602
- router_crop_correctness: `{"correct": 474, "incorrect": 128, "not_applicable": 0}`
- router_part_correctness: `{"correct": 444, "incorrect": 158, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 348, "incorrect": 34, "not_applicable": 220}`
- classless_supported_probes: `{"answered": 8, "answered_target_correct": 8, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 72, "total": 80}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 209, "accept_router": 187, "use_prototype": 206}`
- answered_wrong_by_target: `{"apricot__fruit": 7, "apricot__leaf": 1, "grape__fruit": 5, "grape__leaf": 9, "strawberry__fruit": 8, "strawberry__leaf": 1, "tomato__fruit": 3}`
- prototype_correct_but_abstained: 142 `{"apricot__fruit": 22, "apricot__leaf": 16, "grape__fruit": 9, "grape__leaf": 5, "strawberry__fruit": 24, "strawberry__leaf": 8, "tomato__fruit": 22, "tomato__leaf": 36}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 7

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 64 | 38 | 26 | 41 | 23 | 31 | 7 |
| apricot__leaf | 47 | 25 | 22 | 33 | 14 | 23 | 0 |
| grape__fruit | 65 | 50 | 15 | 52 | 13 | 42 | 0 |
| grape__leaf | 87 | 78 | 9 | 76 | 11 | 66 | 0 |
| grape__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 57 | 30 | 27 | 32 | 25 | 22 | 0 |
| strawberry__leaf | 55 | 44 | 11 | 53 | 2 | 43 | 0 |
| tomato__fruit | 85 | 56 | 29 | 63 | 22 | 53 | 0 |
| tomato__leaf | 118 | 69 | 49 | 78 | 40 | 68 | 0 |
| tomato__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| unknown_crop | 14 | 0 | 14 | 14 | 0 | 0 | 0 |
