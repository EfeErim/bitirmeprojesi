# M2 Demo Analysis Summary

- total: 522
- router_crop_correctness: `{"correct": 444, "incorrect": 78, "not_applicable": 0}`
- router_part_correctness: `{"correct": 448, "incorrect": 74, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 377, "incorrect": 50, "not_applicable": 95}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 95, "accept_router": 188, "use_prototype": 239}`
- answered_wrong_by_target: `{"apricot__fruit": 10, "apricot__leaf": 1, "grape__fruit": 6, "grape__leaf": 9, "grape__unknown_part": 1, "strawberry__fruit": 12, "strawberry__leaf": 2, "tomato__fruit": 4, "tomato__leaf": 5}`
- prototype_correct_but_abstained: 61 `{"apricot__fruit": 13, "apricot__leaf": 2, "grape__fruit": 7, "grape__leaf": 1, "strawberry__fruit": 3, "tomato__fruit": 9, "tomato__leaf": 26}`
- negative_false_accepts: 1 `{"grape__unknown_part": 1}`
- opposite_part_disease_labels: 12

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 41 | 13 | 31 | 23 | 31 | 10 |
| apricot__leaf | 37 | 35 | 2 | 34 | 3 | 34 | 0 |
| grape__fruit | 55 | 48 | 7 | 42 | 13 | 42 | 0 |
| grape__leaf | 77 | 75 | 2 | 66 | 11 | 66 | 0 |
| grape__unknown_part | 2 | 1 | 1 | 1 | 1 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 44 | 3 | 32 | 15 | 32 | 0 |
| strawberry__leaf | 45 | 45 | 0 | 43 | 2 | 43 | 1 |
| tomato__fruit | 75 | 60 | 15 | 56 | 19 | 56 | 0 |
| tomato__leaf | 108 | 78 | 30 | 73 | 35 | 73 | 1 |
| tomato__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| unknown_crop | 14 | 0 | 14 | 14 | 0 | 0 | 0 |
