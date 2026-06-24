# M2 Demo Analysis Summary

- total: 522
- router_crop_correctness: `{"correct": 434, "incorrect": 88, "not_applicable": 0}`
- router_part_correctness: `{"correct": 411, "incorrect": 111, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 348, "incorrect": 36, "not_applicable": 138}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 138, "accept_router": 186, "use_prototype": 198}`
- answered_wrong_by_target: `{"apricot__fruit": 8, "apricot__leaf": 1, "grape__fruit": 1, "grape__leaf": 9, "grape__unknown_part": 1, "strawberry__fruit": 8, "strawberry__leaf": 1, "tomato__fruit": 3, "tomato__leaf": 4}`
- prototype_correct_but_abstained: 100 `{"apricot__fruit": 13, "apricot__leaf": 6, "grape__fruit": 25, "grape__leaf": 1, "strawberry__fruit": 15, "tomato__fruit": 14, "tomato__leaf": 26}`
- negative_false_accepts: 1 `{"grape__unknown_part": 1}`
- opposite_part_disease_labels: 8

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 39 | 15 | 31 | 23 | 31 | 8 |
| apricot__leaf | 37 | 31 | 6 | 30 | 7 | 30 | 0 |
| grape__fruit | 55 | 30 | 25 | 29 | 26 | 29 | 0 |
| grape__leaf | 77 | 75 | 2 | 66 | 11 | 66 | 0 |
| grape__unknown_part | 2 | 1 | 1 | 1 | 1 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 32 | 15 | 24 | 23 | 24 | 0 |
| strawberry__leaf | 45 | 44 | 1 | 43 | 2 | 43 | 0 |
| tomato__fruit | 75 | 55 | 20 | 52 | 23 | 52 | 0 |
| tomato__leaf | 108 | 77 | 31 | 73 | 35 | 73 | 0 |
| tomato__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| unknown_crop | 14 | 0 | 14 | 14 | 0 | 0 | 0 |
