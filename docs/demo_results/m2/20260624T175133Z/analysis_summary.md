# M2 Demo Analysis Summary

- total: 602
- router_crop_correctness: `{"correct": 490, "incorrect": 112, "not_applicable": 0}`
- router_part_correctness: `{"correct": 459, "incorrect": 143, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 365, "incorrect": 54, "not_applicable": 183}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 183, "accept_router": 195, "use_prototype": 224}`
- answered_wrong_by_target: `{"apricot__fruit": 8, "apricot__leaf": 2, "grape__fruit": 12, "grape__leaf": 12, "strawberry__fruit": 8, "strawberry__leaf": 3, "tomato__fruit": 3, "tomato__leaf": 6}`
- prototype_correct_but_abstained: 117 `{"apricot__fruit": 21, "apricot__leaf": 9, "grape__fruit": 8, "grape__leaf": 5, "strawberry__fruit": 22, "strawberry__leaf": 6, "tomato__fruit": 23, "tomato__leaf": 23}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 12

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 64 | 39 | 25 | 31 | 33 | 31 | 8 |
| apricot__leaf | 47 | 32 | 15 | 30 | 17 | 30 | 0 |
| grape__fruit | 65 | 54 | 11 | 42 | 23 | 42 | 3 |
| grape__leaf | 87 | 78 | 9 | 66 | 21 | 66 | 0 |
| grape__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 57 | 32 | 25 | 24 | 33 | 24 | 0 |
| strawberry__leaf | 55 | 46 | 9 | 43 | 12 | 43 | 0 |
| tomato__fruit | 85 | 55 | 30 | 52 | 33 | 52 | 0 |
| tomato__leaf | 118 | 83 | 35 | 77 | 41 | 77 | 1 |
| tomato__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| unknown_crop | 14 | 0 | 14 | 14 | 0 | 0 | 0 |
