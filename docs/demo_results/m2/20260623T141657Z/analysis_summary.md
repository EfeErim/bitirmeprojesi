# M2 Demo Analysis Summary

- total: 512
- router_crop_correctness: `{"correct": 371, "incorrect": 141, "not_applicable": 0}`
- router_part_correctness: `{"correct": 350, "incorrect": 162, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 251, "incorrect": 29, "not_applicable": 232}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 232, "accept_router": 148, "use_prototype": 132}`
- answered_wrong_by_target: `{"apricot__fruit": 2, "apricot__leaf": 1, "grape__fruit": 1, "grape__leaf": 9, "grape__unknown_part": 1, "strawberry__fruit": 8, "strawberry__leaf": 2, "tomato__fruit": 4, "tomato__leaf": 1}`
- prototype_correct_but_abstained: 208 `{"apricot__fruit": 52, "apricot__leaf": 2, "grape__fruit": 25, "grape__leaf": 1, "strawberry__fruit": 15, "tomato__fruit": 10, "tomato__leaf": 103}`
- negative_false_accepts: 1 `{"grape__unknown_part": 1}`
- opposite_part_disease_labels: 4

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 2 | 52 | 0 | 54 | 0 | 2 |
| apricot__leaf | 37 | 35 | 2 | 34 | 3 | 34 | 0 |
| grape__fruit | 55 | 30 | 25 | 29 | 26 | 29 | 0 |
| grape__leaf | 77 | 75 | 2 | 66 | 11 | 66 | 0 |
| grape__unknown_part | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 32 | 15 | 24 | 23 | 24 | 0 |
| strawberry__leaf | 45 | 45 | 0 | 43 | 2 | 43 | 1 |
| tomato__fruit | 75 | 59 | 16 | 55 | 20 | 55 | 0 |
| tomato__leaf | 108 | 1 | 107 | 0 | 108 | 0 | 1 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
