# M2 Demo Analysis Summary

- total: 512
- router_crop_correctness: `{"correct": 248, "incorrect": 264, "not_applicable": 0}`
- router_part_correctness: `{"correct": 223, "incorrect": 289, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 178, "incorrect": 55, "not_applicable": 279}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"disabled": 512}`
- answered_wrong_by_target: `{"grape__fruit": 23, "grape__leaf": 8, "grape__unknown_part": 1, "strawberry__fruit": 16, "strawberry__leaf": 1, "tomato__fruit": 1, "tomato__leaf": 5}`
- prototype_correct_but_abstained: 0
- negative_false_accepts: 1
- opposite_part_disease_labels: 38

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 0 | 54 | 0 | 54 | 0 | 0 |
| apricot__leaf | 37 | 4 | 33 | 4 | 33 | 4 | 0 |
| grape__fruit | 55 | 35 | 20 | 12 | 43 | 12 | 23 |
| grape__leaf | 77 | 73 | 4 | 65 | 12 | 65 | 0 |
| grape__unknown_part | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 24 | 23 | 8 | 39 | 8 | 12 |
| strawberry__leaf | 45 | 42 | 3 | 41 | 4 | 41 | 1 |
| tomato__fruit | 75 | 6 | 69 | 5 | 70 | 5 | 1 |
| tomato__leaf | 108 | 48 | 60 | 43 | 65 | 43 | 1 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
