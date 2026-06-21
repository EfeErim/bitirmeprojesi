# M2 Demo Analysis Summary

- total: 512
- router_crop_correctness: `{"correct": 436, "incorrect": 76, "not_applicable": 0}`
- router_part_correctness: `{"correct": 401, "incorrect": 111, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 325, "incorrect": 46, "not_applicable": 141}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 141, "accept_router": 166, "use_prototype": 205}`
- answered_wrong_by_target: `{"apricot__fruit": 8, "apricot__leaf": 1, "grape__fruit": 1, "grape__leaf": 8, "strawberry__fruit": 9, "strawberry__leaf": 1, "tomato__fruit": 4, "tomato__leaf": 8, "tomato__unknown_part": 1, "unknown_crop": 5}`
- prototype_correct_but_abstained: 118
- negative_false_accepts: 6
- opposite_part_disease_labels: 9

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 31 | 23 | 23 | 31 | 23 | 8 |
| apricot__leaf | 37 | 36 | 1 | 35 | 2 | 35 | 0 |
| grape__fruit | 55 | 22 | 33 | 21 | 34 | 21 | 1 |
| grape__leaf | 77 | 71 | 6 | 63 | 14 | 63 | 0 |
| grape__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 32 | 15 | 23 | 24 | 23 | 0 |
| strawberry__leaf | 45 | 38 | 7 | 37 | 8 | 37 | 0 |
| tomato__fruit | 75 | 64 | 11 | 60 | 15 | 60 | 0 |
| tomato__leaf | 108 | 71 | 37 | 63 | 45 | 63 | 0 |
| tomato__unknown_part | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| unknown_crop | 6 | 5 | 1 | 1 | 5 | 0 | 0 |
