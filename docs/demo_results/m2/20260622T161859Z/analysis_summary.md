# M2 Demo Analysis Summary

- total: 512
- router_crop_correctness: `{"correct": 454, "incorrect": 58, "not_applicable": 0}`
- router_part_correctness: `{"correct": 420, "incorrect": 92, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 349, "incorrect": 53, "not_applicable": 110}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 110, "accept_router": 174, "use_prototype": 228}`
- answered_wrong_by_target: `{"apricot__fruit": 12, "apricot__leaf": 1, "grape__fruit": 1, "grape__leaf": 9, "non_plant": 2, "strawberry__fruit": 10, "strawberry__leaf": 1, "tomato__fruit": 4, "tomato__leaf": 9, "unknown_crop": 4}`
- prototype_correct_but_abstained: 86
- negative_false_accepts: 6
- opposite_part_disease_labels: 15

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 47 | 7 | 35 | 19 | 35 | 13 |
| apricot__leaf | 37 | 37 | 0 | 36 | 1 | 36 | 0 |
| grape__fruit | 55 | 25 | 30 | 24 | 31 | 24 | 1 |
| grape__leaf | 77 | 73 | 4 | 64 | 13 | 64 | 0 |
| grape__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| non_plant | 6 | 2 | 4 | 4 | 2 | 0 | 0 |
| strawberry__fruit | 47 | 35 | 12 | 25 | 22 | 25 | 0 |
| strawberry__leaf | 45 | 44 | 1 | 43 | 2 | 43 | 0 |
| tomato__fruit | 75 | 65 | 10 | 61 | 14 | 61 | 0 |
| tomato__leaf | 108 | 70 | 38 | 61 | 47 | 61 | 1 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 6 | 4 | 2 | 2 | 4 | 0 | 0 |
