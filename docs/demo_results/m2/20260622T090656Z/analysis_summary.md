# M2 Demo Analysis Summary

- total: 64
- router_crop_correctness: `{"correct": 52, "incorrect": 12, "not_applicable": 0}`
- router_part_correctness: `{"correct": 50, "incorrect": 14, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 38, "incorrect": 8, "not_applicable": 18}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 18, "accept_router": 18, "use_prototype": 28}`
- answered_wrong_by_target: `{"apricot__fruit": 1, "apricot__leaf": 1, "tomato__leaf": 5, "unknown_crop": 1}`
- prototype_correct_but_abstained: 11
- negative_false_accepts: 1
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 4 | 3 | 1 | 2 | 2 | 2 | 0 |
| apricot__leaf | 5 | 3 | 2 | 2 | 3 | 2 | 0 |
| grape__fruit | 5 | 2 | 3 | 2 | 3 | 2 | 0 |
| grape__leaf | 5 | 5 | 0 | 5 | 0 | 5 | 0 |
| grape__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| non_plant | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| strawberry__fruit | 5 | 4 | 1 | 4 | 1 | 4 | 0 |
| strawberry__leaf | 5 | 5 | 0 | 5 | 0 | 5 | 0 |
| tomato__fruit | 15 | 13 | 2 | 13 | 2 | 13 | 0 |
| tomato__leaf | 14 | 10 | 4 | 5 | 9 | 5 | 0 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 2 | 1 | 1 | 1 | 1 | 0 | 0 |
