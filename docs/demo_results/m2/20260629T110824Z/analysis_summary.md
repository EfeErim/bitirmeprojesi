# M2 Demo Analysis Summary

- total: 93
- router_crop_correctness: `{"correct": 83, "incorrect": 10, "not_applicable": 0}`
- router_part_correctness: `{"correct": 84, "incorrect": 9, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 72, "incorrect": 8, "not_applicable": 13}`
- classless_supported_probes: `{"answered": 0, "answered_target_correct": 0, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 0, "total": 0}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 12, "accept_router": 35, "use_prototype": 45}`
- answered_wrong_by_target: `{"apricot__fruit": 1, "apricot__leaf": 1, "strawberry__leaf": 1, "tomato__leaf": 5}`
- prototype_correct_but_abstained: 4 `{"tomato__fruit": 1, "tomato__leaf": 3}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 4 | 4 | 0 | 3 | 1 | 3 | 0 |
| apricot__leaf | 5 | 5 | 0 | 4 | 1 | 4 | 0 |
| grape__fruit | 12 | 11 | 1 | 11 | 1 | 11 | 0 |
| grape__leaf | 5 | 5 | 0 | 5 | 0 | 5 | 0 |
| grape__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| non_plant | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| strawberry__fruit | 15 | 15 | 0 | 15 | 0 | 15 | 0 |
| strawberry__leaf | 15 | 15 | 0 | 14 | 1 | 14 | 0 |
| tomato__fruit | 15 | 13 | 2 | 13 | 2 | 13 | 0 |
| tomato__leaf | 16 | 12 | 4 | 7 | 9 | 7 | 0 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
