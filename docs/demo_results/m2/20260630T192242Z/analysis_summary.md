# M2 Demo Analysis Summary

- total: 602
- router_crop_correctness: `{"correct": 540, "incorrect": 62, "not_applicable": 0}`
- router_part_correctness: `{"correct": 523, "incorrect": 79, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 418, "incorrect": 51, "not_applicable": 133}`
- classless_supported_probes: `{"answered": 20, "answered_target_correct": 20, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 60, "total": 80}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 109, "accept_router": 200, "use_prototype": 293}`
- answered_wrong_by_target: `{"apricot__fruit": 9, "apricot__leaf": 1, "grape__fruit": 8, "grape__leaf": 9, "strawberry__fruit": 12, "strawberry__leaf": 1, "tomato__fruit": 4, "tomato__leaf": 7}`
- prototype_correct_but_abstained: 43 `{"apricot__fruit": 5, "apricot__leaf": 2, "grape__fruit": 3, "grape__leaf": 3, "strawberry__fruit": 8, "strawberry__leaf": 6, "tomato__fruit": 10, "tomato__leaf": 6}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 64 | 55 | 9 | 52 | 12 | 42 | 0 |
| apricot__leaf | 47 | 39 | 8 | 46 | 1 | 36 | 0 |
| grape__fruit | 65 | 56 | 9 | 55 | 10 | 45 | 0 |
| grape__leaf | 87 | 80 | 7 | 76 | 11 | 66 | 0 |
| grape__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 57 | 46 | 11 | 43 | 14 | 33 | 0 |
| strawberry__leaf | 55 | 46 | 9 | 53 | 2 | 43 | 0 |
| tomato__fruit | 85 | 68 | 17 | 73 | 12 | 63 | 0 |
| tomato__leaf | 118 | 99 | 19 | 100 | 18 | 90 | 0 |
| tomato__unknown_part | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| unknown_crop | 14 | 0 | 14 | 14 | 0 | 0 | 0 |
