# M2 Demo Analysis Summary

- total: 89
- router_crop_correctness: `{"correct": 42, "incorrect": 47, "not_applicable": 0}`
- router_part_correctness: `{"correct": 42, "incorrect": 47, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 19, "incorrect": 5, "not_applicable": 65}`
- classless_supported_probes: `{"answered": 0, "answered_target_correct": 0, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 0, "total": 0}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 65, "use_prototype": 24}`
- answered_wrong_by_target: `{"strawberry__fruit": 5}`
- prototype_correct_but_abstained: 50 `{"apricot__fruit": 12, "apricot__leaf": 5, "grape__fruit": 4, "grape__leaf": 1, "strawberry__fruit": 2, "tomato__fruit": 5, "tomato__leaf": 21}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 14 | 0 | 14 | 0 | 14 | 0 | 0 |
| apricot__leaf | 6 | 1 | 5 | 1 | 5 | 1 | 0 |
| grape__fruit | 4 | 0 | 4 | 0 | 4 | 0 | 0 |
| grape__leaf | 2 | 0 | 2 | 0 | 2 | 0 | 0 |
| strawberry__fruit | 16 | 14 | 2 | 9 | 7 | 9 | 0 |
| strawberry__leaf | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| tomato__fruit | 19 | 8 | 11 | 8 | 11 | 8 | 0 |
| tomato__leaf | 27 | 1 | 26 | 1 | 26 | 1 | 0 |
