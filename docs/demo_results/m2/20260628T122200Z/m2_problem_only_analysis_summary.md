# M2 Demo Analysis Summary

- total: 89
- router_crop_correctness: `{"correct": 32, "incorrect": 57, "not_applicable": 0}`
- router_part_correctness: `{"correct": 20, "incorrect": 69, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 1, "incorrect": 1, "not_applicable": 87}`
- classless_supported_probes: `{"answered": 0, "answered_target_correct": 0, "answered_target_incorrect": 0, "failed": 0, "reviewed_or_abstained": 0, "total": 0}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 87, "use_prototype": 2}`
- answered_wrong_by_target: `{"strawberry__fruit": 1}`
- prototype_correct_but_abstained: 72 `{"apricot__fruit": 12, "apricot__leaf": 6, "grape__fruit": 4, "grape__leaf": 1, "strawberry__fruit": 15, "tomato__fruit": 12, "tomato__leaf": 22}`
- negative_false_accepts: 0 `{}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 14 | 0 | 14 | 0 | 14 | 0 | 0 |
| apricot__leaf | 6 | 0 | 6 | 0 | 6 | 0 | 0 |
| grape__fruit | 4 | 0 | 4 | 0 | 4 | 0 | 0 |
| grape__leaf | 2 | 0 | 2 | 0 | 2 | 0 | 0 |
| strawberry__fruit | 16 | 1 | 15 | 0 | 16 | 0 | 0 |
| strawberry__leaf | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| tomato__fruit | 19 | 1 | 18 | 1 | 18 | 1 | 0 |
| tomato__leaf | 27 | 0 | 27 | 0 | 27 | 0 | 0 |
