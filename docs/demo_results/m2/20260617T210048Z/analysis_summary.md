# M2 Demo Analysis Summary

- total: 512
- router_crop_correctness: `{"correct": 252, "incorrect": 260, "not_applicable": 0}`
- router_part_correctness: `{"correct": 238, "incorrect": 274, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 152, "incorrect": 12, "not_applicable": 348}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 348, "accept_router": 150, "use_prototype": 14}`
- opposite_part_disease_labels: 0

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 0 | 54 | 0 | 54 | 0 | 0 |
| apricot__leaf | 37 | 4 | 33 | 4 | 33 | 4 | 0 |
| grape__fruit | 55 | 4 | 51 | 4 | 51 | 4 | 0 |
| grape__leaf | 77 | 64 | 13 | 58 | 19 | 58 | 0 |
| grape__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 15 | 32 | 11 | 36 | 11 | 0 |
| strawberry__leaf | 45 | 33 | 12 | 32 | 13 | 32 | 0 |
| tomato__fruit | 75 | 7 | 68 | 7 | 68 | 7 | 0 |
| tomato__leaf | 108 | 37 | 71 | 36 | 72 | 36 | 0 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
