# M2 Demo Analysis Summary

- total: 512
- router_crop_correctness: `{"correct": 387, "incorrect": 125, "not_applicable": 0}`
- router_part_correctness: `{"correct": 358, "incorrect": 154, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 268, "incorrect": 33, "not_applicable": 211}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- prototype_reconciliation: `{"abstain": 211, "accept_router": 150, "use_prototype": 151}`
- opposite_part_disease_labels: 7

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 23 | 31 | 16 | 38 | 16 | 7 |
| apricot__leaf | 37 | 33 | 4 | 32 | 5 | 32 | 0 |
| grape__fruit | 55 | 16 | 39 | 16 | 39 | 16 | 0 |
| grape__leaf | 77 | 64 | 13 | 58 | 19 | 58 | 0 |
| grape__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 27 | 20 | 21 | 26 | 21 | 0 |
| strawberry__leaf | 45 | 34 | 11 | 33 | 12 | 33 | 0 |
| tomato__fruit | 75 | 41 | 34 | 37 | 38 | 37 | 0 |
| tomato__leaf | 108 | 61 | 47 | 55 | 53 | 55 | 0 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 6 | 2 | 4 | 4 | 2 | 0 | 0 |
