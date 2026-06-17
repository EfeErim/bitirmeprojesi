# M2 Demo Analysis Summary

- total: 512
- router_crop_correctness: `{"correct": 250, "incorrect": 262, "not_applicable": 0}`
- router_part_correctness: `{"correct": 224, "incorrect": 288, "not_applicable": 0}`
- normalized_disease_class_correctness: `{"correct": 179, "incorrect": 56, "not_applicable": 277}`
- adapter_unavailable: `{"missing_adapter": 0, "unknown": 0, "wrong_router": 0}`
- opposite_part_disease_labels: 39

## Per Target

| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| apricot__fruit | 54 | 0 | 54 | 0 | 54 | 0 | 0 |
| apricot__leaf | 37 | 4 | 33 | 4 | 33 | 4 | 0 |
| grape__fruit | 55 | 36 | 19 | 13 | 42 | 13 | 23 |
| grape__leaf | 77 | 73 | 4 | 65 | 12 | 65 | 0 |
| grape__unknown_part | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| non_plant | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
| strawberry__fruit | 47 | 25 | 22 | 8 | 39 | 8 | 13 |
| strawberry__leaf | 45 | 42 | 3 | 41 | 4 | 41 | 1 |
| tomato__fruit | 75 | 6 | 69 | 5 | 70 | 5 | 1 |
| tomato__leaf | 108 | 48 | 60 | 43 | 65 | 43 | 1 |
| tomato__unknown_part | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| unknown_crop | 6 | 0 | 6 | 6 | 0 | 0 | 0 |
