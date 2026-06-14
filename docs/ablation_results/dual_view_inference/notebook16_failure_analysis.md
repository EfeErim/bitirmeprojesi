# Notebook 16 Failure Analysis

Source report: `docs/ablation_results/dual_view_inference/multi_target_report.json`
Calibration artifact: `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json`

## Focus Target: `tomato__leaf`

- samples: `1204`
- wrong predictions: `121`
- accuracy: `0.8995`
- review capture on wrong predictions: `0.3058`
- missed wrong predictions: `84`
- false-positive review rate: `0.0416`
- calibration status: `target_specific`

Top confusion pairs:

- `domates_late_blight_yaprak -> domates_early_blight_yaprak`: `18`
- `domates_early_blight_yaprak -> domates_late_blight_yaprak`: `15`
- `domates_bacterial_spot_and_speck_yaprak -> domates_septoria_leaf_spot_yaprak`: `11`
- `domates_bacterial_spot_and_speck_yaprak -> domates_powdery_mildew_yaprak`: `6`
- `domates_septoria_leaf_spot_yaprak -> domates_early_blight_yaprak`: `6`
- `domates_bacterial_spot_and_speck_yaprak -> domates_early_blight_yaprak`: `5`
- `domates_late_blight_yaprak -> domates_powdery_mildew_yaprak`: `5`
- `domates_mosaic_virüs_yaprak -> domates_yellow_leaf_curl_yaprak`: `4`
- `domates_early_blight_yaprak -> domates_septoria_leaf_spot_yaprak`: `3`
- `domates_late_blight_yaprak -> domates_leaf_mold_yaprak`: `3`

ROI evidence status:

- `supports_full`: `697`
- `roi_too_large`: `231`
- `conflicts_with_full`: `141`
- `roi_too_small`: `135`

ROI quality status:

- `roi_ok`: `838`
- `roi_too_large`: `231`
- `roi_too_small`: `135`

### Missed-Wrong Drilldown

Missed wrong confidence distribution:

- `0.70-0.90`: `34`
- `0.90-0.95`: `15`
- `0.95-0.98`: `12`
- `0.98-0.99`: `6`
- `>=0.99`: `17`

Missed wrong ROI evidence status:

- `supports_full`: `31`
- `roi_too_large`: `30`
- `conflicts_with_full`: `13`
- `roi_too_small`: `10`

Missed wrong ROI quality status:

- `roi_ok`: `44`
- `roi_too_large`: `30`
- `roi_too_small`: `10`

Confidence threshold sweep over existing review decisions:

| Threshold | Review capture | Missed wrong | False-positive review | Review rate | Added reviews | Newly captured wrong |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.9500` | `0.7107` | `35` | `0.1163` | `0.1761` | `130` | `49` |
| `0.9800` | `0.8099` | `23` | `0.1681` | `0.2326` | `198` | `61` |
| `0.9900` | `0.8595` | `17` | `0.2170` | `0.2816` | `257` | `67` |

Top missed confusion examples:

- `domates_early_blight_yaprak -> domates_late_blight_yaprak`: `12` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_early_blight_yaprak/tomato_1070.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_early_blight_yaprak/IMG_2646.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_early_blight_yaprak/tomato_201.jpg`
- `domates_late_blight_yaprak -> domates_early_blight_yaprak`: `12` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/tomato_377.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/tomato_713.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/tomato_354.jpg`
- `domates_bacterial_spot_and_speck_yaprak -> domates_septoria_leaf_spot_yaprak`: `11` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_IMG_3175_jpg.rf.3f51138a2a1172fea6c5982d22fad29c.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/tomato_1147.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_BCH-20medium-20thumbnail-20for-20upload-20-2814-29_jpg.rf.eac6e0de03884e9dd6c309e3b5eb81b2.jpg`
- `domates_late_blight_yaprak -> domates_powdery_mildew_yaprak`: `5` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/Kaggle_Lb43.JPG`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/Kaggle_Lb29.JPG`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/Kaggle_LB_ (278).JPG`
- `domates_bacterial_spot_and_speck_yaprak -> domates_powdery_mildew_yaprak`: `4` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_Bs62.JPG`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_Bs30.JPG`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_Bs5.JPG`

Review-gate focus decision:

- `tomato__leaf` stays report-only; it is not a runtime promotion.
- missed wrong predictions: `84`
- `0.95` confidence-threshold simulation: review capture `0.7107`, false-positive review `0.1163`.

## All Targets

| Target | Samples | Wrong | Accuracy | Review capture | Missed wrong | False-positive review | Calibration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tomato__leaf` | `1204` | `121` | `0.8995` | `0.3058` | `84` | `0.0416` | `target_specific` |
| `strawberry__fruit` | `204` | `112` | `0.4510` | `0.2500` | `84` | `0.3152` | `no_eligible_policy` |
| `grape__leaf` | `324` | `39` | `0.8796` | `0.5897` | `16` | `0.0632` | `target_specific` |
| `apricot__leaf` | `150` | `28` | `0.8133` | `0.5357` | `13` | `0.1066` | `group_fallback` |
| `apricot__fruit` | `280` | `14` | `0.9500` | `0.1429` | `12` | `0.0038` | `group_fallback` |
| `grape__fruit` | `174` | `16` | `0.9080` | `0.6250` | `6` | `0.0823` | `no_eligible_policy` |
| `tomato__fruit` | `119` | `12` | `0.8992` | `0.5833` | `5` | `0.1121` | `group_fallback` |
| `strawberry__leaf` | `491` | `1` | `0.9980` | `0.0000` | `1` | `0.0000` | `group_fallback` |

## Data/Label Audit Target: `strawberry__fruit`

- samples: `204`
- wrong predictions: `112`
- accuracy: `0.4510`
- review capture on wrong predictions: `0.2500`
- missed wrong predictions: `84`
- false-positive review rate: `0.3152`
- calibration status: `no_eligible_policy`

Top confusion pairs:

- `strawberry_healthy_fruit -> strawberry_unripe_fruit`: `79`
- `strawberry_gray_mold_fruit -> strawberry_powdery_mildew_fruit`: `13`
- `strawberry_gray_mold_fruit -> strawberry_anthracnose_fruit`: `9`
- `strawberry_powdery_mildew_fruit -> strawberry_anthracnose_fruit`: `6`
- `strawberry_powdery_mildew_fruit -> strawberry_gray_mold_fruit`: `5`

ROI evidence status:

- `supports_full`: `108`
- `roi_too_large`: `70`
- `conflicts_with_full`: `23`
- `roi_too_small`: `3`

ROI quality status:

- `roi_ok`: `131`
- `roi_too_large`: `70`
- `roi_too_small`: `3`

### Missed-Wrong Drilldown

Missed wrong confidence distribution:

- `0.70-0.90`: `7`
- `0.90-0.95`: `6`
- `0.95-0.98`: `8`
- `0.98-0.99`: `23`
- `>=0.99`: `40`

Missed wrong ROI evidence status:

- `roi_too_large`: `44`
- `supports_full`: `39`
- `conflicts_with_full`: `1`

Missed wrong ROI quality status:

- `roi_too_large`: `44`
- `roi_ok`: `40`

Confidence threshold sweep over existing review decisions:

| Threshold | Review capture | Missed wrong | False-positive review | Review rate | Added reviews | Newly captured wrong |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.9500` | `0.3661` | `71` | `0.8152` | `0.5686` | `59` | `13` |
| `0.9800` | `0.4375` | `63` | `0.9022` | `0.6471` | `75` | `21` |
| `0.9900` | `0.6429` | `40` | `0.9674` | `0.7892` | `104` | `44` |

Top missed confusion examples:

- `strawberry_healthy_fruit -> strawberry_unripe_fruit`: `78` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_healthy_fruit/fresa_419.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_healthy_fruit/fresa_479.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_healthy_fruit/fresa_472.jpg`
- `strawberry_gray_mold_fruit -> strawberry_anthracnose_fruit`: `3` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_gray_mold_fruit/gray_mold_81.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_gray_mold_fruit/gray_mold_308.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_gray_mold_fruit/gray_mold_94.jpg`
- `strawberry_gray_mold_fruit -> strawberry_powdery_mildew_fruit`: `2` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_gray_mold_fruit/gray_mold_298.jpg`, `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_gray_mold_fruit/gray_mold_292.jpg`
- `strawberry_powdery_mildew_fruit -> strawberry_anthracnose_fruit`: `1` examples; `/content/bitirmeprojesi/data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_powdery_mildew_fruit/powdery_mildew_fruit_112.jpg`

## Decision

- Keep this as analysis/reporting only.
- Treat `tomato__leaf` as the review-gate focus target for this Notebook 16 pass.
- Do not change Notebook 16 final-decision behavior from this artifact alone.
- Do not promote v2 calibration policies into runtime without a separate validation decision.
