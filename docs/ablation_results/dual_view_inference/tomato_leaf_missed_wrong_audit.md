# `tomato__leaf` Missed-Wrong Audit

Source report: `docs/ablation_results/dual_view_inference/multi_target_report.json`

## Summary

- wrong predictions: `121`
- missed wrong predictions: `84`
- local files available: `84`

## Top Missed Confusions

| Confusion pair | Count |
| --- | ---: |
| `domates_early_blight_yaprak -> domates_late_blight_yaprak` | `12` |
| `domates_late_blight_yaprak -> domates_early_blight_yaprak` | `12` |
| `domates_bacterial_spot_and_speck_yaprak -> domates_septoria_leaf_spot_yaprak` | `11` |
| `domates_late_blight_yaprak -> domates_powdery_mildew_yaprak` | `5` |
| `domates_bacterial_spot_and_speck_yaprak -> domates_powdery_mildew_yaprak` | `4` |
| `domates_septoria_leaf_spot_yaprak -> domates_early_blight_yaprak` | `4` |
| `domates_early_blight_yaprak -> domates_septoria_leaf_spot_yaprak` | `3` |
| `domates_mosaic_virüs_yaprak -> domates_powdery_mildew_yaprak` | `3` |
| `domates_mosaic_virüs_yaprak -> domates_yellow_leaf_curl_yaprak` | `3` |
| `domates_septoria_leaf_spot_yaprak -> domates_powdery_mildew_yaprak` | `3` |

## Review Guidance

- Audit these rows as data/label quality first; do not promote a runtime policy from this table alone.
- Prioritize high-count confusion pairs before one-off mistakes.
- Keep full-image adapter prediction as final until a refreshed Notebook 16 artifact passes promotion gates.

## First Rows

| Rank | Expected | Predicted | Confidence | Local path |
| ---: | --- | --- | ---: | --- |
| `1` | `domates_bacterial_spot_and_speck_yaprak` | `domates_powdery_mildew_yaprak` | `0.9999` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_Bs62.JPG` |
| `2` | `domates_late_blight_yaprak` | `domates_powdery_mildew_yaprak` | `0.9999` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/Kaggle_Lb43.JPG` |
| `3` | `domates_bacterial_spot_and_speck_yaprak` | `domates_septoria_leaf_spot_yaprak` | `0.9997` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_IMG_3175_jpg.rf.3f51138a2a1172fea6c5982d22fad29c.jpg` |
| `4` | `domates_septoria_leaf_spot_yaprak` | `domates_powdery_mildew_yaprak` | `0.9996` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_septoria_leaf_spot_yaprak/Kaggle_Gls43.jpg` |
| `5` | `domates_late_blight_yaprak` | `domates_powdery_mildew_yaprak` | `0.9992` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/Kaggle_Lb29.JPG` |
| `6` | `domates_bacterial_spot_and_speck_yaprak` | `domates_powdery_mildew_yaprak` | `0.9989` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_Bs30.JPG` |
| `7` | `domates_yellow_leaf_curl_yaprak` | `domates_spotted_wilt_yaprak` | `0.9987` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_yellow_leaf_curl_yaprak/tomato_yellow_leaf_curl_virus_27.jpg` |
| `8` | `domates_mosaic_virüs_yaprak` | `domates_powdery_mildew_yaprak` | `0.9987` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_mosaic_virüs_yaprak/Kaggle_MV_ (784).jpg` |
| `9` | `domates_yellow_leaf_curl_yaprak` | `domates_spotted_wilt_yaprak` | `0.9986` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_yellow_leaf_curl_yaprak/tomato_yellow_leaf_curl_virus_google_0075.jpg` |
| `10` | `domates_early_blight_yaprak` | `domates_late_blight_yaprak` | `0.9984` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_early_blight_yaprak/tomato_1070.jpg` |
| `11` | `domates_septoria_leaf_spot_yaprak` | `domates_powdery_mildew_yaprak` | `0.9981` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_septoria_leaf_spot_yaprak/Kaggle_Gls56.JPG` |
| `12` | `domates_late_blight_yaprak` | `domates_powdery_mildew_yaprak` | `0.9975` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak/Kaggle_LB_ (278).JPG` |
| `13` | `domates_septoria_leaf_spot_yaprak` | `domates_powdery_mildew_yaprak` | `0.9967` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_septoria_leaf_spot_yaprak/Kaggle_Gls69.JPG` |
| `14` | `domates_mosaic_virüs_yaprak` | `domates_powdery_mildew_yaprak` | `0.9958` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_mosaic_virüs_yaprak/Kaggle_tobacco-mosaic-virus-tomato-1580133887.jpg` |
| `15` | `domates_yellow_leaf_curl_yaprak` | `domates_powdery_mildew_yaprak` | `0.9937` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_yellow_leaf_curl_yaprak/tomato_yellow_leaf_curl_virus_149.jpg` |
| `16` | `domates_early_blight_yaprak` | `domates_late_blight_yaprak` | `0.9906` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_early_blight_yaprak/IMG_2646.jpg` |
| `17` | `domates_septoria_leaf_spot_yaprak` | `domates_early_blight_yaprak` | `0.9902` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_septoria_leaf_spot_yaprak/Kaggle_septoria-leafspot-on-tomato-leaf-rp-c_jpg.rf.2fa21dba8df784c6f3beb6d633adf226.jpg` |
| `18` | `domates_bacterial_spot_and_speck_yaprak` | `domates_powdery_mildew_yaprak` | `0.9893` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_Bs5.JPG` |
| `19` | `domates_bacterial_spot_and_speck_yaprak` | `domates_septoria_leaf_spot_yaprak` | `0.9885` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/tomato_1147.jpg` |
| `20` | `domates_bacterial_spot_and_speck_yaprak` | `domates_septoria_leaf_spot_yaprak` | `0.9872` | `data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak/Kaggle_BCH-20medium-20thumbnail-20for-20upload-20-2814-29_jpg.rf.eac6e0de03884e9dd6c309e3b5eb81b2.jpg` |
