# OOD/OE Quality Audit

- dataset: `strawberry__leaf`
- records: 2958
- issues: 17
- roles: `{"continual": 1606, "oe": 143, "ood": 227, "test": 491, "val": 491}`
- issue types: `{"near_duplicate_perceptual_hash": 5, "semantic_slice_suspicion": 12}`

## Top Review Items

- `review` `near_duplicate_perceptual_hash` `ood/off_crop_secondary/off_crop_secondary__blossom_blight_49.jpg` vs `oe/same_crop_blossom_negatives/blossom_blight/ilek_Outlier-20260503T132606Z-3-001/blossom_blight_109.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_secondary/off_crop_secondary__blossom_blight_49.jpg` vs `oe/same_crop_blossom_negatives/blossom_blight/ilek_Outlier-20260503T132606Z-3-001/blossom_blight_121.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_secondary/off_crop_secondary__blossom_blight_49.jpg` vs `oe/same_crop_blossom_negatives/blossom_blight/ilek_Outlier-20260503T132606Z-3-001/blossom_blight_134.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_secondary/off_crop_secondary__blossom_blight_49.jpg` vs `oe/same_crop_blossom_negatives/blossom_blight/ilek_Outlier-20260503T132606Z-3-001/blossom_blight_139.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_secondary/off_crop_secondary__blossom_blight_49.jpg` vs `oe/same_crop_blossom_negatives/blossom_blight/ilek_Outlier-20260503T132606Z-3-001/blossom_blight_151.jpg` - dHash Hamming distance <= 6
- `review` `semantic_slice_suspicion` `oe/same_crop_fruit_negatives/anthracnose_fruit/internet_20260506_bugwood_1572815_strawberry_anthracnose_fruit.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `oe/same_crop_fruit_negatives/anthracnose_fruit/internet_20260506_bugwood_1572819_strawberry_anthracnose_fruit.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `oe/same_crop_fruit_negatives/anthracnose_fruit/internet_20260506_bugwood_1572820_strawberry_anthracnose_fruit.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `oe/same_crop_fruit_negatives/fruit_rot/internet_20260506_wikimedia_strawberry_fruit_rot.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `oe/same_crop_fruit_negatives/powdery_mildew_fruit/internet_20260506_bugwood_5631933_strawberry_powdery_mildew.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `oe/same_crop_fruit_negatives/powdery_mildew_fruit/internet_20260506_wikimedia_oidium_du_fraisier.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/non_plant_misc__Bakteriyel_Kanser_ve_Zamklanma_1.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/non_plant_misc__Ekran_görüntüsü_2026-03-14_195623.png` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/non_plant_misc__Ekran_görüntüsü_2026-03-15_143215.png` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/non_plant_misc__depositphotos_390716206-stock-photo-gardening-concept-male-gardener-spraying.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/non_plant_misc__image-2-1.png` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/other_crop_disease/internet_20260506_wikimedia_tomato_late_blight_fruit_rot.jpg` - leaf adapter pool item has fruit-like path or slice tokens
