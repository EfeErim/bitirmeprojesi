# OOD/OE Quality Audit

- dataset: `grape__leaf`
- records: 2842
- issues: 33
- roles: `{"continual": 1200, "oe": 564, "ood": 430, "test": 324, "val": 324}`
- issue types: `{"near_duplicate_perceptual_hash": 8, "semantic_slice_suspicion": 25}`

## Top Review Items

- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__161.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/157.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__161.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/165.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__177.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/173.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__177.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/181.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__193.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/197.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__257_1_.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/261.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__273_1_.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/277.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/off_crop_or_root_disease/off_crop_or_root_disease__65.jpg` vs `oe/unsupported_leaf_disease_candidates/mites_leaf/z_m_Outlier-20260503T132613Z-3-001/61.jpg` - dHash Hamming distance <= 6
- `review` `semantic_slice_suspicion` `oe/same_crop_fruit_negatives/powdery_mildew_fruit/internet_20260506_wikimedia_uncinula_necator_on_grapes.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/grape_specific_unknowns/internet_20260509_plantdoc_train_grape_leaf_black_rot_072109_20Hartman_20Grape_20black_20rot-fruit_20_20lvs.JPG.jpg.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_or_root_disease/off_crop_or_root_disease__grape_black_rot_google_0204.jpg` - off-crop slice has same-crop path tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_or_root_disease/off_crop_or_root_disease__grape_black_rot_google_0209.jpg` - off-crop slice has same-crop path tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_or_root_disease/off_crop_or_root_disease__grape_black_rot_google_0223.jpg` - off-crop slice has same-crop path tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_or_root_disease/off_crop_or_root_disease__grape_black_rot_google_0237.jpg` - off-crop slice has same-crop path tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_or_root_disease/off_crop_or_root_disease__grape_black_rot_google_0260.jpg` - off-crop slice has same-crop path tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/internet_20260506_bugwood_1572815_strawberry_anthracnose_fruit.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/internet_20260506_bugwood_1572819_strawberry_anthracnose_fruit.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/internet_20260506_bugwood_1572820_strawberry_anthracnose_fruit.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/internet_20260506_wikimedia_strawberry_fruit_rot.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/internet_20260506_wikimedia_tomato_late_blight_fruit_rot.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/off_crop_secondary__anthracnose_fruit_rot_58.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/off_crop_secondary__anthracnose_fruit_rot_81.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/off_crop_secondary__powdery_mildew_fruit_59.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/off_crop_secondary__powdery_mildew_fruit_60.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/off_crop_secondary__powdery_mildew_fruit_62.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_anthracnose_fruit_rot_48_jpg_adlı_dosyanın_kopyası_6cead78b7677.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_failure_unsupported_5ecf710898f3.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_failure_unsupported_d8a63afdc248.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_unknown_unsupported_4c34f1efb974.png` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_unknown_unsupported_87ada48efa45.png` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_unknown_unsupported_eb92846a1a2c.png` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_unknown_unsupported_f6e052191997.png` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_unknown_unsupported_f900958933de.png` - leaf adapter pool item has fruit-like path or slice tokens
