# OOD/OE Quality Audit

- dataset: `apricot__leaf`
- records: 1125
- issues: 25
- roles: `{"continual": 508, "oe": 120, "ood": 197, "test": 150, "val": 150}`
- issue types: `{"near_duplicate_perceptual_hash": 8, "semantic_slice_suspicion": 17}`

## Top Review Items

- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_123_811cafee8208.jpeg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_123_9b7ad7e82bca.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_139_21f7e8033689.jpeg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_67_269b13bb2f50.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_25_abc2147ec6be.jpg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_21_67add931afff.jpeg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_30_825e16f1fd52.jpeg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_22_a62a761dfc67.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_39_7401b27a2d2f.jpeg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_40_7dcc0492d395.jpeg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_77_a605e1e909a3.jpeg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_117_54161c941b21.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_89_cf1d3d028ed0.jpeg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_109_201bc94a40d4.jpg` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `ood/same_crop_unsupported_unknowns/takviye_20260514_93_c2c718442dea.jpg` vs `oe/unsupported_leaf_disease_candidates/takviye_20260514_105_5802ebdc7732.jpeg` - dHash Hamming distance <= 6
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/New folder (2)_kayisi_11.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (20).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (21).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (22).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (23).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (24).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (26).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (27).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (28).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_anthracnose_fruit_rot_22_jpg_adlı_dosyanın_kopyası_57a3d674cb45.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_failure_unsupported_38d458287e3f.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_failure_unsupported_5ecf710898f3.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_failure_unsupported_6df6d4c0a743.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_failure_unsupported_924abebd5aa1.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_failure_unsupported_f9ae1c2a6c33.jpg` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_unknown_unsupported_21253066e915.png` - leaf adapter pool item has fruit-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_to_ood_fruit_unknown_unsupported_eb92846a1a2c.png` - leaf adapter pool item has fruit-like path or slice tokens
