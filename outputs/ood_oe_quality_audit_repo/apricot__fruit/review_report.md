# OOD/OE Quality Audit

- dataset: `apricot__fruit`
- records: 1727
- issues: 37
- roles: `{"continual": 871, "oe": 120, "ood": 176, "test": 280, "val": 280}`
- issue types: `{"near_duplicate_perceptual_hash": 3, "semantic_slice_suspicion": 34}`

## Top Review Items

- `review` `near_duplicate_perceptual_hash` `continual/kayısıda_yaprak_delen_çil_hastalığı_meyve_128/Kayısıda yaprak delen çil 103.JPG` vs `ood/same_crop_unsupported_unknowns/Ekran görüntüsü 2026-05-04 125712.png` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `val/kayısıda_şeftali_karalekesi_meyve_232/Şeftali Karalekesi 9.JPG` vs `ood/same_crop_unsupported_unknowns/Ekran görüntüsü 2026-05-04 125548.png` - dHash Hamming distance <= 6
- `review` `near_duplicate_perceptual_hash` `val/kayısıda_şeftali_karalekesi_meyve_232/peach_scab_Bing_0000.jpg` vs `ood/same_crop_unsupported_unknowns/Ekran görüntüsü 2026-05-04 125548.png` - dHash Hamming distance <= 6
- `review` `semantic_slice_suspicion` `oe/near_id_fruit_disease_and_mixed_background/takviye_20260514_grape_leaf_spot_26_4b689a5d2644.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `oe/near_id_fruit_disease_and_mixed_background/takviye_20260514_grape_leaf_spot_google_0002_b8b9437330eb.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `oe/near_id_fruit_disease_and_mixed_background/takviye_20260514_grape_leaf_spot_google_0016_bfa6ebd48c83.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/3.03.2021-kayisi-hastaliklara-yenik-dusmesin.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/650x344-kayisinin-faydalari-nelerdir-ne-ise-yarar-kayisi-cilde-faydali-mi-1590842163679.webp` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/akdeniz-perfect-red-kayisi-fidani.png` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/hakmar-yaz-meyvesi-kayisi-hakkinda-neler-biliyorsunuz-960x720-1.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (11).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (15).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (17).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (18).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/images (19).jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_non_plant_misc__depositphotos_390716206_stock_photo_gard_0bd9e0f6a67e.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__Frutteto_CVT_100_S_Stage_V_Fie_b0075d52f5fb.webp` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__ZHuki_dolgonosiki_povrezhdayut_bd49acbc60c9.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__alakasız3_cf9b3da61b5f.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__alakasız5_e3999bb0157b.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__alakasız_b3c7c75e8f05.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__alakaısz_05570bc5a41a.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__köpekkk_df438f33198c.jpeg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_scene_context_leak_check__pngtree_golden_suv_parked_amid_8462c5b02bed.webp` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/non_plant_misc/takviye_20260514_web_scene_context_leak_check_9c15b2084c18_9c15b2084c18.jpg` - non-plant slice has plant/crop-like path tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_Strawberry___Leaf_scorch_44_jpg_adlı_dosyanın_kopyası_7a5bc52d3c4e.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_internet_20260509_plantdoc_test_grape_leaf_black_rot_03g_e8468522aba5.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_internet_20260509_plantdoc_train_grape_leaf_black_rot_00_a3946ea0fc75.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_internet_20260509_plantdoc_train_grape_leaf_black_rot_Gu_a7d909e76254.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_internet_20260509_plantdoc_train_grape_leaf_black_rot_bl_a09bc77f4348.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_leaf_unknown_unsupported_same_cro_328a2f36ea3d.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_leaf_unknown_unsupported_same_cro_41fb8828476b.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_leaf_unknown_unsupported_same_cro_73cc35b555be.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_leaf_unknown_unsupported_same_cro_b9717c9ad2ad.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_oe_leaf_unknown_unsupported_same_cro_c19f2ae8b86e.jpg` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_zip_leaf_unknown_image_0001_domates_9f709021e0bb.png` - fruit adapter pool item has leaf-like path or slice tokens
- `review` `semantic_slice_suspicion` `ood/off_crop_secondary/takviye_20260514_rebalanced_20260510_zip_leaf_unknown_image_0176_domates_208eb7d23ecb.png` - fruit adapter pool item has leaf-like path or slice tokens
