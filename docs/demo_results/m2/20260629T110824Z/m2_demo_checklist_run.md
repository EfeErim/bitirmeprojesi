# M2 Demo Checklist Run

- started_at: `2026-06-29T10:22:51.619294+00:00`
- finished_at: `2026-06-29T11:08:23.355810+00:00`
- elapsed: `45m 32s` (2731.737s)
- generated_at: `2026-06-29T11:08:23.355810+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`
- batch_size: `10`
- adapter_batch_size: `24`
- handoff_cache: `{"enabled": true, "path": "/content/bitirmeprojesi/.runtime_tmp/m2_router_prototype_handoff_cache.json", "refresh": true, "stats": {"hits": 0, "misses": 592, "writes": 592}}`

## Summary

- total: 93
- passed: 78
- failed: 15
- answered: 80
- abstained_or_reviewed: 13
- asset_ready: 0
- failure_buckets: `{"dependency_access": 1, "router": 6}`

## Rows

| image_id | status | pass_fail | failure_bucket | predicted | message |
|---|---|---|---|---|---|
| demo_001 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_002 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_003 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_004 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_005 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_006 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_007 | unknown_crop | fail | router | potato / unknown /  | Part abstained for crop=potato: confidence (0.3830) < threshold (0.4000) |
| demo_008 | success | fail |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_009 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_010 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_011 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_012 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_013 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_014 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_015 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_016 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_017 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_018 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_019 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_020 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_021 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_022 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_023 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_024 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_025 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_026 | success | fail |  | apricot / fruit / kayısı_sağlıklı_meyve_800 |  |
| demo_027 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_028 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_029 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_030 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_031 | success | fail |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_032 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_033 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_034 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_035 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_036 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_037 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_038 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_039 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_040 | success | fail |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_041 | router_uncertain | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_042 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_043 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_044 | router_uncertain | pass |  | grape / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_045 | router_uncertain | pass |  | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_046 | router_uncertain | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_049 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_050 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_051 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_052 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_053 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_054 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_055 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_056 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_057 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_058 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_059 | success | fail |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_060 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_061 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_062 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_063 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_064 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_065 | unknown_crop | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_066 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_067 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_068 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_069 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_070 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_071 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_072 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_073 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_074 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_075 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_076 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_077 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_078 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_079 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_080 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_081 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_082 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_083 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_084 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_085 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_086 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_087 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_088 | success | fail |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_089 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_090 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_091 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_092 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_093 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_094 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_095 | router_unavailable | fail | dependency_access |  /  /  | CUDA out of memory. Tried to allocate 1.03 GiB. GPU 0 has a total capacity of 39.49 GiB of which 348.62 MiB is free. Process 530 has 7.44 GiB memory in use. Including non-PyTorch memory, this process has 31.70 GiB memory |
