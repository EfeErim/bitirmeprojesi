# M2 Demo Checklist Run

- generated_at: `2026-06-22T09:06:54.388281+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`

## Summary

- total: 64
- passed: 43
- failed: 21
- answered: 46
- abstained_or_reviewed: 18
- asset_ready: 0
- failure_buckets: `{"router": 13}`

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
| demo_011 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_012 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_013 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_014 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_015 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_016 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_017 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_018 | router_uncertain | fail | router | grape / fruit /  | Router result is not eligible for adapter prediction. |
| demo_019 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_020 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_021 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_022 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_023 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_024 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_025 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_026 | success | fail |  | apricot / fruit / kayısı_sağlıklı_meyve_800 |  |
| demo_027 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_028 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_029 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_030 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_031 | success | fail |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_032 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_033 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_034 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_035 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_036 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_037 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_038 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_039 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_040 | success | fail |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_041 | unknown_crop | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_042 | success | fail |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_043 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_044 | router_uncertain | pass |  | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_045 | unknown_crop | pass |  | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_046 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_049 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_050 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_051 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_052 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_053 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_054 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_055 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_056 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_057 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_058 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_059 | success | fail |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_060 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_061 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_062 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_063 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_064 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_065 | unknown_crop | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_066 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
