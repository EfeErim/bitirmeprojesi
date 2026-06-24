# M2 Demo Checklist Run

- started_at: `2026-06-24T19:25:47.326158+00:00`
- finished_at: `2026-06-24T20:06:49.853216+00:00`
- elapsed: `41m 3s` (2462.527s)
- generated_at: `2026-06-24T20:06:49.853216+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`
- batch_size: `12`
- adapter_batch_size: `32`
- handoff_cache: `{"enabled": true, "path": "/content/bitirmeprojesi/.runtime_tmp/m2_router_prototype_handoff_cache.json", "refresh": false, "stats": {"hits": 0, "misses": 602, "writes": 602}}`

## Summary

- total: 602
- passed: 469
- failed: 133
- answered: 415
- abstained_or_reviewed: 187
- asset_ready: 0
- failure_buckets: `{"router": 93}`

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
| demo_029 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_030 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_031 | success | fail |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_032 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_033 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2868) < threshold (0.4000); margin (-0.1894) < threshold (0.1000) |
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
| demo_056 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_057 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_058 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_059 | success | fail |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_060 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_061 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_062 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_063 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_064 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_065 | unknown_crop | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_066 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_067 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_068 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_069 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_070 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_071 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_072 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_073 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_074 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_075 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_076 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_077 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_078 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
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
| demo_095 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_096 | success | fail |  | grape / fruit / u_zu_m_sag_lıklı_meyve |  |
| demo_097 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_098 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_099 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_100 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_101 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_102 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_103 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_104 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_105 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_106 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_107 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_108 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_109 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_110 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_111 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_112 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_113 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_114 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_115 | success | fail |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_116 | success | pass |  | apricot / fruit / kayısıda_şarka_virüsü_meyve_230 |  |
| demo_117 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_118 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_119 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_120 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_121 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_122 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_123 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_124 | router_uncertain | fail | router | apricot / leaf /  | Router result is not eligible for adapter prediction. |
| demo_125 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_126 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_127 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_128 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_129 | router_uncertain | fail | router | tomato / fruit /  | Router result is not eligible for adapter prediction. |
| demo_130 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_131 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_132 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_133 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_134 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_135 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_136 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_137 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_138 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_139 | router_uncertain | pass |  | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_140 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_141 | router_uncertain | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_142 | router_uncertain | pass |  | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_143 | router_uncertain | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_144 | router_uncertain | pass |  | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_145 | unknown_crop | fail | router | cucumber / fruit /  | Router crop 'cucumber' is outside the final demo supported crop set. |
| demo_146 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_147 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_148 | success | pass |  | apricot / fruit / kayısıda_şarka_virüsü_meyve_230 |  |
| demo_149 | success | fail |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_150 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_151 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_152 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_153 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_154 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_155 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_156 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_157 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_158 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_159 | success | fail |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_160 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_161 | success | fail |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_162 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_163 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_164 | success | fail |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_165 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_166 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_167 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_168 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_169 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_170 | success | fail |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_171 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_172 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_173 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_174 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_175 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_176 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_177 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_178 | success | fail |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_179 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_180 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_181 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_182 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_183 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_184 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_185 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_186 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_187 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_188 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_189 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_190 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_191 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_192 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_193 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_194 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_195 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_196 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_197 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_198 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_199 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_200 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_201 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_202 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_203 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_204 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_205 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_206 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_207 | success | fail |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_208 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_209 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_210 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_211 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_212 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_213 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_214 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_215 | unknown_crop | fail | router | cassava / unknown /  | Part abstained for crop=cassava: no compatible parts configured for crop (cassava) |
| demo_216 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_217 | success | fail |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_218 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_219 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_220 | success | fail |  | grape / fruit / u_zu_m_sag_lıklı_meyve |  |
| demo_221 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_222 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_223 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_224 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_225 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_226 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_227 | router_uncertain | fail | router | grape / whole plant /  | Router part 'whole plant' is outside the final demo supported part set. |
| demo_228 | success | fail |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_229 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_230 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_231 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_232 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_233 | success | fail |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_234 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_235 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_236 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_237 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_238 | success | pass |  | grape / fruit / u_zu_m_sag_lıklı_meyve |  |
| demo_239 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_240 | success | pass |  | grape / fruit / u_zu_m_sag_lıklı_meyve |  |
| demo_241 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_242 | success | pass |  | grape / fruit / u_zu_m_sag_lıklı_meyve |  |
| demo_243 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_244 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_245 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_246 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_247 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_248 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_249 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_250 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_251 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_252 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_253 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_254 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_255 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_256 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_257 | success | fail |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_258 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_259 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_260 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_261 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_262 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_263 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_264 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_265 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_266 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_267 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_268 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_269 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_270 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_271 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_272 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_273 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_274 | success | fail |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_275 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_276 | success | fail |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_277 | success | fail |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_278 | success | fail |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_279 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_280 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_281 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_282 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_283 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_284 | success | fail |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_285 | unknown_crop | fail | router | celery / unknown /  | Part abstained for crop=celery: no compatible parts configured for crop (celery) |
| demo_286 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_287 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_288 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_289 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_290 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_291 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_292 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_293 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_294 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_295 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_296 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_297 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_298 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_299 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_300 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_301 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_302 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_303 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_304 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_305 | unknown_crop | fail | router | orange / unknown /  | Part abstained for crop=orange: no compatible parts configured for crop (orange) |
| demo_306 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_307 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_308 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_309 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_310 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_311 | unknown_crop | fail | router | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_312 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_313 | success | fail |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_314 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_315 | success | fail |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_316 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_317 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_318 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_319 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_320 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_321 | success | fail |  | strawberry / fruit / strawberry_unripe_fruit |  |
| demo_322 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_323 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_324 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_325 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_326 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_327 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_328 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_329 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_330 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_331 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_332 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_333 | success | fail |  | strawberry / fruit / strawberry_unripe_fruit |  |
| demo_334 | success | fail |  | strawberry / fruit / strawberry_unripe_fruit |  |
| demo_335 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_336 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_337 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_338 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_339 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_340 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_341 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_342 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_343 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_344 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_345 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_346 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_347 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_348 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_349 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_350 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_351 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_352 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_353 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_354 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_355 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_356 | router_uncertain | fail | router | strawberry / fruit /  | Router result is not eligible for adapter prediction. |
| demo_357 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_358 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_359 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_360 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_361 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_362 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_363 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_364 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_365 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_366 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_367 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_368 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_369 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_370 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_371 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_372 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_373 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_374 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_375 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_376 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_377 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_378 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_379 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_380 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_381 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_382 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_383 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_384 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_385 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_386 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_387 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_388 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_389 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_390 | success | fail |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_391 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_392 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_393 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_394 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_395 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_396 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_397 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_398 | unknown_crop | fail | router | avocado / unknown /  | Part abstained for crop=avocado: no compatible parts configured for crop (avocado) |
| demo_399 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_400 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_401 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_402 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_403 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_404 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_405 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_406 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_407 | success | fail |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_408 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_409 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_410 | success | fail |  | tomato / fruit / domates_sag_lıklı_meyve |  |
| demo_411 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_412 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_413 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_414 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_415 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_416 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_417 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_418 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_419 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_420 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_421 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_422 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_423 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_424 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_425 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_426 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_427 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_428 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_429 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_430 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2671) < threshold (0.4000); margin (0.0232) < threshold (0.1000) |
| demo_431 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_432 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_433 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_434 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_435 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_436 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_437 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_438 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_439 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_440 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_441 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_442 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_443 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_444 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_445 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_446 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_447 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_448 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_449 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_450 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_451 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_452 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_453 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_454 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_455 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_456 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_457 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_458 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_459 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_460 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_461 | unknown_crop | fail | router | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_462 | unknown_crop | fail | router | quinoa / unknown /  | Part abstained for crop=quinoa: no compatible parts configured for crop (quinoa) |
| demo_463 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_464 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_465 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_466 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_467 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_468 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_469 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_470 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_471 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_472 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_473 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_474 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_475 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_476 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_477 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_478 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_479 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_480 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_481 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_482 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_483 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_484 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_485 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_486 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_487 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_488 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_489 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_490 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_491 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_492 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_493 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_494 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_495 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_496 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_497 | success | pass |  | tomato / leaf / domates_sağlıklı_yaprak |  |
| demo_498 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_499 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_500 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_501 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_502 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_503 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_504 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_505 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_506 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_507 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_508 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_509 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_510 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_511 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_512 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_513 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_514 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_515 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_516 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_517 | router_uncertain | pass |  | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_518 | router_uncertain | pass |  | tomato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_519 | router_uncertain | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_520 | router_uncertain | pass |  | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_521 | router_uncertain | pass |  | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_522 | router_uncertain | pass |  | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_523 | router_uncertain | pass |  | potato / leaf /  | Expected demo row is marked unsupported/unknown; adapter prediction is blocked even when router and prototype agree on a supported target. |
| demo_524 | router_uncertain | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_525 | unknown_crop | pass |  | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_526 | unknown_crop | pass |  | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_527 | unknown_crop | pass |  | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_528 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_529 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_530 | unknown_crop | pass |  | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_531 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_532 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_533 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_534 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_535 | router_uncertain | pass |  | tomato / fruit /  | Classless supported probe target disagrees with the router/prototype handoff; adapter prediction is blocked and the row is treated as review. |
| demo_536 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_537 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_538 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_539 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_540 | router_uncertain | pass |  | tomato / fruit /  | Router result is not eligible for adapter prediction. |
| demo_541 | unknown_crop | pass |  | quinoa / unknown /  | Part abstained for crop=quinoa: no compatible parts configured for crop (quinoa) |
| demo_542 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_543 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_544 | router_uncertain | pass |  | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_545 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_546 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_547 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_548 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_549 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_550 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_551 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_552 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_553 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_554 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_555 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_556 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_557 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_558 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_559 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_560 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_561 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_562 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_563 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_564 | router_uncertain | pass |  | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_565 | success | pass |  | grape / fruit / u_zu_m_sag_lıklı_meyve |  |
| demo_566 | unknown_crop | pass |  | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_567 | router_uncertain | pass |  | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_568 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_569 | router_uncertain | pass |  | grape / leaf /  | Classless supported probe target disagrees with the router/prototype handoff; adapter prediction is blocked and the row is treated as review. |
| demo_570 | unknown_crop | pass |  | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_571 | router_uncertain | pass |  | grape / leaf /  | Classless supported probe target disagrees with the router/prototype handoff; adapter prediction is blocked and the row is treated as review. |
| demo_572 | router_uncertain | pass |  | grape / leaf /  | Classless supported probe target disagrees with the router/prototype handoff; adapter prediction is blocked and the row is treated as review. |
| demo_573 | router_uncertain | pass |  | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_574 | success | pass |  | grape / fruit / u_zu_m_sag_lıklı_meyve |  |
| demo_575 | unknown_crop | pass |  | orange / unknown /  | Part abstained for crop=orange: no compatible parts configured for crop (orange) |
| demo_576 | router_uncertain | pass |  | grape / unknown /  | Part abstained for crop=grape: confidence (0.3806) < threshold (0.4000) |
| demo_577 | router_uncertain | pass |  | grape / fruit /  | Router result is not eligible for adapter prediction. |
| demo_578 | router_uncertain | pass |  | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_579 | router_uncertain | pass |  | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_580 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_581 | unknown_crop | pass |  | lambsquarters / unknown /  | Part abstained for crop=lambsquarters: no compatible parts configured for crop (lambsquarters) |
| demo_582 | router_uncertain | pass |  | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_583 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_584 | success | pass |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_585 | unknown_crop | pass |  | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_586 | unknown_crop | pass |  | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_587 | unknown_crop | pass |  | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_588 | unknown_crop | pass |  | cotton / unknown /  | Part abstained for crop=cotton: no compatible parts configured for crop (cotton) |
| demo_589 | unknown_crop | pass |  | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_590 | unknown_crop | pass |  |  /  /  | SAM3 produced 1 instances for prompts=plant,leaf,fruit,stem but retained 0 detections after ROI filtering/classification (roi_seen=1, roi_kept=0, classification_min_confidence=0.25). |
| demo_591 | unknown_crop | pass |  | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_592 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_593 | unknown_crop | pass |  | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_594 | unknown_crop | pass |  | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_595 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_596 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_597 | unknown_crop | pass |  | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_598 | unknown_crop | pass |  | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_599 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_600 | unknown_crop | pass |  | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_601 | unknown_crop | pass |  | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_602 | unknown_crop | pass |  | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_603 | unknown_crop | pass |  | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_604 | unknown_crop | pass |  | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
