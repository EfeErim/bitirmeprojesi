# M2 Demo Checklist Run

- generated_at: `2026-06-21T21:42:08.718338+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`

## Summary

- total: 512
- passed: 333
- failed: 179
- answered: 371
- abstained_or_reviewed: 141
- asset_ready: 0
- failure_buckets: `{"router": 133}`

## Rows

| image_id | status | pass_fail | failure_bucket | predicted | message |
|---|---|---|---|---|---|
| demo_001 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_002 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_003 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_004 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_005 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_006 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_007 | success | fail |  | tomato / leaf / domates_early_blight_yaprak |  |
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
| demo_019 | router_uncertain | fail | router | grape / fruit /  | Router result is not eligible for adapter prediction. |
| demo_020 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_021 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_022 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_023 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_024 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_025 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_026 | success | fail |  | apricot / fruit / kayısı_sağlıklı_meyve_800 |  |
| demo_027 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_028 | success | pass |  | apricot / fruit / kayısıda_çiçek_monilyası_meyve_40 |  |
| demo_029 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_030 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_031 | success | fail |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_032 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_033 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_034 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_035 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_036 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_037 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_038 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_039 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_040 | success | fail |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_041 | unknown_crop | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_042 | success | fail |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_043 | success | fail |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
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
| demo_064 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_065 | unknown_crop | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_066 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_067 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_068 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
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
| demo_089 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_090 | router_uncertain | fail | router | grape / fruit /  | Router result is not eligible for adapter prediction. |
| demo_091 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_092 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_093 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_094 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_095 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_096 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
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
| demo_112 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_113 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_114 | unknown_crop | fail | router | walnut / unknown /  | Part abstained for crop=walnut: no compatible parts configured for crop (walnut) |
| demo_115 | success | fail |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_116 | success | pass |  | apricot / fruit / kayısıda_şarka_virüsü_meyve_230 |  |
| demo_117 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_118 | success | fail |  | apricot / fruit / kayısı_sağlıklı_meyve_800 |  |
| demo_119 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_120 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_121 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_122 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_123 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_124 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_125 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_126 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_127 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_128 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_129 | router_uncertain | fail | router | tomato / fruit /  | Router result is not eligible for adapter prediction. |
| demo_130 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_131 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_132 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_133 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_134 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_135 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_136 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_137 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_138 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_139 | success | fail |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_140 | success | fail |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_141 | unknown_crop | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_142 | unknown_crop | pass |  | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_143 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_144 | unknown_crop | pass |  | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_145 | success | fail |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_146 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_147 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_148 | success | pass |  | apricot / fruit / kayısıda_şarka_virüsü_meyve_230 |  |
| demo_149 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_150 | success | pass |  | apricot / fruit / kayısıda_şarka_virüsü_meyve_230 |  |
| demo_151 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_152 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_153 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_154 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_155 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_156 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_157 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_158 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_159 | success | fail |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_160 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_161 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_162 | success | fail |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_163 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_164 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_165 | success | fail |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_166 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_167 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_168 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_169 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_170 | success | fail |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_171 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_172 | success | pass |  | apricot / fruit / kayısı_sağlıklı_meyve_800 |  |
| demo_173 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_174 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_175 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_176 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_177 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_178 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_179 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_180 | success | pass |  | apricot / fruit / kayısı_sağlıklı_meyve_800 |  |
| demo_181 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_182 | success | pass |  | apricot / fruit / kayısıda_yaprak_delen_çil_hastalığı_meyve_128 |  |
| demo_183 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_184 | success | pass |  | apricot / fruit / kayısıda_şeftali_karalekesi_meyve_232 |  |
| demo_185 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_186 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_187 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_188 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_189 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_190 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_191 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_192 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_193 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
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
| demo_206 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_207 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_208 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_209 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_210 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_211 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_212 | router_uncertain | fail | router | grape / fruit /  | Router result is not eligible for adapter prediction. |
| demo_213 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_214 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_215 | unknown_crop | fail | router | cassava / unknown /  | Part abstained for crop=cassava: no compatible parts configured for crop (cassava) |
| demo_216 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_217 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_218 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_219 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_220 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_221 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_222 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_223 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_224 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_225 | router_uncertain | fail | router | grape / whole plant /  | Router part 'whole plant' is outside the final demo supported part set. |
| demo_226 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_227 | router_uncertain | fail | router | grape / whole plant /  | Router part 'whole plant' is outside the final demo supported part set. |
| demo_228 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_229 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_230 | router_uncertain | fail | router | grape / fruit /  | Router result is not eligible for adapter prediction. |
| demo_231 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_232 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_233 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_234 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_235 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_236 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_237 | router_uncertain | fail | router | grape / fruit /  | Router result is not eligible for adapter prediction. |
| demo_238 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_239 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_240 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_241 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_242 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_243 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_244 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
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
| demo_256 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_257 | success | fail |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_258 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_259 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_260 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_261 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_262 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_263 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
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
| demo_311 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
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
| demo_340 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_341 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_342 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_343 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_344 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_345 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_346 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_347 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_348 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_349 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
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
| demo_366 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_367 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_368 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_369 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_370 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_371 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_372 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_373 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_374 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_375 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_376 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_377 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_378 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_379 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_380 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_381 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_382 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_383 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_384 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_385 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_386 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_387 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_388 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_389 | success | fail |  | tomato / fruit / domates_sag_lıklı_meyve |  |
| demo_390 | success | fail |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_391 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_392 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_393 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_394 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_395 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_396 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_397 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_398 | unknown_crop | fail | router | avocado / unknown /  | Part abstained for crop=avocado: no compatible parts configured for crop (avocado) |
| demo_399 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_400 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_401 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_402 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_403 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_404 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_405 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_406 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_407 | success | fail |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_408 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_409 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_410 | success | fail |  | tomato / fruit / domates_sag_lıklı_meyve |  |
| demo_411 | success | pass |  | tomato / fruit / domates_gray_mold_meyve |  |
| demo_412 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_413 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_414 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_415 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_416 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_417 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_418 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_419 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_420 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_421 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_422 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_423 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_424 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_425 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_426 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_427 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_428 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_429 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_430 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_431 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_432 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_433 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_434 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_435 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_436 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_437 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_438 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_439 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_440 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_441 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
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
| demo_459 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_460 | unknown_crop | fail | router | coffee / unknown /  | Part abstained for crop=coffee: no compatible parts configured for crop (coffee) |
| demo_461 | unknown_crop | fail | router | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_462 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_463 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_464 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_465 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_466 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_467 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_468 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_469 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_470 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_471 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_472 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_473 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_474 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_475 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_476 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_477 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_478 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_479 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_480 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_481 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_482 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_483 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_484 | unknown_crop | fail | router | watermelon / unknown /  | Part abstained for crop=watermelon: no compatible parts configured for crop (watermelon) |
| demo_485 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_486 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
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
| demo_499 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_500 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_501 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_502 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_503 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_504 | unknown_crop | fail | router | sunflower / unknown /  | Part abstained for crop=sunflower: no compatible parts configured for crop (sunflower) |
| demo_505 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_506 | unknown_crop | fail | router | zucchini / unknown /  | Part abstained for crop=zucchini: no compatible parts configured for crop (zucchini) |
| demo_507 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_508 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_509 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_510 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_511 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_512 | unknown_crop | fail | router | okra / unknown /  | Part abstained for crop=okra: no compatible parts configured for crop (okra) |
| demo_513 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_514 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
