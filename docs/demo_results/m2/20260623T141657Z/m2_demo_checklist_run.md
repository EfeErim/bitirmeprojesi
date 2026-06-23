# M2 Demo Checklist Run

- generated_at: `2026-06-23T14:16:55.429425+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`

## Summary

- total: 512
- passed: 264
- failed: 248
- answered: 280
- abstained_or_reviewed: 232
- asset_ready: 0
- failure_buckets: `{"router": 219}`

## Rows

| image_id | status | pass_fail | failure_bucket | predicted | message |
|---|---|---|---|---|---|
| demo_001 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_002 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_003 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_004 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_005 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_006 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_007 | unknown_crop | fail | router | potato / unknown /  | Part abstained for crop=potato: confidence (0.3830) < threshold (0.4000) |
| demo_008 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_009 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_010 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_011 | router_uncertain | fail | router | strawberry / leaf /  | Router result is not eligible for adapter prediction. |
| demo_012 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_013 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_014 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_015 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_016 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_017 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_018 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_019 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_020 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_021 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_022 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_023 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_024 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_025 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_026 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_027 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_028 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
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
| demo_040 | unknown_crop | fail | router | basil / unknown /  | Part abstained for crop=basil: no compatible parts configured for crop (basil) |
| demo_041 | unknown_crop | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_042 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_043 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_044 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_045 | unknown_crop | pass |  | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_046 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
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
| demo_059 | unknown_crop | fail | router | bean / unknown /  | Part abstained for crop=bean: no compatible parts configured for crop (bean) |
| demo_060 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_061 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_062 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_063 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_064 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_065 | unknown_crop | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_066 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
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
| demo_109 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_110 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_111 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_112 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_113 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_114 | unknown_crop | fail | router | walnut / unknown /  | Part abstained for crop=walnut: no compatible parts configured for crop (walnut) |
| demo_115 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_116 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_117 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_118 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
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
| demo_130 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_131 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_132 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_133 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_134 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_135 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_136 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_137 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_138 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_139 | unknown_crop | pass |  | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_140 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_141 | unknown_crop | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_142 | unknown_crop | pass |  | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_143 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_144 | unknown_crop | pass |  | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_145 | unknown_crop | fail | router | cucumber / fruit /  | Router crop 'cucumber' is outside the final demo supported crop set. |
| demo_146 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_147 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_148 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_149 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_150 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_151 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_152 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_153 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_154 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_155 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_156 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_157 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_158 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_159 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_160 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_161 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_162 | success | fail |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_163 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_164 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_165 | success | fail |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_166 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_167 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_168 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_169 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_170 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_171 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_172 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_173 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_174 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_175 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_176 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_177 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_178 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_179 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_180 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_181 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_182 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_183 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_184 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
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
| demo_212 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
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
| demo_224 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_225 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_226 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_227 | router_uncertain | fail | router | grape / whole plant /  | Router part 'whole plant' is outside the final demo supported part set. |
| demo_228 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_229 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_230 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_231 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_232 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_233 | success | fail |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_234 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_235 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_236 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_237 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_238 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_239 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_240 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_241 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_242 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_243 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
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
| demo_356 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
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
| demo_372 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_373 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_374 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_375 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_376 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_377 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
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
| demo_389 | success | fail |  | tomato / fruit / domates_sag_lıklı_meyve |  |
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
| demo_416 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_417 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_418 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_419 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_420 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_421 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_422 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_423 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_424 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_425 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_426 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_427 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_428 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_429 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_430 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2671) < threshold (0.4000); margin (0.0232) < threshold (0.1000) |
| demo_431 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_432 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_433 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_434 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_435 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_436 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_437 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_438 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_439 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_440 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_441 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_442 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_443 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_444 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_445 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_446 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_447 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_448 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_449 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_450 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_451 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_452 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_453 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.3982) < threshold (0.4000) |
| demo_454 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2688) < threshold (0.4000); margin (0.0436) < threshold (0.1000) |
| demo_455 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_456 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_457 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_458 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_459 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_460 | unknown_crop | fail | router | coffee / unknown /  | Part abstained for crop=coffee: no compatible parts configured for crop (coffee) |
| demo_461 | unknown_crop | fail | router | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_462 | unknown_crop | fail | router | quinoa / unknown /  | Part abstained for crop=quinoa: no compatible parts configured for crop (quinoa) |
| demo_463 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_464 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_465 | unknown_crop | fail | router | cabbage / unknown /  | Part abstained for crop=cabbage: no compatible parts configured for crop (cabbage) |
| demo_466 | unknown_crop | fail | router | turmeric / unknown /  | Part abstained for crop=turmeric: no compatible parts configured for crop (turmeric) |
| demo_467 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_468 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_469 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_470 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_471 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_472 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_473 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_474 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_475 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_476 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_477 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_478 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_479 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_480 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_481 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_482 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_483 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_484 | unknown_crop | fail | router | watermelon / unknown /  | Part abstained for crop=watermelon: no compatible parts configured for crop (watermelon) |
| demo_485 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_486 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_487 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_488 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_489 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_490 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_491 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2691) < threshold (0.4000); margin (0.0098) < threshold (0.1000) |
| demo_492 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_493 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_494 | unknown_crop | fail | router | spinach / unknown /  | Part abstained for crop=spinach: no compatible parts configured for crop (spinach) |
| demo_495 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_496 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_497 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_498 | success | fail |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_499 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_500 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_501 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_502 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_503 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_504 | unknown_crop | fail | router | sunflower / unknown /  | Part abstained for crop=sunflower: no compatible parts configured for crop (sunflower) |
| demo_505 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_506 | unknown_crop | fail | router | zucchini / unknown /  | Part abstained for crop=zucchini: no compatible parts configured for crop (zucchini) |
| demo_507 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_508 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_509 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_510 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_511 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_512 | unknown_crop | fail | router | okra / unknown /  | Part abstained for crop=okra: no compatible parts configured for crop (okra) |
| demo_513 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_514 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
