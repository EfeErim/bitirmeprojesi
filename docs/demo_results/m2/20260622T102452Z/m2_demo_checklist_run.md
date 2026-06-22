# M2 Demo Checklist Run

- generated_at: `2026-06-22T10:24:50.893349+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`

## Summary

- total: 512
- passed: 191
- failed: 321
- answered: 233
- abstained_or_reviewed: 279
- asset_ready: 0
- failure_buckets: `{"router": 266}`

## Rows

| image_id | status | pass_fail | failure_bucket | predicted | message |
|---|---|---|---|---|---|
| demo_001 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_002 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_003 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_004 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_005 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_006 | success | fail |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_007 | unknown_crop | fail | router | potato / unknown /  | Part abstained for crop=potato: confidence (0.3830) < threshold (0.4000) |
| demo_008 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_009 | unknown_crop | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_010 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_011 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_012 | unknown_crop | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_013 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_014 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_015 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_016 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_017 | success | fail |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_018 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_019 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_020 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
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
| demo_031 | unknown_crop | fail | router | orange / unknown /  | Part abstained for crop=orange: no compatible parts configured for crop (orange) |
| demo_032 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_033 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2868) < threshold (0.4000); margin (-0.1894) < threshold (0.1000) |
| demo_034 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_035 | unknown_crop | fail | router | thistle / unknown /  | Part abstained for crop=thistle: no compatible parts configured for crop (thistle) |
| demo_036 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_037 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_038 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_039 | unknown_crop | fail | router | black pepper / unknown /  | Part abstained for crop=black pepper: no compatible parts configured for crop (black pepper) |
| demo_040 | unknown_crop | fail | router | basil / unknown /  | Part abstained for crop=basil: no compatible parts configured for crop (basil) |
| demo_041 | unknown_crop | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_042 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_043 | unknown_crop | pass |  | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_044 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_045 | unknown_crop | pass |  | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_046 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_049 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_050 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_051 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_052 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_053 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_054 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_055 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_056 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_057 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_058 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_059 | unknown_crop | fail | router | bean / unknown /  | Part abstained for crop=bean: no compatible parts configured for crop (bean) |
| demo_060 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_061 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_062 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_063 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_064 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_065 | unknown_crop | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_066 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_067 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_068 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_069 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_070 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_071 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_072 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.1831) < threshold (0.4000); margin (0.0078) < threshold (0.1000) |
| demo_073 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_074 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.3823) < threshold (0.4000) |
| demo_075 | success | fail |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_076 | success | fail |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_077 | success | fail |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_078 | success | fail |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_079 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_080 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_081 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_082 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_083 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_084 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_085 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_086 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_087 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_088 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.2469) < threshold (0.4000); margin (0.0897) < threshold (0.1000) |
| demo_089 | unknown_crop | fail | router | black pepper / unknown /  | Part abstained for crop=black pepper: no compatible parts configured for crop (black pepper) |
| demo_090 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_091 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_092 | unknown_crop | fail | router | black pepper / unknown /  | Part abstained for crop=black pepper: no compatible parts configured for crop (black pepper) |
| demo_093 | unknown_crop | fail | router | turmeric / unknown /  | Part abstained for crop=turmeric: no compatible parts configured for crop (turmeric) |
| demo_094 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_095 | unknown_crop | fail | router | black pepper / unknown /  | Part abstained for crop=black pepper: no compatible parts configured for crop (black pepper) |
| demo_096 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_097 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
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
| demo_119 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_120 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_121 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_122 | unknown_crop | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_123 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_124 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_125 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_126 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_127 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_128 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_129 | success | fail |  | tomato / fruit / domates_sag_lıklı_meyve |  |
| demo_130 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_131 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_132 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_133 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.2097) < threshold (0.4000); margin (0.0159) < threshold (0.1000) |
| demo_134 | unknown_crop | fail | router | daisy / unknown /  | Part abstained for crop=daisy: no compatible parts configured for crop (daisy) |
| demo_135 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_136 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
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
| demo_162 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_163 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_164 | unknown_crop | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_165 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
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
| demo_185 | unknown_crop | fail | router | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_186 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_187 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_188 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_189 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_190 | unknown_crop | fail | router | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_191 | unknown_crop | fail | router | orange / unknown /  | Part abstained for crop=orange: no compatible parts configured for crop (orange) |
| demo_192 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_193 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_194 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_195 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_196 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_197 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_198 | unknown_crop | fail | router | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_199 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_200 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_201 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_202 | unknown_crop | fail | router | apple / unknown /  | Part abstained for crop=apple: margin (0.0637) < threshold (0.1000) |
| demo_203 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_204 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_205 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_206 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_207 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_208 | unknown_crop | fail | router | cotton / unknown /  | Part abstained for crop=cotton: no compatible parts configured for crop (cotton) |
| demo_209 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_210 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_211 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_212 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_213 | unknown_crop | fail | router | dates / unknown /  | Part abstained for crop=dates: no compatible parts configured for crop (dates) |
| demo_214 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_215 | unknown_crop | fail | router | cassava / unknown /  | Part abstained for crop=cassava: no compatible parts configured for crop (cassava) |
| demo_216 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_217 | success | fail |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_218 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_219 | unknown_crop | fail | router | black pepper / unknown /  | Part abstained for crop=black pepper: no compatible parts configured for crop (black pepper) |
| demo_220 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_221 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_222 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_223 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_224 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_225 | router_uncertain | fail | router | grape / whole plant /  | Router part 'whole plant' is outside the final demo supported part set. |
| demo_226 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_227 | router_uncertain | fail | router | grape / whole plant /  | Router part 'whole plant' is outside the final demo supported part set. |
| demo_228 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_229 | success | fail |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_230 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_231 | unknown_crop | fail | router | dates / unknown /  | Part abstained for crop=dates: no compatible parts configured for crop (dates) |
| demo_232 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_233 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_234 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_235 | unknown_crop | fail | router | dates / unknown /  | Part abstained for crop=dates: no compatible parts configured for crop (dates) |
| demo_236 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_237 | success | fail |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_238 | success | fail |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_239 | unknown_crop | fail | router | dates / unknown /  | Part abstained for crop=dates: no compatible parts configured for crop (dates) |
| demo_240 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_241 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_242 | success | fail |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_243 | success | fail |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_244 | success | fail |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_245 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_246 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_247 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_248 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_249 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_250 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_251 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_252 | router_uncertain | fail | router | grape / unknown /  | Part abstained for crop=grape: confidence (0.2618) < threshold (0.4000); margin (-0.2094) < threshold (0.1000) |
| demo_253 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_254 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_255 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_256 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_257 | success | fail |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_258 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_259 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_260 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_261 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_262 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_263 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_264 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_265 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
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
| demo_295 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
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
| demo_307 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_308 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_309 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_310 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_311 | unknown_crop | fail | router | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_312 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_313 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.3319) < threshold (0.4000) |
| demo_314 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_315 | success | fail |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_316 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.2295) < threshold (0.4000) |
| demo_317 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_318 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_319 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_320 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_321 | unknown_crop | fail | router | cantaloupe / unknown /  | Part abstained for crop=cantaloupe: no compatible parts configured for crop (cantaloupe) |
| demo_322 | unknown_crop | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_323 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_324 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_325 | unknown_crop | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_326 | unknown_crop | fail | router | blackberry / unknown /  | Part abstained for crop=blackberry: no compatible parts configured for crop (blackberry) |
| demo_327 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_328 | unknown_crop | fail | router | cantaloupe / unknown /  | Part abstained for crop=cantaloupe: no compatible parts configured for crop (cantaloupe) |
| demo_329 | unknown_crop | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_330 | unknown_crop | fail | router | blackberry / unknown /  | Part abstained for crop=blackberry: no compatible parts configured for crop (blackberry) |
| demo_331 | success | fail |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_332 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_333 | unknown_crop | fail | router | cantaloupe / unknown /  | Part abstained for crop=cantaloupe: no compatible parts configured for crop (cantaloupe) |
| demo_334 | unknown_crop | fail | router | cantaloupe / unknown /  | Part abstained for crop=cantaloupe: no compatible parts configured for crop (cantaloupe) |
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
| demo_350 | unknown_crop | fail | router | rose / unknown /  | Part abstained for crop=rose: no compatible parts configured for crop (rose) |
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
| demo_361 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.2846) < threshold (0.4000) |
| demo_362 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_363 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_364 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_365 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_366 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_367 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_368 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_369 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_370 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_371 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_372 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_373 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_374 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_375 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_376 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_377 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_378 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_379 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_380 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_381 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_382 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_383 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_384 | success | fail |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_385 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_386 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_387 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_388 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_389 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_390 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_391 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_392 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_393 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_394 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_395 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_396 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_397 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_398 | unknown_crop | fail | router | avocado / unknown /  | Part abstained for crop=avocado: no compatible parts configured for crop (avocado) |
| demo_399 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_400 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_401 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_402 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_403 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_404 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_405 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_406 | unknown_crop | fail | router | cucumber / fruit /  | Router crop 'cucumber' is outside the final demo supported crop set. |
| demo_407 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_408 | unknown_crop | fail | router | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_409 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_410 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_411 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_412 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_413 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_414 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_415 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_416 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_417 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_418 | unknown_crop | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_419 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_420 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_421 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_422 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_423 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_424 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_425 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_426 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_427 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_428 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_429 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_430 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2671) < threshold (0.4000); margin (0.0232) < threshold (0.1000) |
| demo_431 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_432 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_433 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_434 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_435 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_436 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_437 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_438 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_439 | success | fail |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_440 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_441 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_442 | success | fail |  | tomato / leaf / domates_sağlıklı_yaprak |  |
| demo_443 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_444 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_445 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_446 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_447 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_448 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_449 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_450 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_451 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_452 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_453 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.3982) < threshold (0.4000) |
| demo_454 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2688) < threshold (0.4000); margin (0.0436) < threshold (0.1000) |
| demo_455 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_456 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_457 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_458 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_459 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_460 | unknown_crop | fail | router | coffee / unknown /  | Part abstained for crop=coffee: no compatible parts configured for crop (coffee) |
| demo_461 | unknown_crop | fail | router | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_462 | unknown_crop | fail | router | quinoa / unknown /  | Part abstained for crop=quinoa: no compatible parts configured for crop (quinoa) |
| demo_463 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_464 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_465 | unknown_crop | fail | router | cabbage / unknown /  | Part abstained for crop=cabbage: no compatible parts configured for crop (cabbage) |
| demo_466 | unknown_crop | fail | router | turmeric / unknown /  | Part abstained for crop=turmeric: no compatible parts configured for crop (turmeric) |
| demo_467 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_468 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_469 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_470 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_471 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_472 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_473 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_474 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_475 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_476 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_477 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_478 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_479 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_480 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_481 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_482 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_483 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_484 | unknown_crop | fail | router | watermelon / unknown /  | Part abstained for crop=watermelon: no compatible parts configured for crop (watermelon) |
| demo_485 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_486 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_487 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_488 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_489 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_490 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_491 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2691) < threshold (0.4000); margin (0.0098) < threshold (0.1000) |
| demo_492 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_493 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_494 | unknown_crop | fail | router | spinach / unknown /  | Part abstained for crop=spinach: no compatible parts configured for crop (spinach) |
| demo_495 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_496 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_497 | success | pass |  | tomato / leaf / domates_sağlıklı_yaprak |  |
| demo_498 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_499 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_500 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_501 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_502 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_503 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_504 | unknown_crop | fail | router | sunflower / unknown /  | Part abstained for crop=sunflower: no compatible parts configured for crop (sunflower) |
| demo_505 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_506 | unknown_crop | fail | router | zucchini / unknown /  | Part abstained for crop=zucchini: no compatible parts configured for crop (zucchini) |
| demo_507 | unknown_crop | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_508 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_509 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_510 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_511 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_512 | unknown_crop | fail | router | okra / unknown /  | Part abstained for crop=okra: no compatible parts configured for crop (okra) |
| demo_513 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_514 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
