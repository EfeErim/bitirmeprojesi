# M2 Demo Checklist Run

- generated_at: `2026-06-17T08:08:41.863591+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`

## Summary

- total: 512
- passed: 261
- failed: 251
- answered: 246
- abstained_or_reviewed: 266
- asset_ready: 0
- failure_buckets: `{"adapter_loading": 26, "router": 223}`

## Rows

| image_id | status | pass_fail | failure_bucket | predicted | message |
|---|---|---|---|---|---|
| demo_001 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_002 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_003 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_004 | success | pass |  | tomato / fruit / domates_sag_lıklı_meyve |  |
| demo_005 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_006 | router_uncertain | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_007 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_008 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_009 | router_uncertain | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_010 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.3823) < threshold (0.4000) |
| demo_011 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_012 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_013 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.3397) < threshold (0.4000) |
| demo_014 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_015 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_016 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_017 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_018 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_019 | router_uncertain | fail | router | cassava / unknown /  | Part abstained for crop=cassava: no compatible parts configured for crop (cassava) |
| demo_020 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_021 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_022 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_023 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_024 | success | pass |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_025 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_026 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_027 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_028 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_029 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_030 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_031 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_032 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_033 | router_uncertain | pass |  | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_034 | router_uncertain | pass |  | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_035 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_036 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_037 | router_uncertain | pass |  | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_038 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_039 | adapter_unavailable | pass |  | apple / leaf /  | Adapter not found for crop 'apple' part 'leaf' under runs |
| demo_040 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_041 | router_uncertain | pass |  | blueberry / unknown /  | Part abstained for crop=blueberry: no compatible parts configured for crop (blueberry) |
| demo_042 | adapter_unavailable | pass |  | apple / bud /  | Adapter not found for crop 'apple' part 'bud' under runs |
| demo_043 | success | fail |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_044 | success | fail |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_045 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_046 | router_uncertain | pass |  | lambsquarters / unknown /  | Part abstained for crop=lambsquarters: no compatible parts configured for crop (lambsquarters) |
| demo_049 | router_uncertain | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_050 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_051 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_052 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_053 | adapter_unavailable | fail | adapter_loading | tomato / whole plant /  | Adapter not found for crop 'tomato' part 'whole plant' under runs |
| demo_054 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_055 | router_uncertain | fail | router | blueberry / unknown /  | Part abstained for crop=blueberry: no compatible parts configured for crop (blueberry) |
| demo_056 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_057 | router_uncertain | fail | router | daisy / unknown /  | Part abstained for crop=daisy: no compatible parts configured for crop (daisy) |
| demo_058 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_059 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_060 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_061 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_062 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_063 | router_uncertain | fail | router | fig / unknown /  | Part abstained for crop=fig: no compatible parts configured for crop (fig) |
| demo_064 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_065 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_066 | router_uncertain | fail | router | hemp / unknown /  | Part abstained for crop=hemp: no compatible parts configured for crop (hemp) |
| demo_067 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_068 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_069 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_070 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_071 | router_uncertain | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_072 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_073 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_074 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_075 | router_uncertain | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_076 | adapter_unavailable | fail | adapter_loading | potato / tuber /  | Adapter not found for crop 'potato' part 'tuber' under runs |
| demo_077 | router_uncertain | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_078 | router_uncertain | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_079 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_080 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_081 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_082 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_083 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_084 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_085 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_086 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_087 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_088 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_089 | adapter_unavailable | fail | adapter_loading | apple / leaf /  | Adapter not found for crop 'apple' part 'leaf' under runs |
| demo_090 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_091 | router_uncertain | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_092 | unknown_crop | fail | router |  /  /  | SAM3 produced 2 instances for prompts=plant,leaf,fruit,stem but retained 0 detections after ROI filtering/classification (roi_seen=2, roi_kept=0, classification_min_confidence=0.25). |
| demo_093 | adapter_unavailable | fail | adapter_loading | pepper / bud /  | Adapter not found for crop 'pepper' part 'bud' under runs |
| demo_094 | adapter_unavailable | fail | adapter_loading | pepper / bud /  | Adapter not found for crop 'pepper' part 'bud' under runs |
| demo_095 | router_uncertain | fail | router | orange / unknown /  | Part abstained for crop=orange: no compatible parts configured for crop (orange) |
| demo_096 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_097 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_098 | success | pass |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_099 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_100 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_101 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_102 | router_uncertain | fail | router | grape / unknown /  | Part abstained for crop=grape: confidence (0.2817) < threshold (0.4000); margin (-0.2195) < threshold (0.1000) |
| demo_103 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_104 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_105 | success | pass |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_106 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_107 | router_uncertain | fail | router | onion / unknown /  | Part abstained for crop=onion: no compatible parts configured for crop (onion) |
| demo_108 | router_uncertain | fail | router | fig / unknown /  | Part abstained for crop=fig: no compatible parts configured for crop (fig) |
| demo_109 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_110 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_111 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_112 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_113 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_114 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_115 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_116 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_117 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_118 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_119 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_120 | router_uncertain | fail | router | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_121 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_122 | router_uncertain | fail | router | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_123 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_124 | success | pass |  | apricot / leaf / kayısı_sağlıklı_yaprak_302 |  |
| demo_125 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_126 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_127 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_128 | success | pass |  | apricot / leaf / kayısı_yaprak_delen_çil_hastalığı_yaprak_300 |  |
| demo_129 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_130 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_131 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_132 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_133 | router_uncertain | fail | router | daisy / unknown /  | Part abstained for crop=daisy: no compatible parts configured for crop (daisy) |
| demo_134 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_135 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_136 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_137 | router_uncertain | pass |  | blueberry / unknown /  | Part abstained for crop=blueberry: no compatible parts configured for crop (blueberry) |
| demo_138 | adapter_unavailable | pass |  | apple / bud /  | Adapter not found for crop 'apple' part 'bud' under runs |
| demo_139 | adapter_unavailable | pass |  | pepper / leaf /  | Adapter not found for crop 'pepper' part 'leaf' under runs |
| demo_140 | adapter_unavailable | pass |  | pepper / leaf /  | Adapter not found for crop 'pepper' part 'leaf' under runs |
| demo_141 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_142 | router_uncertain | pass |  | lambsquarters / unknown /  | Part abstained for crop=lambsquarters: no compatible parts configured for crop (lambsquarters) |
| demo_143 | unknown_crop | pass |  |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_144 | router_uncertain | pass |  | cacao / unknown /  | Part abstained for crop=cacao: no compatible parts configured for crop (cacao) |
| demo_145 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_146 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_147 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_148 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_149 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_150 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_151 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_152 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_153 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_154 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_155 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_156 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_157 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_158 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_159 | router_uncertain | fail | router | chickpea / unknown /  | Part abstained for crop=chickpea: no compatible parts configured for crop (chickpea) |
| demo_160 | router_uncertain | fail | router | mango / unknown /  | Part abstained for crop=mango: no compatible parts configured for crop (mango) |
| demo_161 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_162 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_163 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_164 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_165 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_166 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_167 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_168 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_169 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_170 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_171 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_172 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_173 | router_uncertain | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_174 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_175 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_176 | router_uncertain | fail | router | mango / unknown /  | Part abstained for crop=mango: no compatible parts configured for crop (mango) |
| demo_177 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_178 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_179 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_180 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_181 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_182 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_183 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_184 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_185 | success | pass |  | apricot / leaf / kayısı_şarka_virüsü_yaprak_206 |  |
| demo_186 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_187 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_188 | router_uncertain | fail | router | almond / unknown /  | Part abstained for crop=almond: no compatible parts configured for crop (almond) |
| demo_189 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_190 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_191 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_192 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_193 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_194 | router_uncertain | fail | router | peach / unknown /  | Part abstained for crop=peach: no compatible parts configured for crop (peach) |
| demo_195 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_196 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_197 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_198 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_199 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_200 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_201 | router_uncertain | fail | router | cherry / unknown /  | Part abstained for crop=cherry: no compatible parts configured for crop (cherry) |
| demo_202 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_203 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_204 | router_uncertain | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_205 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_206 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_207 | router_uncertain | fail | router | black pepper / unknown /  | Part abstained for crop=black pepper: no compatible parts configured for crop (black pepper) |
| demo_208 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_209 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_210 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_211 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_212 | success | pass |  | grape / fruit / u_zu_m_antraknoz_meyve |  |
| demo_213 | router_uncertain | fail | router | black pepper / unknown /  | Part abstained for crop=black pepper: no compatible parts configured for crop (black pepper) |
| demo_214 | success | pass |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_215 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_216 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_217 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_218 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_219 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_220 | router_uncertain | fail | router | grape / unknown /  | Part abstained for crop=grape: confidence (0.3229) < threshold (0.4000) |
| demo_221 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_222 | success | pass |  | grape / fruit / u_zu_m_botrytis_cinerea_meyve |  |
| demo_223 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_224 | router_uncertain | fail | router | cotton / unknown /  | Part abstained for crop=cotton: no compatible parts configured for crop (cotton) |
| demo_225 | router_uncertain | fail | router | cassava / unknown /  | Part abstained for crop=cassava: no compatible parts configured for crop (cassava) |
| demo_226 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_227 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_228 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_229 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_230 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_231 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_232 | router_uncertain | fail | router | dates / unknown /  | Part abstained for crop=dates: no compatible parts configured for crop (dates) |
| demo_233 | success | pass |  | grape / fruit / u_zu_m_ku_lleme_meyve |  |
| demo_234 | router_uncertain | fail | router | dates / unknown /  | Part abstained for crop=dates: no compatible parts configured for crop (dates) |
| demo_235 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_236 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_237 | adapter_unavailable | fail | adapter_loading | grape / whole plant /  | Adapter not found for crop 'grape' part 'whole plant' under runs |
| demo_238 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_239 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_240 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_241 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_242 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_243 | success | pass |  | grape / fruit / u_zu_m_mildiyo_meyve |  |
| demo_244 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_245 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_246 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_247 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_248 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_249 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_250 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_251 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_252 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_253 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_254 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_255 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_256 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_257 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_258 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_259 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_260 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_261 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_262 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_263 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_264 | success | pass |  | grape / leaf / üzüm_kav_esca_yaprak |  |
| demo_265 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_266 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_267 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_268 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_269 | router_uncertain | fail | router | grape / unknown /  | Part abstained for crop=grape: confidence (0.2583) < threshold (0.4000); margin (-0.2166) < threshold (0.1000) |
| demo_270 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_271 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_272 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_273 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_274 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_275 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_276 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_277 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_278 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_279 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_280 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_281 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_282 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_283 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_284 | success | pass |  | grape / leaf / üzüm_leafroll_virüs_yaprak |  |
| demo_285 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_286 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_287 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_288 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_289 | adapter_unavailable | fail | adapter_loading | apple / leaf /  | Adapter not found for crop 'apple' part 'leaf' under runs |
| demo_290 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_291 | success | pass |  | grape / leaf / üzüm_külleme_yaprak |  |
| demo_292 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_293 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_294 | success | pass |  | grape / leaf / üzüm_mildiyö_yaprak |  |
| demo_295 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_296 | success | pass |  | grape / leaf / üzüm_antraknoz_yaprak |  |
| demo_297 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_298 | success | pass |  | grape / leaf / üzüm_sağlıklı_yaprak |  |
| demo_299 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_300 | router_uncertain | fail | router | celery / unknown /  | Part abstained for crop=celery: no compatible parts configured for crop (celery) |
| demo_301 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_302 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_303 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_304 | success | pass |  | grape / leaf / üzüm_yelpaze_virüsü_yaprak |  |
| demo_305 | router_uncertain | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_306 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_307 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_308 | router_uncertain | fail | router | pineapple / unknown /  | Part abstained for crop=pineapple: no compatible parts configured for crop (pineapple) |
| demo_309 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_310 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_311 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_312 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_313 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.1831) < threshold (0.4000); margin (0.0077) < threshold (0.1000) |
| demo_314 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_315 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.3823) < threshold (0.4000) |
| demo_316 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_317 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_318 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_319 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_320 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.2097) < threshold (0.4000); margin (0.0159) < threshold (0.1000) |
| demo_321 | router_uncertain | fail | router | daisy / unknown /  | Part abstained for crop=daisy: no compatible parts configured for crop (daisy) |
| demo_322 | router_uncertain | fail | router | orange / unknown /  | Part abstained for crop=orange: no compatible parts configured for crop (orange) |
| demo_323 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_324 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_325 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_326 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_327 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_328 | router_uncertain | fail | router | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_329 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_330 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.3321) < threshold (0.4000) |
| demo_331 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_332 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_333 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.2295) < threshold (0.4000) |
| demo_334 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_335 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_336 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_337 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_338 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_339 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_340 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_341 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_342 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_343 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_344 | success | pass |  | strawberry / leaf / strawberry_leaf_scorch_leaf |  |
| demo_345 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_346 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_347 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_348 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_349 | router_uncertain | fail | router | strawberry / unknown /  | Part abstained for crop=strawberry: confidence (0.2469) < threshold (0.4000); margin (0.0897) < threshold (0.1000) |
| demo_350 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_351 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_352 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_353 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_354 | success | pass |  | strawberry / leaf / strawberry_leaf_spot_leaf |  |
| demo_355 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_356 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_357 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_358 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_359 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_360 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_361 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_362 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_363 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_364 | success | pass |  | strawberry / leaf / strawberry_powdery_mildew_leaf |  |
| demo_365 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_366 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_367 | success | pass |  | tomato / fruit / domates_antraknoz_meyve |  |
| demo_368 | router_uncertain | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_369 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_370 | router_uncertain | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_371 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_372 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_373 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_374 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_375 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_376 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_377 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_378 | router_uncertain | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_379 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_380 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_381 | router_uncertain | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_382 | router_uncertain | fail | router | lemon / unknown /  | Part abstained for crop=lemon: no compatible parts configured for crop (lemon) |
| demo_383 | success | pass |  | tomato / fruit / domates_bacterial_spot_and_speck_meyve |  |
| demo_384 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_385 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_386 | router_uncertain | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_387 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_388 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_389 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_390 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_391 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_392 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_393 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_394 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_395 | success | pass |  | tomato / fruit / domates_blossom_end_rot_meyve |  |
| demo_396 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_397 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_398 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_399 | router_uncertain | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_400 | router_uncertain | fail | router | rose / unknown /  | Part abstained for crop=rose: no compatible parts configured for crop (rose) |
| demo_401 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_402 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_403 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_404 | router_uncertain | fail | router | pumpkin / unknown /  | Part abstained for crop=pumpkin: no compatible parts configured for crop (pumpkin) |
| demo_405 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_406 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_407 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_408 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_409 | router_uncertain | fail | router | avocado / unknown /  | Part abstained for crop=avocado: no compatible parts configured for crop (avocado) |
| demo_410 | adapter_unavailable | fail | adapter_loading | pepper / bud /  | Adapter not found for crop 'pepper' part 'bud' under runs |
| demo_411 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_412 | router_uncertain | fail | router | mango / unknown /  | Part abstained for crop=mango: no compatible parts configured for crop (mango) |
| demo_413 | success | pass |  | tomato / fruit / domates_late_blight_meyve |  |
| demo_414 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_415 | router_uncertain | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_416 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_417 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_418 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_419 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_420 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_421 | adapter_unavailable | fail | adapter_loading | cucumber / fruit /  | Adapter not found for crop 'cucumber' part 'fruit' under runs |
| demo_422 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_423 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_424 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_425 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_426 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_427 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_428 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_429 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_430 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_431 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_432 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_433 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_434 | success | pass |  | tomato / leaf / domates_bacterial_spot_and_speck_yaprak |  |
| demo_435 | router_uncertain | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_436 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_437 | router_uncertain | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_438 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_439 | success | pass |  | tomato / fruit / domates_sag_lıklı_meyve |  |
| demo_440 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_441 | router_uncertain | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_442 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_443 | success | pass |  | tomato / leaf / domates_early_blight_yaprak |  |
| demo_444 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_445 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_446 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2670) < threshold (0.4000); margin (0.0233) < threshold (0.1000) |
| demo_447 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_448 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_449 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_450 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_451 | router_uncertain | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_452 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_453 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_454 | success | pass |  | tomato / leaf / domates_late_blight_yaprak |  |
| demo_455 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_456 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_457 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_458 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_459 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_460 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_461 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_462 | success | pass |  | tomato / leaf / domates_leaf_mold_yaprak |  |
| demo_463 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.3536) < threshold (0.4000) |
| demo_464 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_465 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_466 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_467 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_468 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_469 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_470 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_471 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.3982) < threshold (0.4000) |
| demo_472 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2688) < threshold (0.4000); margin (0.0435) < threshold (0.1000) |
| demo_473 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_474 | success | pass |  | tomato / leaf / domates_mosaic_virüs_yaprak |  |
| demo_475 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_476 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_477 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_478 | router_uncertain | fail | router | coffee / unknown /  | Part abstained for crop=coffee: no compatible parts configured for crop (coffee) |
| demo_479 | router_uncertain | fail | router | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_480 | router_uncertain | fail | router | quinoa / unknown /  | Part abstained for crop=quinoa: no compatible parts configured for crop (quinoa) |
| demo_481 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_482 | success | pass |  | tomato / leaf / domates_powdery_mildew_yaprak |  |
| demo_483 | router_uncertain | fail | router | cabbage / unknown /  | Part abstained for crop=cabbage: no compatible parts configured for crop (cabbage) |
| demo_484 | router_uncertain | fail | router | turmeric / unknown /  | Part abstained for crop=turmeric: no compatible parts configured for crop (turmeric) |
| demo_485 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_486 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_487 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_488 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_489 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_490 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_491 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_492 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_493 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_494 | success | pass |  | tomato / leaf / domates_septoria_leaf_spot_yaprak |  |
| demo_495 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_496 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_497 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_498 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_499 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_500 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_501 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_502 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_503 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_504 | success | pass |  | tomato / leaf / domates_spotted_wilt_yaprak |  |
| demo_505 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_506 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_507 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_508 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_509 | adapter_unavailable | fail | adapter_loading | pepper / bud /  | Adapter not found for crop 'pepper' part 'bud' under runs |
| demo_510 | success | pass |  | tomato / leaf / domates_yellow_leaf_curl_yaprak |  |
| demo_511 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.3399) < threshold (0.4000) |
| demo_512 | router_uncertain | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_513 | adapter_unavailable | fail | adapter_loading | potato / leaf /  | Adapter not found for crop 'potato' part 'leaf' under runs |
| demo_514 | router_uncertain | fail | router | spinach / unknown /  | Part abstained for crop=spinach: no compatible parts configured for crop (spinach) |
