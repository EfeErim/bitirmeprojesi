# M2 Demo Checklist Run

- started_at: `2026-06-28T13:02:07.114450+00:00`
- finished_at: `2026-06-28T13:07:33.659144+00:00`
- elapsed: `5m 27s` (326.545s)
- generated_at: `2026-06-28T13:07:33.659144+00:00`
- checklist: `docs/demo_checklist.md`
- device: `cuda`
- adapter_root: `runs`
- mode: `official`
- batch_size: `12`
- adapter_batch_size: `32`
- handoff_cache: `{"enabled": true, "path": "/content/bitirmeprojesi/.runtime_tmp/m2_router_prototype_handoff_cache.json", "refresh": true, "stats": {"hits": 0, "misses": 89, "writes": 89}}`

## Summary

- total: 89
- passed: 9
- failed: 80
- answered: 14
- abstained_or_reviewed: 75
- asset_ready: 0
- failure_buckets: `{"router": 75}`

## Rows

| image_id | status | pass_fail | failure_bucket | predicted | message |
|---|---|---|---|---|---|
| demo_145 | unknown_crop | fail | router | cucumber / fruit /  | Router crop 'cucumber' is outside the final demo supported crop set. |
| demo_146 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_174 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_029 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_030 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_285 | unknown_crop | fail | router | celery / unknown /  | Part abstained for crop=celery: no compatible parts configured for crop (celery) |
| demo_305 | unknown_crop | fail | router | orange / unknown /  | Part abstained for crop=orange: no compatible parts configured for crop (orange) |
| demo_308 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_311 | unknown_crop | fail | router | pear / unknown /  | Part abstained for crop=pear: no compatible parts configured for crop (pear) |
| demo_334 | success | fail |  | strawberry / fruit / strawberry_unripe_fruit |  |
| demo_056 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_057 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_373 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_383 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_388 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_396 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_400 | success | pass |  | tomato / fruit / domates_spotted_wilt_meyve |  |
| demo_405 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_412 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_007 | unknown_crop | fail | router | potato / unknown /  | Part abstained for crop=potato: confidence (0.3830) < threshold (0.4000) |
| demo_064 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_065 | unknown_crop | fail | router | guava / unknown /  | Part abstained for crop=guava: no compatible parts configured for crop (guava) |
| demo_429 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_430 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2671) < threshold (0.4000); margin (0.0232) < threshold (0.1000) |
| demo_432 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_440 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_461 | unknown_crop | fail | router | lentil / unknown /  | Part abstained for crop=lentil: no compatible parts configured for crop (lentil) |
| demo_462 | unknown_crop | fail | router | quinoa / unknown /  | Part abstained for crop=quinoa: no compatible parts configured for crop (quinoa) |
| demo_058 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_366 | unknown_crop | fail | router | banana / unknown /  | Part abstained for crop=banana: no compatible parts configured for crop (banana) |
| demo_395 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_398 | unknown_crop | fail | router | avocado / unknown /  | Part abstained for crop=avocado: no compatible parts configured for crop (avocado) |
| demo_399 | unknown_crop | fail | router | pepper / bud /  | Router crop 'pepper' is outside the final demo supported crop set. |
| demo_409 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_509 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_147 | unknown_crop | fail | router |  /  /  | No SAM3 instances for prompts=plant,leaf,fruit,stem threshold=0.60. |
| demo_436 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_097 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_236 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_237 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_243 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_307 | success | fail |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_129 | router_uncertain | fail | router | tomato / fruit /  | Router result is not eligible for adapter prediction. |
| demo_067 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_425 | unknown_crop | fail | router | sugar beet / unknown /  | Part abstained for crop=sugar beet: no compatible parts configured for crop (sugar beet) |
| demo_162 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_265 | router_uncertain | fail | router | grape / leaf /  | Router result is not eligible for adapter prediction. |
| demo_356 | router_uncertain | fail | router | strawberry / fruit /  | Router result is not eligible for adapter prediction. |
| demo_006 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_439 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_442 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_117 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_118 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_150 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_166 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_168 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_172 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_173 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_180 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_372 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_377 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_384 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_389 | unknown_crop | fail | router | eggplant / unknown /  | Part abstained for crop=eggplant: no compatible parts configured for crop (eggplant) |
| demo_063 | unknown_crop | fail | router | petunia / unknown /  | Part abstained for crop=petunia: no compatible parts configured for crop (petunia) |
| demo_431 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_433 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_435 | unknown_crop | fail | router | honeydew / unknown /  | Part abstained for crop=honeydew: no compatible parts configured for crop (honeydew) |
| demo_438 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_441 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_467 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
| demo_011 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_070 | success | pass |  | strawberry / fruit / strawberry_anthracnose_fruit |  |
| demo_075 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_076 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_077 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_078 | success | pass |  | strawberry / fruit / strawberry_gray_mold_fruit |  |
| demo_310 | success | pass |  | strawberry / fruit / strawberry_powdery_mildew_fruit |  |
| demo_323 | success | fail |  | strawberry / fruit / strawberry_unripe_fruit |  |
| demo_324 | success | fail |  | strawberry / fruit / strawberry_unripe_fruit |  |
| demo_327 | success | pass |  | strawberry / fruit / strawberry_healthy_fruit |  |
| demo_331 | success | fail |  | strawberry / fruit / strawberry_unripe_fruit |  |
| demo_124 | router_uncertain | fail | router | apricot / leaf /  | Router result is not eligible for adapter prediction. |
| demo_128 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_186 | unknown_crop | fail | router | kiwi / unknown /  | Part abstained for crop=kiwi: no compatible parts configured for crop (kiwi) |
| demo_193 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_033 | router_uncertain | fail | router | tomato / unknown /  | Part abstained for crop=tomato: confidence (0.2868) < threshold (0.4000); margin (-0.1894) < threshold (0.1000) |
| demo_068 | router_uncertain | fail | router | tomato / leaf /  | Router result is not eligible for adapter prediction. |
| demo_165 | unknown_crop | fail | router | plum / unknown /  | Part abstained for crop=plum: no compatible parts configured for crop (plum) |
| demo_498 | unknown_crop | fail | router | potato / leaf /  | Router crop 'potato' is outside the final demo supported crop set. |
