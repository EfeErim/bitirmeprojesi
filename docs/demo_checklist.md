# Demo Checklist

Last updated: 2026-06-17

Use this file for M1 and M2 execution. The goal is to prove that Colab Notebook 8 can handle expected user-like plant photos without code edits during the demo.

## Demo Surface

- Primary surface: `colab_notebooks/8_auto_router_adapter_prediction.ipynb`
- Exact M2 run path:
  1. Open Notebook 8 in Colab.
  2. Select a GPU runtime and make sure Hugging Face/SAM3 access is available.
  3. Leave `M2_RUN_FULL_DEMO = True`, `M2_DEMO_LIMIT = None`, `M2_BATCH_SIZE = 4`, `M2_ADAPTER_BATCH_SIZE = 8`, and `M2_DEMO_MANIFEST = 'docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv'`.
  4. Leave `M2_AUTO_PUSH_RESULTS = True` and `M2_AUTO_DISCONNECT_RUNTIME = True` when `GH_TOKEN` or `GITHUB_TOKEN` is available in Colab secrets.
  5. Run all cells. The single-image cell is skipped by default, and the final M2 cell runs the saved 512-image manifest.
  6. Read `.runtime_tmp/m2_demo_checklist_run.json` and `.runtime_tmp/m2_demo_checklist_run.md` for the local runtime copy.
  7. The notebook also copies the result into `docs/demo_results/m2/<timestamp>/`, commits/pushes that folder, and disconnects the Colab runtime after the report is written and the push succeeds.
- Optional single-image path:
  set `RUN_SINGLE_IMAGE_DEMO = True`, set `IMAGE_PATH` or upload one image, keep `RETURN_OOD = True`, then run the router/adapter cells.
- Optional CLI smoke path for local debugging only:
  `.\scripts\python.cmd -m src.app.cli inference <image> --config-env colab --adapter-root runs --device cuda`
- Fallback evidence: captured notebook outputs or screenshots stored under `.runtime_tmp/final_demo_fallbacks/`, then referenced from the final presentation or handoff guide.
- Decision policy: correct disease predictions and correct abstentions both count as useful behavior; wrong confident disease labels are failures.

## Target Coverage

The final target set is eight crop/part surfaces:

| Target | M1 inventory status | Adapter source | Notes |
|---|---|---|---|
| tomato__fruit | supported_candidate | `runs/**/tomato/fruit/continual_sd_lora_adapter` | Prior Notebook 16 accuracy is acceptable; final label still depends on M2 checklist run. |
| tomato__leaf | supported_candidate | `runs/**/tomato/leaf/continual_sd_lora_adapter` | Large operational surface; review-gate misses remain a known limitation. |
| strawberry__fruit | low_confidence_candidate | `runs/**/strawberry/fruit/continual_sd_lora_adapter` | Prior Notebook 16 report marks this as the main adapter/data outlier. |
| strawberry__leaf | supported_candidate | `runs/**/strawberry/leaf/continual_sd_lora_adapter` | Prior Notebook 16 accuracy is strong. |
| grape__fruit | supported_candidate | `runs/**/grape/fruit/continual_sd_lora_adapter` | Low error count in prior Notebook 16 report. |
| grape__leaf | supported_candidate | `runs/**/grape/leaf/continual_sd_lora_adapter` | Good demo candidate; calibration remains report-only. |
| apricot__fruit | supported_candidate | `runs/**/apricot/fruit/continual_sd_lora_adapter` | Strong prior Notebook 16 accuracy. |
| apricot__leaf | low_confidence_candidate | `runs/**/apricot/leaf/continual_sd_lora_adapter` | Moderate prior Notebook 16 accuracy and review burden. |

## Image Set Requirements

Target size: at least 500 rows unless runtime or asset access blocks it, because every supported disease class must have at least 10 trial images.

Minimum composition:

| Bucket | Target count | Purpose |
|---|---:|---|
| Supported known disease, clear photo | 48-72 | Prove expected happy path. |
| Supported known disease, difficult/user-like photo | 24-36 | Prove robustness to phone/internet variation. |
| Supported crop/part but unknown or unsafe disease | 12-24 | Prove unknown/OOD/review behavior. |
| Unsupported crop or unsupported part | 12-18 | Prove router/runtime abstention. |
| Non-plant or irrelevant image | 6-12 | Prove input safety or documented limitation. |
| Missing/blocked dependency fallback | 3-6 | Prove presentation continuity when external access fails. |

Image sources should include internet images, phone-captured images, and random/user-like photos. Do not use only clean training-style images.

The expanded internet image set is disease-focused and stored outside git under `.runtime_tmp/final_demo_images/internet_expansion/`. It is indexed by `.runtime_tmp/m2_internet_image_set_manifest.csv`, with disease/condition query metadata such as anthracnose, blight, botrytis, mildew, leaf scorch, plum pox, and shot-hole/Wilsonomyces. Use it with:

`.\scripts\python.cmd scripts\run_demo_checklist.py --extra-manifest .runtime_tmp\m2_internet_image_set_manifest.csv --extra-manifest .runtime_tmp\m2_supported_disease_coverage_manifest.csv --device cuda --adapter-root runs`

The supported-disease coverage manifest is generated from `data/prepared_runtime_datasets/*/{test,val,continual,train}`, excludes healthy classes, and guarantees that every supported disease class appears at least 10 times for every adapter.

The self-contained saved image package is under `docs/demo_assets/m2_full_image_set/`. It contains 512 copied images plus a runnable manifest:

`.\scripts\python.cmd scripts\run_demo_checklist.py --no-checklist --extra-manifest docs\demo_assets\m2_full_image_set\manifests\m2_full_image_set_run_manifest.csv --device cuda --adapter-root runs --batch-size 4 --adapter-batch-size 8`

The latest Colab M2 runs wrote reports with `M2_BATCH_SIZE = 4`, including the full 512-image `20260622T102452Z` result. Keep router batch size 4 for the next official Notebook 8 rerun, keep any smoke limit removed (`M2_DEMO_LIMIT = None`), and use `M2_ADAPTER_BATCH_SIZE = 8` so same-target adapter predictions can run in batches. The runner falls back to the older per-row adapter path if adapter batching is unavailable or unsafe.

## User Photo Guidance To Show

Tell users to provide:

- one main plant or plant part in the image
- visible leaf or fruit surface
- reasonable lighting and focus
- minimal extreme blur, heavy occlusion, or multi-plant clutter
- one of the supported crops when they expect a disease answer: tomato, strawberry, grape, or apricot

Tell users that unsupported crops, unclear parts, non-plant images, and diseases outside the supported class set may return unknown/review instead of a disease.

## Result Log Template

Use one row per image.

| image_id | source | expected_target | expected_behavior | actual_status | predicted_crop | predicted_part | predicted_disease | confidence_or_ood | pass_fail | failure_bucket | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| demo_xxx | internet | tomato__leaf | known disease |  |  |  |  |  |  |  |  |

Allowed `failure_bucket` values:

- `router`
- `adapter_loading`
- `disease_prediction`
- `ood_unknown`
- `input_guard`
- `dependency_access`
- `notebook_runtime`
- `asset_missing`
- `documentation`

## M1 Candidate Checklist

Fill `actual_*`, `pass_fail`, and `failure_bucket` during M2. The `source` column can point to a local test-pool class folder or to a staged external image under `.runtime_tmp/final_demo_images/`.

| image_id | source | expected_target | expected_behavior | actual_status | predicted_crop | predicted_part | predicted_disease | confidence_or_ood | pass_fail | failure_bucket | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| demo_001 | local_test_pool:data/prepared_runtime_datasets/tomato__fruit/test/domates_antraknoz_meyve | tomato__fruit | known disease: tomato anthracnose fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_002 | local_test_pool:data/prepared_runtime_datasets/tomato__fruit/test/domates_bacterial_spot_and_speck_meyve | tomato__fruit | known disease: tomato bacterial spot/speck fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_003 | local_test_pool:data/prepared_runtime_datasets/tomato__fruit/test/domates_late_blight_meyve | tomato__fruit | known disease: tomato late blight fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | difficult supported case |
| demo_004 | local_test_pool:data/prepared_runtime_datasets/tomato__fruit/test/domates_saglikli_meyve | tomato__fruit | known class: healthy tomato fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_005 | local_test_pool:data/prepared_runtime_datasets/tomato__leaf/test/domates_bacterial_spot_and_speck_yaprak | tomato__leaf | known disease: tomato bacterial spot/speck leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_006 | local_test_pool:data/prepared_runtime_datasets/tomato__leaf/test/domates_early_blight_yaprak | tomato__leaf | known disease: tomato early blight leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_007 | local_test_pool:data/prepared_runtime_datasets/tomato__leaf/test/domates_late_blight_yaprak | tomato__leaf | known disease: tomato late blight leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | difficult supported case |
| demo_008 | local_test_pool:data/prepared_runtime_datasets/tomato__leaf/test/domates_septoria_leaf_spot_yaprak | tomato__leaf | known disease: tomato septoria leaf spot; answer or review if evidence is unsafe |  |  |  |  |  |  |  | difficult supported case |
| demo_009 | local_test_pool:data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_anthracnose_fruit | strawberry__fruit | known disease: strawberry anthracnose fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | low-confidence target |
| demo_010 | local_test_pool:data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_gray_mold_fruit | strawberry__fruit | known disease: strawberry gray mold fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | low-confidence target |
| demo_011 | local_test_pool:data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_healthy_fruit | strawberry__fruit | known class: healthy strawberry fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | known confusion with unripe fruit |
| demo_012 | local_test_pool:data/prepared_runtime_datasets/strawberry__fruit/test/strawberry_powdery_mildew_fruit | strawberry__fruit | known disease: strawberry powdery mildew fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | low-confidence target |
| demo_013 | local_test_pool:data/prepared_runtime_datasets/strawberry__leaf/test/healthy | strawberry__leaf | known class: healthy strawberry leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_014 | local_test_pool:data/prepared_runtime_datasets/strawberry__leaf/test/strawberry_leaf_scorch_leaf | strawberry__leaf | known disease: strawberry leaf scorch; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_015 | local_test_pool:data/prepared_runtime_datasets/strawberry__leaf/test/strawberry_leaf_spot_leaf | strawberry__leaf | known disease: strawberry leaf spot; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_016 | local_test_pool:data/prepared_runtime_datasets/strawberry__leaf/test/strawberry_powdery_mildew_leaf | strawberry__leaf | known disease: strawberry powdery mildew leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_017 | local_test_pool:data/prepared_runtime_datasets/grape__fruit/test/uzum_antraknoz_meyve | grape__fruit | known disease: grape anthracnose fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_018 | local_test_pool:data/prepared_runtime_datasets/grape__fruit/test/uzum_botrytis_cinerea_meyve | grape__fruit | known disease: grape botrytis fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_019 | local_test_pool:data/prepared_runtime_datasets/grape__fruit/test/uzum_kulleme_meyve | grape__fruit | known disease: grape powdery mildew fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | difficult supported case |
| demo_020 | local_test_pool:data/prepared_runtime_datasets/grape__fruit/test/uzum_saglikli_meyve | grape__fruit | known class: healthy grape fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_021 | local_test_pool:data/prepared_runtime_datasets/grape__leaf/test/uzum_antraknoz_yaprak | grape__leaf | known disease: grape anthracnose leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_022 | local_test_pool:data/prepared_runtime_datasets/grape__leaf/test/uzum_kav_esca_yaprak | grape__leaf | known disease: grape esca leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_023 | local_test_pool:data/prepared_runtime_datasets/grape__leaf/test/uzum_mildiyo_yaprak | grape__leaf | known disease: grape downy mildew leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | difficult supported case |
| demo_024 | local_test_pool:data/prepared_runtime_datasets/grape__leaf/test/uzum_saglikli_yaprak | grape__leaf | known class: healthy grape leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_025 | local_test_pool:data/prepared_runtime_datasets/apricot__fruit/test/kayisida_cicek_monilyasi_meyve_40 | apricot__fruit | known disease: apricot blossom monilia fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_026 | local_test_pool:data/prepared_runtime_datasets/apricot__fruit/test/kayisida_sarka_virusu_meyve_230 | apricot__fruit | known disease: apricot sharka virus fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_027 | local_test_pool:data/prepared_runtime_datasets/apricot__fruit/test/kayisida_seftali_karalekesi_meyve_232 | apricot__fruit | known disease: apricot peach scab fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | difficult supported case |
| demo_028 | local_test_pool:data/prepared_runtime_datasets/apricot__fruit/test/kayisi_saglikli_meyve_800 | apricot__fruit | known class: healthy apricot fruit; answer or review if evidence is unsafe |  |  |  |  |  |  |  | clear supported case |
| demo_029 | local_test_pool:data/prepared_runtime_datasets/apricot__leaf/test/kayisi_saglikli_yaprak_302 | apricot__leaf | known class: healthy apricot leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | low-confidence target |
| demo_030 | local_test_pool:data/prepared_runtime_datasets/apricot__leaf/test/kayisi_sarka_virusu_yaprak_206 | apricot__leaf | known disease: apricot sharka virus leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | low-confidence target |
| demo_031 | local_test_pool:data/prepared_runtime_datasets/apricot__leaf/test/kayisi_yaprak_delen_cil_hastaligi_yaprak_300 | apricot__leaf | known disease: apricot shot-hole leaf; answer or review if evidence is unsafe |  |  |  |  |  |  |  | low-confidence target |
| demo_032 | staged_phone:.runtime_tmp/final_demo_images/apricot_leaf_user_like_01.jpg | apricot__leaf | supported crop/part but difficult user-like image; answer or review, no crash |  |  |  |  |  |  |  | collect before M2 |
| demo_033 | staged_external:.runtime_tmp/final_demo_images/tomato_leaf_unknown_damage_01.jpg | tomato__leaf | supported crop/part with unknown or unsafe disease; unknown, OOD, or review expected |  |  |  |  |  |  |  | collect before M2 |
| demo_034 | staged_external:.runtime_tmp/final_demo_images/tomato_fruit_unknown_damage_01.jpg | tomato__fruit | supported crop/part with unknown or unsafe disease; unknown, OOD, or review expected |  |  |  |  |  |  |  | collect before M2 |
| demo_035 | staged_external:.runtime_tmp/final_demo_images/strawberry_fruit_unripe_or_ambiguous_01.jpg | strawberry__fruit | supported crop/part but ambiguous fruit state; review or low confidence expected |  |  |  |  |  |  |  | collect before M2 |
| demo_036 | staged_external:.runtime_tmp/final_demo_images/grape_leaf_field_blur_01.jpg | grape__leaf | supported crop/part but difficult field image; answer or review, no crash |  |  |  |  |  |  |  | collect before M2 |
| demo_037 | staged_external:.runtime_tmp/final_demo_images/apricot_leaf_mixed_symptoms_01.jpg | apricot__leaf | supported crop/part with uncertain symptoms; review or low confidence expected |  |  |  |  |  |  |  | collect before M2 |
| demo_038 | staged_external:.runtime_tmp/final_demo_images/strawberry_leaf_unknown_spot_01.jpg | strawberry__leaf | supported crop/part with unknown or unsafe disease; unknown, OOD, or review expected |  |  |  |  |  |  |  | collect before M2 |
| demo_039 | staged_external:.runtime_tmp/final_demo_images/grape_fruit_unknown_rot_01.jpg | grape__fruit | supported crop/part with unknown or unsafe disease; unknown, OOD, or review expected |  |  |  |  |  |  |  | collect before M2 |
| demo_040 | staged_phone:.runtime_tmp/final_demo_images/tomato_leaf_phone_clutter_01.jpg | tomato__leaf | supported crop/part but cluttered user-like image; answer, review, or router_uncertain expected |  |  |  |  |  |  |  | collect before M2 |
| demo_041 | staged_external:.runtime_tmp/final_demo_images/apple_leaf_unsupported_01.jpg | unknown_crop | unsupported crop; unknown_crop or router_uncertain expected, no disease label |  |  |  |  |  |  |  | collect before M2 |
| demo_042 | staged_external:.runtime_tmp/final_demo_images/pepper_fruit_unsupported_01.jpg | unknown_crop | unsupported crop; unknown_crop or router_uncertain expected, no disease label |  |  |  |  |  |  |  | collect before M2 |
| demo_043 | staged_external:.runtime_tmp/final_demo_images/tomato_flower_unsupported_part_01.jpg | tomato__unknown_part | supported crop with unsupported part; router_uncertain or adapter_unavailable expected |  |  |  |  |  |  |  | collect before M2 |
| demo_044 | staged_external:.runtime_tmp/final_demo_images/grape_stem_unsupported_part_01.jpg | grape__unknown_part | supported crop with unsupported part; router_uncertain or adapter_unavailable expected |  |  |  |  |  |  |  | collect before M2 |
| demo_045 | staged_external:.runtime_tmp/final_demo_images/soil_or_pot_non_plant_01.jpg | non_plant | non-plant or irrelevant input; non_plant_rejected if guard is enabled, otherwise documented fallback |  |  |  |  |  |  |  | collect before M2 |
| demo_046 | staged_external:.runtime_tmp/final_demo_images/tool_or_table_non_plant_01.jpg | non_plant | non-plant or irrelevant input; non_plant_rejected if guard is enabled, otherwise documented fallback |  |  |  |  |  |  |  | collect before M2 |
| demo_047 | fallback_capture:.runtime_tmp/final_demo_fallbacks/success_case_notebook8.png | tomato__leaf | fallback screenshot/output for one successful Notebook 8 run |  |  |  |  |  |  |  | capture during M2 |
| demo_048 | fallback_capture:.runtime_tmp/final_demo_fallbacks/unknown_case_notebook8.png | unknown_crop | fallback screenshot/output for one unknown/review Notebook 8 run |  |  |  |  |  |  |  | capture during M2 |

## Pass Criteria

- No crashes or manual code edits across the checklist.
- Target 90% correct disease predictions on answerable supported examples.
- Minimum acceptable disease threshold is 80% if limitations are documented.
- Unsupported, non-plant, missing-adapter, uncertain, or OOD cases do not force a known disease label.
- Final report includes answered count, abstained/review count, failed count, and per-target support status.

## M1 Tasks

- [x] Choose the exact Colab Notebook 8 run path.
- [x] Collect the candidate image list.
- [x] Fill expected target and expected behavior before running inference.
- [x] Confirm adapter roots for all eight target surfaces.
- [x] Mark each target surface as `supported`, `low_confidence`, or `experimental` candidate before M2 evidence.

## M2 Tasks

- Run the full checklist.
- Record failures literally by bucket.
- Fix narrow blockers only.
- Re-run failed cases after each fix.
- Capture fallback outputs/screenshots for the final presentation.
- Freeze the final demo set before presentation rehearsal.

## M2 Run Log

- 2026-06-16 local start command:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --device cpu --adapter-root runs --limit 1 --stop-on-dependency-blocker`
- Output report:
  `.runtime_tmp/m2_demo_checklist_run.json` and `.runtime_tmp/m2_demo_checklist_run.md`
- Result: stopped after `demo_001` because the router could not load gated `facebook/sam3` assets from Hugging Face. The failure bucket is `dependency_access`, not an adapter or checklist-data failure.
- Adapter-only smoke command with trusted expected targets was also blocked for the same reason because Notebook 8's helper initializes the router before adapter prediction:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --device cpu --adapter-root runs --only-local --limit 3 --trust-expected-target --output .runtime_tmp\m2_demo_checklist_trusted_smoke.json --markdown-output .runtime_tmp\m2_demo_checklist_trusted_smoke.md`
- Next action: rerun the full command in a Colab/runtime session with authenticated SAM3 access, then fill the checklist actual columns from the generated report.
- 2026-06-16 asset audit command:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --mode asset-audit --output .runtime_tmp\m2_demo_asset_audit.json --markdown-output .runtime_tmp\m2_demo_asset_audit.md`
- Asset audit result: 31/48 rows are file-ready. `demo_001` through `demo_031` now resolve locally, including Unicode Turkish class folders such as grape and apricot. Remaining 17 rows are expected staged assets or fallback captures that do not exist yet: `demo_032` through `demo_048`.
- Adapter-only smoke command:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --mode adapter-smoke --device cpu --adapter-root runs --only-local --limit 1 --output .runtime_tmp\m2_demo_adapter_smoke_sample.json --markdown-output .runtime_tmp\m2_demo_adapter_smoke_sample.md`
- Adapter-only smoke result: blocked in this CPU runtime by `Requested device 'cuda' but CUDA is not available.` Treat this as a local runtime limitation, not a final Colab/GPU demo result.

## M2 Items To Hand Off

- Provide or stage the 15 external/phone images for `demo_032` through `demo_046` under `.runtime_tmp/final_demo_images/`.
- After authenticated SAM3/Hugging Face access is available, rerun the official command without `--limit`:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --device cuda --adapter-root runs`
- Capture the two Notebook 8 fallback outputs/screenshots for `demo_047` and `demo_048` under `.runtime_tmp/final_demo_fallbacks/`.
- Fill the actual result columns only from the official Notebook 8/helper report, not from `asset-audit` or `adapter-smoke`.
- 2026-06-16 internet expansion command:
  `.\scripts\python.cmd scripts\build_m2_internet_image_set.py`
- Internet expansion result: 96/96 new disease-focused internet images downloaded as `demo_049` through `demo_144`, with source metadata and disease/condition query text in `.runtime_tmp/m2_internet_image_set_manifest.csv`.
- The original placeholder staged images `demo_032` through `demo_046` were also populated from the disease-focused internet set under `.runtime_tmp/final_demo_images/`.
- Expanded asset audit command:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --mode asset-audit --extra-manifest .runtime_tmp\m2_internet_image_set_manifest.csv --output .runtime_tmp\m2_demo_asset_audit_expanded.json --markdown-output .runtime_tmp\m2_demo_asset_audit_expanded.md`
- Expanded asset audit result: 144 total rows, 142 file-ready rows, and 2 expected missing fallback captures: `demo_047` and `demo_048`.
- 2026-06-17 supported-disease coverage command:
  `.\scripts\python.cmd scripts\build_m2_supported_disease_manifest.py`
- Supported-disease coverage result: 37 non-healthy supported disease classes across the eight adapters are indexed in `.runtime_tmp/m2_supported_disease_coverage_manifest.csv`, with 10 image rows per class.
- Full coverage asset audit command:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --mode asset-audit --extra-manifest .runtime_tmp\m2_internet_image_set_manifest.csv --extra-manifest .runtime_tmp\m2_supported_disease_coverage_manifest.csv --output .runtime_tmp\m2_demo_asset_audit_full_coverage.json --markdown-output .runtime_tmp\m2_demo_asset_audit_full_coverage.md`
- Full coverage asset audit result after the 10-per-class expansion: 514 total rows, 512 file-ready rows, and 2 expected missing fallback captures: `demo_047` and `demo_048`.
- 2026-06-17 saved image package: 512 asset-ready images were copied to `docs/demo_assets/m2_full_image_set/images/`. The runnable saved manifest is `docs/demo_assets/m2_full_image_set/manifests/m2_full_image_set_run_manifest.csv`.
- Saved package asset audit command:
  `.\scripts\python.cmd scripts\run_demo_checklist.py --no-checklist --mode asset-audit --extra-manifest docs\demo_assets\m2_full_image_set\manifests\m2_full_image_set_run_manifest.csv --output .runtime_tmp\m2_saved_image_set_asset_audit.json --markdown-output .runtime_tmp\m2_saved_image_set_asset_audit.md`
- Saved package asset audit result: 512/512 rows are file-ready.
