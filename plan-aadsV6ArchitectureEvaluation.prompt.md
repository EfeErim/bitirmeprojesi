# Architecture Evaluation & Subtask Plan: AADS v6 Overhaul

## TL;DR

AADS v6 is a **plant disease classification system using rehearsal-free continual learning** (SD-LoRA + frozen DINOv3 + INT8 quantization). The v6 overhaul from v5.5 is architecturally well-designed across 10 phases (A–J). The core source modules are implemented and pass all audit gates (188/188 files pass, 0 defects per the V6_FULL_REPO_AUDIT.md). However, there are **concrete issues** in config drift, stale metadata, duplicated code, display bugs, and incomplete cross-references. Below are all subtasks organized by domain.

---

## TRACK 1 — Configuration & Contract Hygiene

### S1.1 — Fix `setup.py` staleness (CRITICAL)

- `setup.py` declares `name="aads-ulora-v5.5"`, `version="5.5.0"`, package `aads_ulora_v55`
- Entry point `aads-train` → `src.training.phase1_training:main` — this module no longer exists in v6
- Update name to `aads-v6`, version to `6.0.0`, fix/remove stale entry points, align `install_requires` version pins
- **Also fix**: `transformers` version is `==4.56.0` in setup.py vs `>=4.41.0` in requirements.txt vs `>=4.50.0,<5.0.0` in requirements_colab.txt — unify to a single compatible range

### S1.2 — Resolve dual OOD config surface

- `config/base.json` and `config/colab.json` both have a top-level `ood` key (Mahalanobis-only, threshold 0.95) AND `training.continual.ood` (ensemble, threshold_factor 2)
- Determine which is authoritative; remove or alias the other; update `config_manager.py` accordingly

### S1.3 — Fix phantom config references in architecture docs

- `docs/architecture/overview.md` references `config/training_config.json` and `config/router_config.json` — these files do not exist (only `base.json` and `colab.json` do)
- Update doc to reference the actual config files

### S1.4 — Reconcile `perf_guardrails_phase5.json` with v6 pipeline

- `config/perf_guardrails_phase5.json` defines CPU benchmark latency limits but the relationship to v6's new pipeline path (DINOv3 backbone, INT8, multi-scale fusion) is undocumented
- Document whether these thresholds still apply or need recalibration for the v6 engine

---

## TRACK 2 — Training & Quantization Engine

### S2.1 — Validate `ContinualSDLoRATrainer` end-to-end (Phase B)

- `src/training/continual_sd_lora.py` (523 lines) — verify `from_training_config()`, `initialize_engine()`, `add_classes()`, `train_increment()`, `predict()` all work with a real DINOv3-like backbone
- Cross-check with `specs/adapter-spec.json` targets: accuracy ≥ 0.93, OOD AUROC ≥ 0.92, FPR ≤ 0.05

### S2.2 — Harden INT8 quantization path (Phase C)

- `src/training/quantization.py` (147 lines) — the `load_hybrid_int8_backbone()` has a fallback path when backend is missing
- Verify fail-loud behavior when `fallback_to_non_quantized=False` in strict production mode
- Ensure `find_prohibited_4bit_flags()` catches all config surfaces (base.json, colab.json, notebook-injected configs)

### S2.3 — Verify multi-scale fusion correctness (Phase D)

- `src/adapter/multi_scale_fusion.py` (88 lines) — `MultiScaleFeatureFusion` uses layers `[2,5,8,11]` with softmax gating
- Test coverage exists in `tests/unit/adapter/test_multi_scale_fusion.py` (only 24 lines, 3 tests) — expand coverage for edge cases (wrong layer count, mismatched dimensions, gradient flow)

---

## TRACK 3 — Adapter Lifecycle

### S3.1 — Audit `IndependentCropAdapter` lifecycle completeness (Phase E)

- `src/adapter/independent_crop_adapter.py` (260 lines) — verify full `initialize_engine → add_classes → train_increment → calibrate_ood → save_adapter → load_adapter` cycle
- Cross-check adapter metadata schema (required keys: `schema_version`, `engine`, `backbone`, `quantization`, `fusion`, `class_to_idx`, `ood_calibration`, `target_modules_resolved`) per `specs/adapter-spec.json`

### S3.2 — Remove phase-numbering remnants from training lifecycle skill

- The `aads-training-lifecycle` skill description still references "phase1/phase2/phase3" invariants — contradicts v6's clean-break policy
- Update `skills/aads-training-lifecycle/SKILL.md` to reflect v6 single-engine semantics

---

## TRACK 4 — OOD Detection

### S4.1 — Validate ensemble OOD scoring (Phase F)

- `src/ood/continual_ood.py` (154 lines) — weighted ensemble: 0.6 × mahalanobis_z + 0.4 × energy_z
- Verify per-class threshold management via `src/ood/dynamic_thresholds.py` (753 lines) and calibration versioning
- Tests exist in `tests/unit/ood/test_continual_ood.py` (only 34 lines) — expand

### S4.2 — Consolidate OOD test files

- Three separate OOD test files with overlapping scope:
  - `tests/unit/ood/test_ood_comprehensive.py` (737 lines)
  - `tests/unit/ood/test_dynamic_thresholds_improved.py` (190 lines, script-style)
  - `tests/unit/test_ood.py` (144 lines)
- Consolidate into a single structured test file under `tests/unit/ood/`

---

## TRACK 5 — Router & Pipeline

### S5.1 — Verify `VLMPipeline` as sole router (Phase G)

- `src/router/vlm_pipeline.py` (2,023 lines) — largest file in the project; verify `SimpleCropRouter` is fully archived
- Confirm policy graph stage order: SAM3 ROI Filter → ROI Classification → Open-Set Gate → Postprocess → Best Detection
- Policy regression tests in `tests/unit/router/test_vlm_policy_stage_order.py` (432 lines) — review completeness

### S5.2 — Audit `IndependentMultiCropPipeline` v6 contract

- `src/pipeline/independent_multi_crop_pipeline.py` (760 lines) — verify result contract keys match `specs/pipeline-spec.json`: status, crop, part, diagnosis, confidence, ood_analysis, router_confidence, crop_confidence, cache_hit
- Verify error states: router_unavailable, adapter_unavailable, unknown_crop

### S5.3 — Audit ROI helpers for correctness

- `src/router/roi_pipeline.py` (371 lines) and `src/router/roi_helpers.py` (238 lines) — no dedicated unit tests found; add coverage for `suppress_overlapping_detections()`, `select_best_detection()`, `bbox_iou()`

---

## TRACK 6 — Colab Notebooks (Phase H)

### S6.1 — Extract shared notebook boilerplate to a module

- Both `1_crop_router_pipeline.ipynb` and `2_interactive_adapter_training.ipynb` have ~80 identical lines for `resolve_repo_root()` / `maybe_clone_repo()` / `is_repo_root()`
- Extract to a shared module (e.g., `scripts/colab_quick_setup.py` or `src/core/colab_contract.py`)

### S6.2 — Fix display bug in Notebook 2 Cell 6

- OOD calibration cell creates an `HTML` widget but never calls `display()` — calibration status is silently discarded
- Add `display(cal_html)` call

### S6.3 — Fix duplicate `normalize_class_name()` in Notebook 2

- Defined twice: once inside `prepare_runtime_dataset_layout()` and once at module scope in Cell 4
- Remove the inner duplicate

### S6.4 — Replace hardcoded class names with taxonomy lookup

- Notebook 2 (Cell 4) has hardcoded `expected_class_names_by_crop` dict for tomato/pepper/corn
- Should source from `config/plant_taxonomy.json` instead

### S6.5 — Add error handling to analysis cells in Notebook 1

- Cells 4–5 call `ROUTER.analyze_image()` and `pipeline.process_image()` without try/except
- Add user-friendly error messages for common failures (model not loaded, bad image format)

### S6.6 — Investigate `ROUTER.router_analyzer = None` override

- Notebook 1 Cell 5 explicitly sets `ROUTER.router_analyzer = None` before pipeline execution — determine if this is intentional or a workaround, and document

---

## TRACK 7 — Test Matrix (Phase I)

### S7.1 — Expand thin test files

- `tests/unit/adapter/test_multi_scale_fusion.py` — 24 lines (3 tests) for an 88-line module
- `tests/unit/ood/test_continual_ood.py` — 34 lines (2 tests) for a 154-line module
- `tests/unit/router/test_router_comprehensive.py` — 37 lines (2 tests) for a 2,023-line module
- Target: at minimum, test every public method of each class

### S7.2 — Add missing ROI module tests

- No dedicated tests for `src/router/roi_pipeline.py` (371 lines) or `src/router/roi_helpers.py` (238 lines)
- Create `tests/unit/router/test_roi_pipeline.py` and `tests/unit/router/test_roi_helpers.py`

### S7.3 — Add missing `diagnostic_scouting.py` tests

- `src/router/diagnostic_scouting.py` (117 lines) — `DiagnosticScoutingAnalyzer` class has no dedicated test file

### S7.4 — Add missing `policy_taxonomy_utils.py` tests

- `src/router/policy_taxonomy_utils.py` (182 lines) — helper functions (`deep_merge_dicts`, `build_policy_graph`, `resolve_requested_profile`, `load_taxonomy`, `load_crop_part_compatibility`) have no dedicated unit tests

### S7.5 — Consolidate verification scripts

- `tests/unit/utils/verify_optimizations.py` (212 lines) and `tests/unit/utils/verify_optimizations_simple.py` (157 lines) — script-style, not proper pytest; consolidate or convert
- `tests/unit/ood/test_dynamic_thresholds_improved.py` (190 lines), `tests/unit/utils/test_minimal_implementation.py` (94 lines) — also script-style, convert to proper pytest classes

### S7.6 — Validate test suite runner groups

- `scripts/run_test_suites.py` defines suite groups (quick, unit, colab, integration, all) — verify all test files are covered by at least one group

---

## TRACK 8 — Scripts Audit

### S8.1 — Audit Colab-specific scripts for v6 alignment

- 11 Colab scripts in `scripts/` — verify all reference v6 modules (not v5.5 phase-based training)
- Key files: `colab_auto_orchestrator.py`, `install_colab.py`, `download_data_colab.py`, `colab_setup_dependencies.py`

### S8.2 — Verify `colab_auto_orchestrator.py` notebook references

- `scripts/colab_auto_orchestrator.py` (231 lines) — confirm it references the current 2-notebook flow (not the archived 6-notebook flow)

### S8.3 — Validate `benchmark_router_phase5.py` against v6 pipeline

- `scripts/benchmark_router_phase5.py` (155 lines) — uses mocked model hooks; verify mocks match current `VLMPipeline` API surface

### S8.4 — Verify `evaluate_dataset_layout.py` integration with Notebook 2

- `scripts/evaluate_dataset_layout.py` is imported by Notebook 2 Cell 3; confirm API contract (`evaluate_layout()` return shape) is stable

---

## TRACK 9 — Documentation Sync

### S9.1 — Update architecture overview for actual config paths

- `docs/architecture/overview.md` references `config/training_config.json` and `config/router_config.json` — replace with `config/base.json` and `config/colab.json`

### S9.2 — Verify `REPO_FILE_RELATIONS.md` is current

- `docs/REPO_FILE_RELATIONS.md` — generated by `scripts/generate_repo_relationships.py` (1,090 lines); re-run to ensure import graph is current post-v6

### S9.3 — Update Colab training manual for current cell layout

- `docs/user_guide/colab_training_manual.md` — verify step descriptions match actual notebook cell contents

### S9.4 — Align `docs/guides/SEAMLESS_AUTOTRAIN_GUIDE.md` with notebook changes

- If S6.1–S6.6 modify notebooks, update the guide accordingly

---

## TRACK 10 — Repo Hygiene (Phase J)

### S10.1 — DINOv3 backbone availability verification (CRITICAL)

- The locked backbone `facebook/dinov3-vitl16-pretrain-lvd1689m` is referenced throughout — verify this is a real HuggingFace model ID or document the actual model to use

### S10.2 — Validate all markdown links

- Run `scripts/check_markdown_links.py --root .` and fix any broken links introduced by v6 changes

### S10.3 — Re-run full audit and verify 188/188 pass rate holds

- Re-run `scripts/audit_v6_repo.py` after any changes to confirm no regressions

---

## Subtask Summary Table

| Track | # | Description | Priority | Phase |
|-------|---|-------------|----------|-------|
| 1 Config | S1.1 | Fix stale `setup.py` | **Critical** | J |
| 1 Config | S1.2 | Resolve dual OOD config | High | A |
| 1 Config | S1.3 | Fix phantom config doc refs | Medium | J |
| 1 Config | S1.4 | Reconcile perf guardrails | Medium | G |
| 2 Training | S2.1 | Validate trainer E2E | High | B |
| 2 Training | S2.2 | Harden INT8 path | High | C |
| 2 Training | S2.3 | Expand fusion tests | Medium | D |
| 3 Adapter | S3.1 | Audit adapter lifecycle | High | E |
| 3 Adapter | S3.2 | Remove phase refs from skill | Medium | J |
| 4 OOD | S4.1 | Validate ensemble scoring | High | F |
| 4 OOD | S4.2 | Consolidate OOD tests | Medium | I |
| 5 Router | S5.1 | Verify VLM sole router | High | G |
| 5 Router | S5.2 | Audit pipeline contract | High | G |
| 5 Router | S5.3 | Audit ROI helpers | Medium | G |
| 6 Notebook | S6.1 | Extract shared boilerplate | Medium | H |
| 6 Notebook | S6.2 | Fix display bug NB2 Cell 6 | Low | H |
| 6 Notebook | S6.3 | Fix duplicate function NB2 | Low | H |
| 6 Notebook | S6.4 | Taxonomy lookup vs hardcode | Medium | H |
| 6 Notebook | S6.5 | Add error handling NB1 | Medium | H |
| 6 Notebook | S6.6 | Document router_analyzer override | Low | H |
| 7 Tests | S7.1 | Expand thin test files | High | I |
| 7 Tests | S7.2 | Add ROI module tests | High | I |
| 7 Tests | S7.3 | Add diagnostic_scouting tests | Medium | I |
| 7 Tests | S7.4 | Add policy_taxonomy tests | Medium | I |
| 7 Tests | S7.5 | Convert script-style tests | Medium | I |
| 7 Tests | S7.6 | Validate suite runner groups | Low | I |
| 8 Scripts | S8.1 | Audit Colab scripts for v6 | Medium | H |
| 8 Scripts | S8.2 | Verify orchestrator refs | Medium | H |
| 8 Scripts | S8.3 | Validate benchmark mocks | Medium | G |
| 8 Scripts | S8.4 | Verify dataset layout API | Low | H |
| 9 Docs | S9.1 | Fix arch overview config paths | Medium | J |
| 9 Docs | S9.2 | Regenerate REPO_FILE_RELATIONS | Low | J |
| 9 Docs | S9.3 | Update Colab training manual | Low | J |
| 9 Docs | S9.4 | Align SEAMLESS_AUTOTRAIN guide | Low | J |
| 10 Hygiene | S10.1 | Verify DINOv3 model ID | **Critical** | B |
| 10 Hygiene | S10.2 | Validate markdown links | Low | J |
| 10 Hygiene | S10.3 | Re-run full audit | Low | J |

## Recommended Execution Order

S1.1, S10.1 (critical blockers) → S1.2, S2.1–S2.3 (core engine) → S3.1, S4.1, S5.1–S5.2 (contract validation) → S6.1–S6.6, S8.1–S8.4 (notebooks/scripts) → S7.1–S7.6 (tests) → S9.1–S9.4, S10.2–S10.3 (docs/hygiene) → S1.3–S1.4, S3.2 (cleanup)
