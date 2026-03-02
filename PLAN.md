# AADS v6 Overhaul Plan: SD-LoRA-Only Continual Learning Stack

## Summary
This overhaul replaces the current 3-phase `DoRA -> SD-LoRA -> CoNeC-LoRA` flow with a single-engine, rehearsal-free continual learning architecture based on **SD-LoRA only**, **frozen DINOv3-giant**, **hybrid INT8 quantization**, **4-stage multi-scale fusion**, and **transformer all-linear adapter targeting**.  
Scope is repo-wide, with legacy v5.5 materials archived and redirected.

## Public APIs, Interfaces, and Type Changes
1. Remove runtime/public use of:
   - `ColabPhase1Trainer` (`src/training/colab_phase1_training.py`)
   - `ColabPhase3Trainer`, `CoNeCConfig` (`src/training/colab_phase3_conec_lora.py`, `src/training/phase3_*`)
   - `phase1_initialize()` and `phase3_fortify()` lifecycle semantics in adapter docs/config contracts.
2. Introduce new primary training surface:
   - `src/training/continual_sd_lora.py`
   - `ContinualSDLoRAConfig`
   - `ContinualSDLoRATrainer`
3. Replace `training.phase1/phase2/phase3` config contract with `training.continual`:
   - `backbone.model_name` (default `facebook/dinov3-giant`)
   - `quantization.mode` (`int8_hybrid`)
   - `adapter.target_modules_strategy` (`all_linear_transformer`)
   - `fusion.layers` (`[2,5,8,11]`)
   - `ood` (new v6 OOD config surface)
4. Adapter metadata contract (`adapter_meta.json`) moves to v6 schema:
   - Required keys: `schema_version`, `engine`, `backbone`, `quantization`, `fusion`, `class_to_idx`, `ood_calibration`, `target_modules_resolved`.
   - Remove phase-indexed fields (`current_phase`, phase-specific retention fields).
5. Notebook/script entrypoints move to continual naming:
   - Historical note: prior v6 Colab notebooks are now archived under `colab_notebooks/archive/v6_superseded_2026-03-02/`.
   - Remove runtime dependency on `2_phase1_training.ipynb` and `4_phase3_training.ipynb`.

## Implementation Plan With Subtasks

### Phase A: Contract-First Cutover
1. `A1` Create v6 config schema and validator rules for `training.continual` in `src/core/schemas.py`.
2. `A2` Update `config/colab.json`, `config/base.json`, `specs/adapter-spec.json` to v6 fields only.
3. `A3` Add hard validation guard that rejects 4-bit settings (`load_in_4bit`, NF4/QLoRA flags) in config and runtime parser.
4. `A4` Update `src/core/config_manager.py` access paths to v6 training keys and remove phase key assumptions.

### Phase B: Core Training Engine Replacement
1. `B1` Implement `ContinualSDLoRAConfig` and `ContinualSDLoRATrainer` in `src/training/continual_sd_lora.py`.
2. `B2` Implement `all_linear_transformer` module resolver: target all `nn.Linear` in transformer blocks, exclude classifier/router heads.
3. `B3` Remove Stable Diffusion-specific semantics from SD-LoRA surfaces and rename docs/comments/types to continual classification semantics.
4. `B4` Delete or fully retire runtime references to `phase1` and `phase3` trainer modules (clean break).

### Phase C: Hybrid INT8 Quantization Layer
1. `C1` Add quantization utility module (`src/training/quantization.py`) with hybrid INT8 load path using `bitsandbytes` outlier-aware behavior.
2. `C2` Integrate INT8 backbone loading into continual trainer and adapter loading; keep adapter/classifier training path numerically stable (mixed precision where needed).
3. `C3` Add runtime safeguards: if INT8 backend unavailable, fail with explicit actionable error (no silent fallback to 4-bit).
4. `C4` Persist quantization metadata in adapter manifests and checkpoint outputs.

### Phase D: Multi-Scale Feature Fusion (MAA-like)
1. `D1` Add `src/adapter/multi_scale_fusion.py` implementing 4-stage token fusion from DINOv3 blocks `[2,5,8,11]`.
2. `D2` Add fusion head integration in continual trainer and adapter inference path.
3. `D3` Ensure fused representation is used consistently for classification and OOD feature extraction.
4. `D4` Add config knobs for fusion dim/dropout/gating while keeping default enabled.

### Phase E: Adapter Lifecycle Redesign
1. `E1` Refactor `src/adapter/independent_crop_adapter.py` to v6 lifecycle:
   - `initialize_engine(...)`
   - `add_classes(...)`
   - `train_increment(...)`
   - `save_adapter(...)` / `load_adapter(...)` with v6 metadata.
2. `E2` Remove phase numbering from adapter state machine and public summaries.
3. `E3` Update adapter save paths/artifact names to continual naming (`continual_sd_lora_adapter`, non-phase directories).
4. `E4` Remove DoRA/Conec references and assumptions from adapter logic and logs.

### Phase F: OOD Redesign (In-Scope)
1. `F1` Implement new OOD module (`src/ood/continual_ood.py`) using fused features and ensemble score:
   - `score = 0.6 * mahalanobis_z + 0.4 * energy_z`.
2. `F2` Replace phase-specific prototype/OOD hooks with continual calibration hooks after each class increment.
3. `F3` Update pipeline OOD response contract to include `ensemble_score`, `class_threshold`, `is_ood`, and calibration version.
4. `F4` Wire new OOD config under `training.continual.ood` and remove CoNeC-linked OOD coupling.

### Phase G: Router/Pipeline Alignment
1. `G1` Set `SimpleCropRouter` default backbone to `facebook/dinov3-giant` and update related tests/docs.
2. `G2` Update `src/pipeline/independent_multi_crop_pipeline.py` to load v6 adapter artifacts and consume new OOD output keys.
3. `G3` Keep VLMPipeline routing logic intact unless required by interface break; only adjust integration contracts.

### Phase H: Notebook and Script Migration
1. `H1` Historical task (superseded): update archived `0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb` flow assumptions to continual stages:
   - Data prep -> Continual training -> Validation -> Monitoring.
2. `H2` Historical task (superseded): add/replace training notebook with `2_continual_sd_lora_training.ipynb`.
3. `H3` Update `scripts/colab_auto_orchestrator.py` stage names, notebook paths, checkpoint keys, and logs.
4. `H4` Remove runtime logic that toggles phase1/phase2/phase3 independently.

### Phase I: Test Matrix Rebuild
1. `I1` Replace phase-specific unit tests with v6 suites:
   - `tests/unit/training/test_continual_sd_lora.py`
   - `tests/unit/adapter/test_continual_adapter.py`
   - `tests/unit/ood/test_continual_ood.py`
   - `tests/unit/training/test_int8_quantization.py`
2. `I2` Rewrite `tests/colab/test_smoke_training.py` for continual-only entrypoints.
3. `I3` Rewrite relevant integration tests (`tests/integration/test_colab_integration.py`) to v6 config and lifecycle.
4. `I4` Add static guard tests ensuring no code/config paths contain DoRA, CoNeC, or 4-bit usage.

### Phase J: Repo-Wide Cleanup and Archival
1. `J1` Archive legacy v5.5 docs/notebooks/reports into versioned archive paths:
   - `docs/archive/v5_legacy/`
   - `colab_notebooks/archive/v5_legacy/`
   - optional `plans/archive/v5_legacy/`.
2. `J2` Keep redirect stubs at key old entrypoints (README/docs index/notebook README) pointing to v6 docs.
3. `J3` Update all user-facing docs (`README.md`, `docs/architecture/overview.md`, `docs/development/development-setup.md`, guides) to v6 terminology and flow.
4. `J4` Remove stale phase names from script docs and config examples.

## Dependency and Execution Order
1. Execute `A -> B -> C -> D -> E -> F` sequentially (contract and core runtime first).
2. Execute `G` after `E/F` (pipeline depends on new adapter/OOD contracts).
3. Execute `H` after `A/B/E` (notebooks depend on finalized interfaces).
4. Execute `I` after `B..H` (tests against final behavior).
5. Execute `J` last, except archive scaffolding can start early.

## Validation Gates and Test Scenarios
1. Static architecture gate:
   - `rg -n "DoRA|CoNeC|phase1|phase3|load_in_4bit|nf4|QLoRA" src config colab_notebooks scripts docs`
   - Expect no runtime references outside archived v5 paths.
2. Contract gate:
   - Config load/validate must pass with new `training.continual` and fail on 4-bit keys.
3. Unit behavior gate:
   - Continual trainer initializes DINOv3-giant frozen backbone + all-linear adapters.
   - Target module resolver includes attention and MLP linear layers, excludes classifier/router heads.
   - Fusion module returns stable fused embeddings across batch sizes.
4. Quantization gate:
   - INT8 path active in trainer metadata.
   - Memory footprint improves vs FP16 baseline under same batch/model settings.
5. OOD gate:
   - New ensemble OOD scores generated and calibrated.
   - AUROC/FPR targets tracked on validation sets.
6. Integration gate:
   - End-to-end Colab continual run produces v6 artifacts, manifests, and checkpoint logs.
   - Pipeline consumes v6 adapters and returns valid diagnosis + redesigned OOD payload.
7. Repo hygiene gate:
   - Markdown link checker passes after archive/redirect operations.
   - CI suites run with updated suite definitions and no phase-specific failures.

## Assumptions and Defaults (Locked)
1. Clean break v6: no runtime compatibility wrappers for old Phase 1/3 APIs.
2. Backbone default is always `facebook/dinov3-giant`.
3. Quantization rollout is hybrid INT8 first; 4-bit paths are prohibited.
4. Multi-scale fusion is 4-stage token fusion.
5. Adapter targeting is transformer all-linear only (excluding classifier/router heads).
6. OOD is redesigned in this same overhaul, not deferred.
7. Repo-wide cleanup uses archive-and-redirect policy for v5.5 legacy materials.
