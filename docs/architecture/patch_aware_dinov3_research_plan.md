# Patch-Aware DINOv3 Research Plan

## Summary

This document defines a research-branch implementation plan for two related changes:

1. Patch-aware dataset audit in Notebook 0 and `scripts.prepare_grouped_runtime_dataset`
2. A full local-global DINOv3 training head that uses register-aware token handling

This plan is intentionally non-default. The maintained repo behavior must remain unchanged unless the research path is explicitly enabled.

Default invariants:

- Notebook 0 keeps the current audit path when research flags are off.
- Training keeps `legacy_multiscale` as the default fusion mode.
- Existing adapters remain loadable without migration.

Current repo boundary:

- Cross-class exact-hash and identical-pHash conflict blocking already exists and must stay active in both legacy and research modes.
- The missing cross-class piece is representation-based candidate mining and patch-aware reranking.
- In research mode, global DINO cosine may be used to generate tractable candidate sets, but it must not be the final DINO decision signal.

## Implementation Changes

### 1. Patch-aware Notebook 0 audit

Implement a research-only audit mode in `scripts.prepare_grouped_runtime_dataset.py` and expose it from `colab_notebooks/0_grouped_dataset_preparation.ipynb`.

Research-mode behavior:

- Keep the current file scan, pHash, BioCLIP, family grouping, and runtime materialization flow.
- Extend DINOv3 extraction so each image yields:
  - one global DINO embedding
  - one patch-token tensor from `last_hidden_state`
  - register-token count from the loaded DINOv3 config
- Exclude CLS and register tokens from patch comparisons.
- Use patch-aware DINO as the authoritative DINO signal in research mode.
- Keep global DINO cosine as an auxiliary diagnostic artifact only.
- Treat patch-token tensors as intermediate audit data. Do not write full per-image patch tensors as tracked or guided artifacts by default.

DINOv3 token contract:

- This research path assumes a DINOv3 ViT-style backbone output with rank-3 `last_hidden_state` shaped `[B, T, D]`.
- Token splitting follows the Hugging Face DINOv3 ViT layout:
  - `last_hidden_state[:, 0, :]` is the CLS token
  - `last_hidden_state[:, 1:1 + num_register_tokens, :]` are register tokens
  - `last_hidden_state[:, 1 + num_register_tokens:, :]` are patch tokens
- `num_register_tokens` must come from `model.config.num_register_tokens`, defaulting to `0` only when the attribute is absent.
- If the loaded model exposes a ConvNeXT-style feature map or any non-rank-3 DINO output, research patch-aware mode should fail clearly instead of silently falling back to global pooling.

Patch-aware similarity definition:

- Normalize patch embeddings per image.
- Compute all-pairs cosine similarity between patch tokens of image A and image B.
- For each patch in A, keep the best matching patch in B; repeat from B to A.
- Apply top-k trimming to both directional match vectors by keeping the highest `PATCH_AWARE_TOPK` best-match scores per direction. If `PATCH_AWARE_TOPK` is larger than the available patch count, use all available patch matches.
- Define `patch_dino_score` as the mean of the trimmed directional means.

Research-mode decision policy:

- Same-class auto-merge requires `patch_dino_score >= PATCH_DINO_AUTO_MIN` and at least one stabilizer:
  - pHash inside the auto threshold, or
  - BioCLIP at or above the review threshold
- Same-class review uses `PATCH_DINO_REVIEW_MIN`
- Cross-class block uses `PATCH_DINO_CROSS_CLASS_BLOCK_MIN`
- Exact duplicates still block immediately

Same-class patch-aware candidate generation:

- Keep exact duplicate and pHash family behavior unchanged.
- Generate same-class representation candidates with the current DINO neighbor workflow or an equivalent global-DINO-first coarse index.
- Rerank those same-class candidates with `patch_dino_score`.
- Use `patch_dino_score`, not global DINO cosine, for research-mode same-class auto-merge and review decisions.
- BioCLIP remains a stabilizer for conservative auto-merge and review context, not the authoritative DINO signal.

Cross-class patch-aware candidate generation:

- Add an explicit representation-based cross-class candidate-mining stage. Exact-hash and identical-pHash cross-class blocking already exist; the new stage is only for DINO-based cross-class conflicts.
- Candidate generation should be global-DINO-first to keep the research path tractable:
  - build a pooled global DINO index across readable images
  - retrieve only top cross-class candidates per image
  - rerank those candidates with the patch-aware score
- Only the reranked candidate set participates in cross-class patch-aware blocking.
- Cross-class candidate generation must deduplicate symmetric pairs before patch reranking.

Notebook 0 parameter additions:

- `PATCH_AWARE_AUDIT_ENABLED`
- `PATCH_AWARE_TOPK`
- `PATCH_AWARE_CROSS_CLASS_CANDIDATES`
- `PATCH_DINO_AUTO_MIN`
- `PATCH_DINO_REVIEW_MIN`
- `PATCH_DINO_CROSS_CLASS_BLOCK_MIN`

Defaults:

- All research audit flags default to off in the tracked notebook.
- When off, current audit behavior remains unchanged.

Artifact changes:

- Extend existing CSV outputs with:
  - `patch_dino_score`
  - `global_dino_cosine`
  - `patch_match_topk`
  - `dino_score_source`
- Add one compact JSON artifact that records:
  - whether patch-aware audit was enabled
  - thresholds used
  - patch-aware score distribution summary
- Keep current artifact paths and guided-artifact flow where possible.

### 2. Full experimental local-global DINOv3 training head

Implement a new fusion mode for the trainer. Do not change the semantics of the existing `MultiScaleFeatureFusion`.

Architecture rules:

- Keep frozen DINOv3 backbone behavior unchanged.
- Keep current LoRA target-module resolution unchanged.
- Add a new fusion mode:
  - `legacy_multiscale` for current behavior
  - `research_local_global_dinov3` for the new path
- Keep the existing `MultiScaleFeatureFusion` implementation and tests valid for `legacy_multiscale`.

Fusion input contract:

- Legacy mode keeps the current contract:
  - trainer extracts hidden states
  - trainer selects `training.continual.fusion.layers`
  - `MultiScaleFeatureFusion` receives the selected tensors and may mean-pool token tensors
- Research mode must not reuse that lossy contract.
- Research mode should either:
  - pass the selected hidden states into a research fusion module that preserves token axes, or
  - pass the full hidden-state sequence into a research fusion module that performs its own layer selection
- The chosen approach must be explicit in code and tests. Do not hide the behavior behind `MultiScaleFeatureFusion`.

Token handling in research mode:

- Split DINOv3 outputs into:
  - CLS token
  - register tokens, using `model.config.num_register_tokens`
  - patch tokens
- Stop collapsing `[B, T, D]` features via plain mean pooling in research mode.
- Validate that each selected hidden state has enough tokens for `1 + num_register_tokens` before slicing patch tokens.

Research head structure:

- Global branch:
  - select configured hidden states
  - attention-pool patch tokens per selected layer
  - fuse pooled patch context with CLS and optional pooled register summary
- Local branch:
  - score last-layer patch tokens with a learned patch-saliency scorer
  - select top-k patch tokens
  - attention-pool only the selected local tokens
- Final branch fusion:
  - concatenate global and local branch outputs
  - apply learned MLP or gate fusion
  - emit one final fused feature vector

Trainer integration constraints:

- The research head must still be exposed to the trainer as one fusion module through `trainer.fusion`.
- Do not introduce separate top-level trainer fields such as `trainer.global_branch` or `trainer.local_branch`.
- Optimizer setup, checkpointing, export, and reload must continue to operate on:
  - `trainer.adapter_model`
  - `trainer.classifier`
  - `trainer.fusion`

Scope limit for this iteration:

- OOD calibration remains on the final fused feature vector.
- No separate OOD branch is introduced.
- No router or inference payload shape changes are included.

### 3. Config, metadata, and reconstruction

Extend the normalized `training.continual.fusion` surface so the new fusion path can be configured and reconstructed.

New fusion fields:

- `mode`
- `token_pooling`
- `use_register_tokens`
- `register_pooling`
- `local_branch_enabled`
- `local_branch_layers`
- `local_topk_patches`
- `local_score_hidden_dim`
- `global_output_dim`
- `local_output_dim`
- `final_fusion_hidden_dim`

Backward-compatibility rules:

- Existing configs without `fusion.mode` resolve to `legacy_multiscale`.
- Existing adapter metadata without `fusion.mode` loads through the legacy path.
- No schema bump is required unless tracked `config/*.json` files are changed in a way that requires migration logic.

Persistence requirements:

- Adapter metadata `fusion` payload must include `mode` and all branch-shaping fields required for reconstruction.
- Checkpoint save and load must restore the correct fusion module before state restoration.
- Exported adapter load must reconstruct the correct fusion module from `adapter_meta.json`.

Implementation touchpoints:

- `src/training/services/config_surface.py` must normalize `fusion.mode` and research-only fusion fields while preserving legacy defaults.
- `ContinualSDLoRAConfig` must expose the normalized fusion mode and branch-shaping fields through both `from_training_config(...)` and `as_contract_dict()`.
- Trainer initialization must use a fusion factory that chooses `MultiScaleFeatureFusion` for `legacy_multiscale` and the research head for `research_local_global_dinov3`.
- `encode(...)` must route hidden states according to fusion mode so the research head receives token-preserving tensors.
- `build_adapter_metadata(...)`, runtime adapter metadata, checkpoint snapshots, checkpoint restore, exported adapter load, and direct adapter summaries must preserve and surface the `fusion` payload without dropping research fields.

### 4. Notebook and validation updates

Notebook 0:

- Add the research audit parameters to the parameter cell.
- Pass them into `build_grouped_dataset_plan`.
- Keep current output locations and execution flow unchanged outside research mode.

Notebook 2:

- Do not make the research head the default.
- Add a visible research toggle or explicit research config override path that sets `training.continual.fusion.mode = "research_local_global_dinov3"`.
- Do not introduce hidden notebook-only behavior.

Notebook 3:

- No new public interface is required.
- `scripts.colab_adapter_smoke_test.py` already carries arbitrary `fusion` metadata through its summary dictionaries.
- Extend summary and discovery coverage so research `fusion.mode` is visible and non-breaking.

Validation:

- Update `scripts.validate_notebook_imports.py` so Notebook 0 contract checks require the new parameter-cell symbols.
- Keep legacy validation expectations intact when research mode is off.
- `scripts.validate_config_schema.py` does not need changes unless tracked config JSON files are updated.

### 5. Documentation

Update only the docs that would otherwise become misleading:

- Notebook 0 user guidance: describe patch-aware audit as research-only
- Architecture overview: describe the local-global head as an experimental path, not maintained default
- Avoid any claim that the implementation is a paper-faithful reproduction
- Cite the official DINOv3 token-layout documentation when describing CLS, register, and patch token slicing.
- Label the patch-aware audit and local-global head as engineering research adaptations unless the implementation and evaluation are later aligned with a specific paper.

## Test Plan

### Unit tests

Add tests for:

- patch-token extraction excluding CLS and register tokens
- clear failure for non-rank-3 or non-ViT-style DINO outputs in patch-aware mode
- symmetric patch-aware similarity scoring
- top-k trimmed patch score behavior on synthetic token layouts
- research-mode same-class merge and review decisions
- research-mode cross-class candidate mining and blocking
- symmetric deduplication of cross-class representation candidates
- config normalization for legacy and research fusion modes
- research fusion input routing that preserves token axes
- research fusion module forward-pass output shape
- research adapter metadata roundtrip
- checkpoint restore and adapter reload for research fusion mode

### Regression tests

Keep and extend current tests so:

- legacy grouped dataset audit behavior is unchanged when research flags are off
- existing `MultiScaleFeatureFusion` tests remain unchanged
- Notebook 0 import and contract tests pass with the new parameter symbols
- Notebook 3 summary and discovery tests accept research `fusion.mode`

### Smoke and integration tests

Add:

- one grouped dataset plan test with mocked patch-token embeddings verifying patch-aware decisions
- one grouped dataset test covering cross-class candidate generation plus reranked blocking
- one training smoke test that trains, exports, reloads, and predicts with `research_local_global_dinov3`
- one direct adapter smoke-test summary case that verifies research fusion metadata is exposed cleanly

## Assumptions And Defaults

- This work lands as a research branch, not as the maintained default path.
- Notebook 0 and training defaults remain unchanged unless research mode is explicitly enabled.
- OOD calibration stays on the final fused representation in this iteration.
- Patch-aware audit uses full patch-aware DINO scoring in research mode, but auto-merge remains conservative.
- Existing adapters and config payloads remain backward-compatible without migration.

## References

- Hugging Face Transformers DINOv3 documentation, for the ViT `last_hidden_state` layout with CLS, register tokens, and patch tokens: <https://huggingface.co/docs/transformers/en/model_doc/dinov3>
