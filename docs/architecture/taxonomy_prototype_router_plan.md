# Taxonomy + Prototype Router Plan

## Summary

The current M2 image set is technically clean, but the latest Notebook 8 run shows that the router handoff does not generalize well to realistic user-guided photos. This plan replaces manual crop/part decision-surface tuning with an automatic taxonomy-aware prototype router layer.

The goal is not to retrain or hand-tune the router every time a new plant is added. The target design is:

- Build a local taxonomy registry from adapter inventory and external taxonomy sources.
- Build visual prototype banks from each prepared runtime dataset.
- Reconcile VLM/router outputs against prototype similarity and taxonomy distance.
- Calibrate thresholds automatically on held-out data before promoting runtime behavior.
- Keep Notebook 8 as the live demo surface, but expose the extra evidence fields needed to debug decisions.

## Current Evidence

Latest M2 result inspected before this plan:

- Result folder: `docs/demo_results/m2/20260617T141041Z/`
- Total images: 512
- Passed: 192
- Failed: 320
- Router failure bucket: 264
- Adapter unavailable: 0
- Crop correct: 250 / 512
- Part correct: 224 / 512

Main failing pattern: realistic fruit/leaf user photos often route to close visual or taxonomic neighbors, or to `unknown_crop`, even though adapters exist. This means the next fix should focus on the router handoff and open-set rejection surface, not adapter availability.

## Literature Basis

The plan follows these findings and tool families:

- CLIP-style open-vocabulary recognition is useful for zero-shot semantic matching, but needs calibration and domain grounding before runtime decisions.
- BioCLIP-style biological image-language embeddings are a better first fit than generic image-text embeddings for plant/domain similarity.
- Grounding DINO / OWL-ViT style open-vocabulary detection can remain evidence for object/part localization, but should not be the only crop decision source.
- DINOv2-style self-supervised visual embeddings can support visual nearest-neighbor/prototype checks when text labels are ambiguous.
- Open-set and hierarchical open-set recognition literature supports using semantic hierarchy, risk/coverage curves, and fallback decisions instead of one global confidence threshold.
- GBIF Species API can provide the primary scientific-name, genus, and family backbone; EPPO or Wikidata can enrich plant-disease/crop naming where needed.

References:

- CLIP zero-shot/open-vocabulary recognition: <https://arxiv.org/abs/2103.00020>
- Open-vocabulary calibration for fine-tuned CLIP: <https://arxiv.org/html/2402.04655v4>
- C-TPT calibration: <https://arxiv.org/html/2403.14119v3>
- BioCLIP: <https://imageomics.github.io/bioclip/>
- Grounding DINO: <https://arxiv.org/abs/2303.05499>
- Grounding DINO 1.5: <https://arxiv.org/abs/2405.10300>
- OWL-ViT: <https://arxiv.org/abs/2205.06230>
- DINOv2: <https://arxiv.org/abs/2304.07193>
- Hierarchical open-set recognition: <https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Learning_Visual-Semantic_Hierarchical_Attribute_Space_for_Interpretable_Open-Set_Recognition_WACV_2025_paper.pdf>
- GBIF Species API: <https://techdocs.gbif.org/en/openapi/v1/species>
- EPPO Global Database: <https://gd.eppo.int/>

## Target Architecture

### 1. Taxonomy Registry

Create a durable registry that maps each supported adapter target to:

- `target_id`: for example `tomato__fruit`
- `crop_canonical_name`
- `part`: `leaf` or `fruit`
- `scientific_name`
- `common_names`
- `synonyms`
- `genus`
- `family`
- `supported_disease_labels`
- source metadata and cache timestamp

First implementation should be local and deterministic:

- Discover adapter targets from `data/prepared_runtime_datasets/*` and/or `runs/**/<crop>/<part>/continual_sd_lora_adapter`.
- Fetch/cache taxonomy metadata into a generated cache file.
- Keep hand-written overrides small and explicit only for unresolved names.

### 2. Prototype Bank

Create visual prototypes per supported crop/part and disease class using prepared runtime datasets:

- Input roots: `data/prepared_runtime_datasets/<crop>__<part>/{train,val,test,continual}`
- Exclude obvious OOD/OE rows unless explicitly building unknown prototypes.
- Store target centroid, class centroids, dispersion, sample count, and split metadata.
- First backend: existing BioCLIP/OpenCLIP embedding surface if available in repo/runtime.
- Optional second backend: DINOv2 embeddings for visual-only evidence.

Prototype artifacts should be generated, not manually curated:

```text
runs/_index/router_prototypes/<timestamp>/
  taxonomy_registry.json
  prototype_bank.json
  calibration_report.json
  summary.md
```

### 3. Reconciliation Layer

Add a narrow runtime helper that takes existing router evidence and prototype evidence:

- VLM/router predicted crop and part
- VLM/router confidence and margin
- prototype nearest target
- prototype nearest/second margin
- taxonomy relation between VLM crop and prototype crop
- current supported adapter inventory

Decision policy:

- If router and prototype agree on supported crop/part, accept.
- If router predicts a close taxonomy neighbor but prototype strongly matches a supported target, reconcile to the supported target.
- If prototype margin is weak or taxonomy relation is distant, return `unknown_crop` or `review/router_uncertain`.
- If part evidence conflicts strongly, abstain before loading an adapter.

This keeps the existing router useful while preventing brittle manual alias rules.

### 4. Automatic Calibration

Extend the calibration flow so thresholds are selected from data:

- crop similarity minimum
- nearest/second prototype margin
- taxonomy-distance allowance
- part confidence minimum
- unknown rejection threshold
- review/abstain operating point

Calibration should report risk/coverage tradeoffs globally and per target. Runtime promotion requires a held-out gate, not a hand-picked threshold.

### 5. Notebook 8 Reporting

Notebook 8 should keep the current user flow, but output extra decision fields:

- `vlm_crop`
- `vlm_part`
- `prototype_crop`
- `prototype_part`
- `reconciled_crop`
- `reconciled_part`
- `taxonomy_relation`
- `prototype_similarity`
- `prototype_margin`
- `reconcile_decision`
- `router_uncertainty_reasons`

These fields make M2 result analysis actionable without changing the demo UX.

## Implementation Plan

### Implemented Prototype Surface

The first report-only and config-gated runtime slice is implemented:

- `src/router/taxonomy_registry.py` builds `taxonomy_registry.v1` from `data/prepared_runtime_datasets/*`, optional adapter discovery under `runs/`, `config/plant_taxonomy.json`, optional override JSON, and an optional GBIF species-match cache. The default path is offline/cache-first; `--refresh-gbif` updates `external_taxonomy_cache.v1`.
- `src/router/prototype_bank.py` builds `router_prototype_bank.v1` from prepared runtime dataset images and excludes `ood`/`oe` unless explicitly requested. It supports the deterministic `image_stats_v1` backend and the optional `bioclip_open_clip` backend for BioCLIP/OpenCLIP embeddings.
- `src/router/prototype_reconciler.py` loads the registry and prototype bank, computes nearest/second target evidence, reports taxonomy relation, and returns `accept_router`, `use_prototype`, or `abstain`.
- `scripts/build_taxonomy_registry.py` writes a standalone registry artifact.
- `scripts/build_router_prototype_bank.py` writes the planned artifact folder shape:
- `scripts/calibrate_router_prototype_reconciler.py` sweeps prototype similarity and margin thresholds on a held-out manifest and writes `router_prototype_calibration.v1`.
- `scripts/colab_auto_router_adapter_prediction.py` can use the reconciler when `enable_prototype_reconciler=True` or `AADS_ENABLE_PROTOTYPE_RECONCILER=1`.
- `scripts/run_demo_checklist.py` exposes `--enable-prototype-reconciler`, `--prototype-bank`, `--taxonomy-registry`, `--prototype-calibration-report`, `--prototype-min-similarity`, and `--prototype-min-margin`. When a calibration report contains `selected_policy`, the runner can fill missing threshold arguments from that policy. It records `vlm_*`, `prototype_*`, `reconciled_*`, `taxonomy_relation`, `prototype_similarity`, `prototype_margin`, and `reconcile_decision`.
- Notebook 8 exposes the same prototype toggles for single-image and M2 runs. For M2, `M2_ENABLE_PROTOTYPE_RECONCILER=True` plus `M2_AUTO_BUILD_PROTOTYPES=True` builds missing prototype artifacts before the manifest run; `M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER=True` then sweeps thresholds and only keeps the reconciler enabled when a selected policy exists, unless `M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY=False`.
- Notebook 8 M2 publishing copies the run report, analysis summary, calibration report, taxonomy registry, prototype bank, and prototype summary into `docs/demo_results/m2/<timestamp>/` so the pushed demo result contains the evidence needed to reproduce the reconciliation decision surface.

```text
runs/_index/router_prototypes/<timestamp>/
  taxonomy_registry.json
  prototype_bank.json
  summary.md
```

The initial local backend is `image_stats_v1`, a deterministic lightweight image-statistics feature backend for artifact plumbing, smoke tests, and calibration scaffolding. Colab/GPU runs can build BioCLIP prototypes with `--embedding-backend bioclip_open_clip --embedding-model-id imageomics/bioclip-2.5-vith14 --embedding-device cuda`. Any backend is config-gated and should be promoted only after a held-out calibration report passes the documented risk/coverage constraints. DINOv2 can still be added later without changing the registry/reconciler contract.

Notebook 8 M2 prototype run shape:

```python
M2_ENABLE_PROTOTYPE_RECONCILER = True
M2_AUTO_BUILD_PROTOTYPES = True
M2_PROTOTYPE_EMBEDDING_BACKEND = "bioclip_open_clip"
M2_PROTOTYPE_EMBEDDING_MODEL_ID = "imageomics/bioclip-2.5-vith14"
M2_PROTOTYPE_EMBEDDING_DEVICE = DEVICE
M2_AUTO_CALIBRATE_PROTOTYPE_RECONCILER = True
M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY = True
M2_PROTOTYPE_MIN_SIMILARITY = None  # filled from selected calibration policy
M2_PROTOTYPE_MIN_MARGIN = None      # filled from selected calibration policy
```

### Phase 1: Registry

- Status: implemented.
- Registry discovers dataset/adapter targets, reads local crop disease taxonomy, applies explicit overrides, can use cached GBIF species metadata, and can refresh that cache on demand with `--refresh-gbif`.
- Unit coverage includes canonical metadata, unresolved crops, external cache resolution, and non-target directory filtering.

### Phase 2: Prototype Builder

- Add prototype-bank builder under `src/router/` or `src/pipeline/` and a thin script wrapper.
- Generate prototypes from prepared runtime datasets.
- Store counts, split provenance, embedding backend, and artifact hashes.
- Add a small smoke test on a synthetic/minimal fixture.

### Phase 3: Reconciler

- Status: implemented as a pure helper plus config-gated Notebook 8 integration.
- Unit coverage includes router/prototype agreement, unknown-router correction, wrong-part abstain, same-family taxonomy relation, and Notebook 8 helper handoff.

### Phase 4: Calibration

- Status: calibration CLI implemented for prototype similarity/margin threshold sweeps, and Notebook 8 M2 can auto-run it before the manifest pass.
- Runtime promotion rule: if `M2_REQUIRE_CALIBRATED_PROTOTYPE_POLICY=True` and calibration does not produce `selected_policy`, Notebook 8 disables the prototype reconciler for that run instead of applying uncalibrated thresholds.
- Remaining evidence work: run the calibration on the full held-out M2/router-eval surface in Colab/GPU with production constraints and compare the selected policy against the previous M2 baseline.

### Phase 5: Notebook 8 and M2 Rerun

- Status: Notebook 8 helper path, Notebook 8 parameter cell, M2 auto-build, M2 auto-calibration, M2 runner fields, and M2 result provenance publishing are wired behind explicit flags.
- Remaining validation: run a bounded M2 smoke with generated artifacts, then the full 512-image M2 manifest in Colab/GPU and compare against `docs/demo_results/m2/20260617T141041Z/`.

## Acceptance Gates

- Taxonomy registry builds for all eight current supported targets.
- Prototype bank builds without manual per-target threshold edits.
- Reconciler unit tests cover agreement, correction, abstain, and unsupported cases.
- Calibration report selects thresholds from held-out data and records rejected policies.
- Notebook 8 still runs the same canonical manifest and records reconciliation fields when enabled.
- Full M2 rerun improves router failure count materially without increasing unsupported or wrong-part adapter loads.
- Adapter availability remains 0 failures for supported reconciled targets.

## Assumptions

- Existing adapter datasets are good enough to build first-pass visual prototypes.
- The first implementation can be report-first or config-gated before runtime promotion.
- New plant addition should require dataset/prototype/taxonomy refresh, not manual decision-surface tuning.
- External taxonomy lookup is cached so Colab/demo runs do not depend on live network access.
