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

The first report-only slice is implemented without changing Notebook 8 runtime decisions:

- `src/router/taxonomy_registry.py` builds `taxonomy_registry.v1` from `data/prepared_runtime_datasets/*`, optional adapter discovery under `runs/`, `config/plant_taxonomy.json`, and optional override JSON.
- `src/router/prototype_bank.py` builds `router_prototype_bank.v1` from prepared runtime dataset images and excludes `ood`/`oe` unless explicitly requested.
- `scripts/build_taxonomy_registry.py` writes a standalone registry artifact.
- `scripts/build_router_prototype_bank.py` writes the planned artifact folder shape:

```text
runs/_index/router_prototypes/<timestamp>/
  taxonomy_registry.json
  prototype_bank.json
  summary.md
```

The initial local backend is `image_stats_v1`, a deterministic lightweight image-statistics feature backend for artifact plumbing, smoke tests, and calibration scaffolding. It is intentionally not a runtime promotion backend. BioCLIP/DINOv2 embedding backends can replace or supplement this artifact schema in a later phase without changing the registry contract.

### Phase 1: Registry

- Add taxonomy registry schema under `src/router/`.
- Add a script to build/cache the registry from adapter discovery.
- Add unit tests for canonical names, synonyms, and unresolved-name behavior.
- Document the registry artifact contract.

### Phase 2: Prototype Builder

- Add prototype-bank builder under `src/router/` or `src/pipeline/` and a thin script wrapper.
- Generate prototypes from prepared runtime datasets.
- Store counts, split provenance, embedding backend, and artifact hashes.
- Add a small smoke test on a synthetic/minimal fixture.

### Phase 3: Reconciler

- Add a pure helper that reconciles router evidence, prototype evidence, taxonomy distance, and supported inventory.
- Keep runtime behavior behind a config flag until calibration passes.
- Unit-test agreement, close-neighbor correction, weak-margin abstain, wrong-part abstain, and unsupported-crop fallback.

### Phase 4: Calibration

- Extend or add a calibration script for prototype-router thresholds.
- Produce a machine-readable report plus a short Markdown summary.
- Include target-level risk/coverage and unknown rejection metrics.
- Require holdout validation before enabling the reconciler by default.

### Phase 5: Notebook 8 and M2 Rerun

- Wire the reconciler into the Notebook 8 auto helper path.
- Add the reporting fields to `scripts/run_demo_checklist.py` analysis output.
- Run a bounded M2 smoke first, then the full 512-image M2 manifest.
- Compare against `docs/demo_results/m2/20260617T141041Z/`.

## Acceptance Gates

- Taxonomy registry builds for all eight current supported targets.
- Prototype bank builds without manual per-target threshold edits.
- Reconciler unit tests cover agreement, correction, abstain, and unsupported cases.
- Calibration report selects thresholds from held-out data and records rejected policies.
- Notebook 8 still runs the same canonical manifest.
- Full M2 rerun improves router failure count materially without increasing unsupported or wrong-part adapter loads.
- Adapter availability remains 0 failures for supported reconciled targets.

## Assumptions

- Existing adapter datasets are good enough to build first-pass visual prototypes.
- The first implementation can be report-first or config-gated before runtime promotion.
- New plant addition should require dataset/prototype/taxonomy refresh, not manual decision-surface tuning.
- External taxonomy lookup is cached so Colab/demo runs do not depend on live network access.
