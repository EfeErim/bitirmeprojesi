# Colab Training Manual

This guide is written for someone who may be opening the AADS notebooks for the first time.

It covers the maintained Colab surfaces:

- Notebook 0: `colab_notebooks/0_grouped_dataset_preparation.ipynb`
- Notebook 2: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Notebook 3: `colab_notebooks/3_adapter_smoke_test.ipynb`

Notebook 0 is the audit-first data-preparation surface. Notebook 2 is the training surface and consumes the prepared runtime dataset produced by Notebook 0. Notebook 3 checks an already exported adapter directly.

The repo also tracks one auxiliary notebook:

- Notebook 4: `colab_notebooks/4_simple_adapter_smoke_test.ipynb`

Notebook 4 is a minimal convenience UI over the same direct-adapter smoke-test helpers used by Notebook 3. It is useful for quick manual checks, but it is not a separate canonical workflow contract.

If you are brand new to the repo, read [../../README.md](../../README.md) first.

## What You Will Do In These Notebooks

### Notebook 0

Notebook 0 audits a flat class-root dataset from the repo workspace, groups likely duplicate and augmentation families with DINOv3 and BioCLIP-2.5 similarity signals, applies conservative metadata-free source-style and label-risk triage, writes review artifacts, and can materialize a prepared runtime dataset. If `CROP_NAME` or `PART_NAME` are left blank in the parameter cell, Notebook 0 now prompts for them after the dataset is selected and suggests defaults from the dataset folder name when possible.

### Notebook 2

Notebook 2 now does one thing only:

- select a prepared runtime dataset under `data/prepared_runtime_datasets/`
- train from that dataset directly

### Notebook 3

Notebook 3 loads one saved adapter directly so you can inspect it and run a quick prediction without the router.

### Notebook 4

Notebook 4 exposes the same direct adapter validation path behind a smaller widget-based UI for quick manual smoke tests.

## Important Terms

- `flat class-root dataset`: the simple folder layout you prepare by hand for Notebook 0
- `runtime dataset`: the split layout used by workflow training; Notebook 0 creates it and Notebook 2 consumes it
- `adapter bundle`: the saved training output that contains model weights, metadata, and OOD state
- `telemetry`: the logs and mirrored artifacts saved during a notebook run
- `OOD`: "out of distribution," meaning the image may not belong to the supported disease classes
- `readiness`: the final deployment verdict written to `production_readiness.json`

## Before You Start

Make sure you have these things ready:

1. A dataset for one crop.
2. A Colab runtime that has GPU access if you intend to use the default `cuda` path.
3. A Hugging Face token if the required models need authenticated access.
4. Enough runtime storage for outputs, telemetry, and checkpoints.

Private repo note:

- if the notebook starts outside the repo workspace and this GitHub repo is private, set `GH_TOKEN` or `GITHUB_TOKEN` as a Colab secret before running the bootstrap cell
- alternatively, set `AADS_REPO_ROOT` to an existing checkout so the notebook does not need to clone

Important current behavior:

- requesting `device="cuda"` fails immediately when CUDA is unavailable
- the notebook validates the dataset before training starts
- auto-push to GitHub still requires a token with write access to the target repository, but push failures now log a warning and keep the run artifacts locally instead of aborting the notebook
- training now resolves supported-class reference counts from `split_manifest.json` when that runtime manifest exists, otherwise from the runtime `continual` split counts
- if any supported class resolves below `100` images, training fails before adapter initialization instead of silently trying a fragile few-shot run, unless you explicitly enable the research-only few-shot mode
- every maintained notebook now begins with an access/update check cell so you can confirm repo freshness, GitHub access mode, and Hugging Face access mode before long runs
- Notebook 2 now expects a prepared runtime dataset under `data/prepared_runtime_datasets/`
- Notebook 0 is the maintained surface for grouped prep, runtime-dataset materialization, and optional OOD pool injection
- Notebook 0 skips OOD copying by default when `OOD_DATASET_NAME` and `OOD_ROOT` are blank; set one of those values, or set `ASK_FOR_OOD_ROOT=True`, when you want an interactive OOD path prompt

## Notebook 2 In Plain English

This is the current Notebook 2 training flow from start to finish:

1. find or initialize the repo workspace
2. install notebook requirements
3. prepare repo-local telemetry and checkpoint directories
4. set `CROP_NAME` and `PART_NAME` in the run-identity cell
5. run the access/update check cell and confirm token needs before a long run
6. resolve a Hugging Face token from environment variables or Colab secrets
7. select and validate a prepared runtime dataset under `data/prepared_runtime_datasets/<dataset_key>/`
8. inspect the selected runtime dataset and the current runtime hardware
9. preview recommended training parameters, review warnings or blockers, and optionally accept them with one yes/no prompt
10. apply any `MANUAL_PARAM_OVERRIDES` on top of the accepted or rejected recommendation result
11. train the continual SD-LoRA adapter
12. restore the best model state
13. calibrate OOD
14. write validation and test artifacts
15. ask for an OOD folder path, use that real OOD data when provided, fall back to the selected runtime dataset's `ood/` folder when you press Enter and it exists, otherwise run the held-out fallback benchmark automatically
16. write `production_readiness.json`
17. write guided navigation files such as `guided/00_start_here.md`, `guided/01_run_overview.json`, and `guided/02_file_catalog.json` without deleting raw artifacts
18. write canonical training traceability files `training/experiment_manifest.json` and `training/optimization_record.json`
19. mirror outputs into `runs/<RUN_ID>/`
20. optionally auto-push the mirrored run record to GitHub
21. optionally auto-disconnect the Colab runtime after final exports succeed

Important recommendation:

- use Notebook 0 first to inspect the audit artifacts and materialize the runtime dataset
- use Notebook 2 only after the runtime dataset is ready

## Notebook 0 In Plain English

This is the current Notebook 0 flow from start to finish:

1. find or initialize the repo workspace
2. install notebook requirements
3. run the access/update check cell and confirm token needs before the audit
4. resolve a Hugging Face token from environment variables or Colab secrets
5. choose a dataset: if `DRIVE_DATASET_PATH` is blank, repo-local datasets under `REPO_DATASET_ROOT` are shown as numbered options when `REPO_DATASET_NAME` is blank; if `DRIVE_DATASET_PATH` is filled and `IMPORT_FROM_DRIVE=True`, the notebook scans that Drive path, shows Drive dataset options when `DRIVE_DATASET_NAME` is blank, copies the selected dataset into the repo workspace, and updates `DATASET_ROOT`
6. scan the flat class-root dataset
7. normalize class names against the crop taxonomy when possible
8. audit exact duplicates, perceptual-hash neighbors, DINOv3/BioCLIP similarity families, source-style proxy groups, and label-risk cues
9. write review artifacts, label-risk artifacts, a grouped split manifest, and guided navigation files such as `guided/00_start_here.md` and `guided/02_file_catalog.json`
10. optionally materialize a prepared runtime dataset under `data/prepared_runtime_datasets/<dataset_key>/` and pull a repo OOD tree from `data/ood_dataset/<dataset_name>/` into the runtime `ood/` folder when you want Notebook 0 to complete the full prep flow itself
11. if `SAVE_RUNTIME_DATASET_TO_GITHUB=True`, force-add and push the prepared runtime dataset path to GitHub when a token is available; token or push failures leave the local dataset and run artifacts in place

Current Notebook 0 behavior keeps audit outputs under the repo workspace and mirrored repo run directory. It no longer copies the data-prep artifacts or prepared runtime dataset into the Drive telemetry tree. By default, Notebook 0 also prepares the report-based working copy, materializes the runtime dataset after a clean audit, and attempts to push the ready runtime dataset to GitHub; set `PREPARE_DATASET_FROM_REPORTS=False` or `MATERIALIZE_AFTER_REVIEW=False` only when you intentionally want an audit-only pass, and set `SAVE_RUNTIME_DATASET_TO_GITHUB=False` when you want the prepared runtime dataset to remain local to the Colab checkout.

For adapter performance, treat Notebook 0 as a curation tool, not just a cleanup tool:

- keep exact duplicates and near-duplicate families together before the final split
- prefer same-crop unsupported diseases, blur, occlusion, and other realistic hard negatives in `ood/`
- keep reused source packages, capture sessions, and offline augmentation families visible in the audit outputs
- materialize only after the report says the class support and source separation are trustworthy enough for training

Notebook 0 keeps canonical `val` and `test` stricter than `continual`. A sample must be the family canonical item and must avoid synthetic, eval-quality, source-style, and label-risk flags before it can enter canonical evaluation. Risky but usable third-party samples are retained for `continual` by default. The label triage is heuristic and review-assisted; it is not ground truth.

If a class has zero evaluation-eligible families after grouped prep, Notebook 0 records it under `skipped_classes` and omits that class from the materialized runtime dataset. Classes with only one or two eligible families still block materialization because they cannot support the maintained `continual`/`val`/`test` split contract.

## The Dataset Format Notebook 2 Accepts

Notebook 2 expects the prepared runtime layout produced by Notebook 0:

```text
data/prepared_runtime_datasets/<dataset_key>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

Important distinction:

- Notebook 0 owns flat class-root audit and split materialization
- Notebook 2 trains only from the already prepared runtime dataset
- the real `ood/` pool is separate unknown-input evidence, not another class
- if you maintain a real OOD pool, the preferred durable location is `data/prepared_runtime_datasets/<dataset_key>/ood/`; Notebook 0 is the maintained surface that writes it there
- Notebook 2 also asks for an explicit OOD folder path. Use that prompt when the OOD pool already exists elsewhere in the Colab workspace and you do not want to rematerialize the runtime dataset.

The generated runtime dataset includes:

- `split_manifest.json`

Notebook 2 now also inspects that runtime dataset before engine initialization. It uses repo-visible signals such as split sizes, manifest class counts, grouped-prep risk hints, real `ood/` availability, and current hardware capacity to recommend a bounded set of training and runtime parameters. The notebook does not silently apply a hidden profile:

- it prints the current-vs-recommended parameter diff
- it asks once whether to apply the recommended values
- `MANUAL_PARAM_OVERRIDES = {}` always wins over both the raw parameter cell and the accepted recommendation result


Contract reminder:

- Notebook 0 accepts the flat class-root layout and writes the runtime dataset
- Notebook 2 and `TrainingWorkflow.run(...)` both use the runtime split layout
- Notebook 0 remains the maintained audit-first surface when you want to inspect or fix prep issues before training

## Notebook 2 run identity and traceability

Notebook 2 run identity is still anchored on the mirrored repo run directory:

```text
runs/<RUN_ID>/
```

Current traceability rules:

- `RUN_ID` identifies the notebook execution record
- `crop_name`, `part_name`, and `dataset_key` identify the training cohort
- `dataset_key` should match the prepared runtime dataset folder selected under `data/prepared_runtime_datasets/`
- dataset lineage is finalized by the selected runtime dataset plus the SHA-256 of its `split_manifest.json`

The canonical optimizer-facing files live inside the mirrored artifact tree:

```text
runs/<RUN_ID>/outputs/colab_notebook_training/artifacts/training/
  summary.json
  run_context.json
  experiment_manifest.json
  optimization_record.json
```

Notebook 2 keeps using the summary-merge helper surface, but it now updates the same canonical manifest and optimization-record schema that the workflow writes. That means notebook-specific metadata such as the selected runtime dataset key, notebook parameter overrides, readiness summary, and mirrored export paths lands in the same traceability contract instead of a notebook-only side structure.

When the canonical traceability files are updated successfully, the repo-local aggregate registry is also refreshed best-effort under:

```text
runs/_index/
  trials.jsonl
  latest_registry.json
  pareto_inputs.json
  pareto_frontiers.json
```

Current phase-2 analysis behavior:

- `pareto_frontiers.json` lists the non-dominated runs inside each comparable cohort
- Bayesian proposal generation is disabled, so fresh registry rebuilds do not write `bayesian_recommendations.json`
- registry files are rebuilt automatically when Notebook 2 traceability updates refresh the local run registry

Notebook 2 campaign behavior:

- Bayesian optimizer campaign automation is disabled.
- Notebook 2 uses the visible notebook parameters only.
- If `OPTIMIZATION_CAMPAIGN_MODE` is set to `continue` or `stop`, the helper ignores it and records disabled campaign status.

When you want to inspect the comparable cohort directly from the repo, use:

```powershell
.\scripts\python.cmd scripts/optimize_training_runs.py --dataset-lineage-key <dataset_key>::<split_manifest_sha256> --crop-name <crop> --part-name <part>
```

This command reports Pareto context only. Bayesian proposals and `--execute` optimization runs are disabled.

## How The Split Is Created

The dataset materialization step is handled by `scripts/colab_dataset_layout.py`.

Current behavior:

- grouped Notebook 0 prep uses a 60/20/20-style family split target with small-class safeguards
- the older direct `scripts/colab_dataset_layout.py` helper still uses the 80/10/10-style split policy for non-grouped runtime layout conversion
- the runtime split names are `continual`, `val`, and `test`
- Notebook 0 uses the automatic materialization strategy, which prefers links over full copies on Colab and other non-Windows systems
- on Windows, materialization defaults to copying files instead of symlinks

## Prepare Trustworthy Evaluation Data Before Training

Notebook 2 no longer owns auto-splitting. If you want a trustworthy benchmark, do the data preparation work in Notebook 0 before training.

Use the Notebook 0 prep path when you want grouped prep before training. It is most appropriate when:

- each class folder contains only real, independent images
- you are not mixing multiple public sources into one flat pool
- you are not mixing offline augmentations or synthetic images into the same evaluation candidate set
- you accept a random image-level split as the benchmark

Do not treat a quick runtime-only smoke run as the main benchmark path when your flat dataset was built by merging public sources, adding offline augmentations, including GAN-generated images, or mixing repeated captures of the same plant, session, or original photo family.

Why this matters:

- if you merge sources first and split later, source style can leak across `continual`, `val`, and `test`
- if offline augmentations or synthetic variants are present before splitting, sibling images can land in different splits
- if repeated captures of the same leaf or plant exist, the random split can produce an unrealistically easy test set

This is consistent with duplicate-aware image-benchmark guidance such as ciFAIR and with literature warning that PlantVillage-style plant-disease datasets can contain shortcut cues and domain bias instead of deployment-like variability.[^cifair] [^plantvillage-bias]

### Recommended data-preparation workflow

1. Deduplicate before final splitting.

   Run both:

   - exact duplicate checks such as file hash comparisons
   - near-duplicate checks such as perceptual-hash or visual-family clustering

   Remove exact duplicates from the benchmark set and keep near-duplicate families together in one split.

2. Split by group, not by image.

   Keep these together:

   - one original image and all of its offline augmentations
   - one synthetic image family and its close variants
   - images from the same plant, same capture session, or same collection event when that metadata exists

3. Prefer real images for authoritative `val` and `test`.

   A practical rule is:

   - `continual/` may include curated augmentations when you intentionally want them for training
   - `val/` and `test/` should prefer real, non-synthetic images
   - GAN-generated images should be train-only or evaluated in a separate ablation, not mixed silently into the authoritative test split

4. Keep grouped evaluation when sources are merged.

   Preferred order:

   - grouped held-out test split
   - grouped balanced split
   - image-level random split only as a convenience baseline

   If your data comes from multiple public datasets, a grouped held-out test split is much more trustworthy than a random split after merge.

5. Materialize the runtime dataset through Notebook 0 when you want audit-first control.

   When you need a trustworthy benchmark and want to inspect the prep outputs before training, let Notebook 0 write this layout directly:

   ```text
   data/<crop>/
     continual/<class>/*
     val/<class>/*
     test/<class>/*
     ood/*
   ```

   Then train through the canonical workflow or CLI:

   ```powershell
   .\scripts\python.cmd -m src.app.cli training tomato data\prepared_runtime_datasets outputs\training_run --config-env colab
   ```

6. Keep OOD separate from classification splitting.

   Do not mix unknown images into the supported class-root dataset. Keep `ood/` as a separate pool under the runtime dataset and match it to the real deployment contract.

### Practical rules for merged public tomato-leaf datasets

If you combine multiple public tomato-leaf datasets, use these rules:

- never merge raw images, augmented variants, and GAN samples into one flat folder and trust the notebook random split as the main benchmark
- keep all derivatives of one original image in the same split
- keep one clean held-out test slice of real leaf images that were never used to generate training augmentations
- report whether the test slice is grouped held-out, grouped balanced, or only random-after-merge
- treat random-after-merge metrics as smoke-test evidence, not deployment evidence

### Roboflow and other offline augmentation tools

Roboflow, offline scripts, and similar dataset-expansion tools are not the problem by themselves. The problem is when augmented images are mixed back into the full pool before the authoritative split is created.

Use this rule:

- split first by original-image family
- generate or assign augmentations after the split
- keep all derivatives of one original image in the same split
- prefer real images for authoritative `val` and `test`

For example, if you have a grape dataset and you multiplied the images with Roboflow:

- it is acceptable to keep those Roboflow variants in `continual/` when you intentionally want them for training
- it is not acceptable to mix originals and Roboflow variants into one flat folder and let Notebook 2 random-split them as the main benchmark

If you want to measure whether augmentation helps, compare two controlled runs against the same clean held-out test set instead of silently mixing augmented variants into that test set.

When you need materialized offline variants after a clean runtime split, use the repo helper instead of re-splitting a mixed folder:

```powershell
.\scripts\python.cmd scripts/augment_runtime_train_split.py --source-root data\prepared_runtime_datasets\grape__fruit_curated --output-root data\prepared_runtime_datasets\grape__fruit_curated_train_aug --variants-per-image 2
```

The helper copies the selected runtime dataset, generates deterministic PIL variants only from `continual/` images, leaves `val/`, `test/`, and `ood/` unchanged, and writes `reference_image_count` into `split_manifest.json`. Training and Notebook 2 use that reference count for the production minimum, so generated variants can improve optimization exposure without being counted as independent real-image support.

### When source separation is uncertain

Public plant-disease datasets often do not provide enough metadata to prove that all sources are fully independent. Different dataset packages may reuse the same original photo, a resized copy, a crop, or an offline augmentation family.

Do not claim full source independence unless you can actually verify it. Instead, use a conservative grouped protocol that reduces leakage risk as much as possible:

1. keep dataset-package boundaries visible when you still know them
2. run exact duplicate detection
3. run near-duplicate detection
4. let Notebook 0 use path tokens, stock-site or screenshot hints, aspect/resolution buckets, border/layout traits, and web-export cues as weak provenance proxies when metadata is missing
5. group suspiciously similar or source-style-related images together even if you are not fully certain they share one original
6. split by those groups instead of by individual images
7. describe the evaluation honestly as grouped held-out, grouped balanced, or random-after-merge

This is still much more trustworthy than merging everything first and relying on an image-level random split.

### A conservative protocol when lineage is incomplete

If you cannot fully reconstruct which augmented images came from which originals, use the most conservative practical workflow you can support:

1. start from the rawest available image export rather than the already mixed training folder
2. keep dataset-package boundaries visible in folder names or manifests
3. run duplicate and near-duplicate audits before finalizing the runtime split
4. keep flagged duplicate families in one split even when that lowers the apparent benchmark score
5. reserve a clean real-image test slice that excludes synthetic and likely augmented families
6. treat any remaining random-after-merge result as a convenience baseline, not as the headline claim

When lineage is weak, the goal is not to prove perfect independence. The goal is to remove obvious leakage paths and make the benchmark defensible.

### When a quick runtime-only smoke run is still acceptable

Notebook 2 is still useful when you want:

- a fast smoke run to validate training and export surfaces
- an early baseline before building a stricter evaluation split
- a convenience experiment on a small, clean, single-source dataset whose runtime split was already prepared cleanly

Use the stricter Notebook 0 audit-first materialized runtime dataset when the result will be used as the main claim about model quality.

[^cifair]: C. Barz and J. Denzler, "[Do We Train on Test Data? Purging CIFAR of Near-Duplicates](https://pubmed.ncbi.nlm.nih.gov/34460587/)," *Journal of Imaging*, 2021.
[^plantvillage-bias]: A. Naqvi et al., "[Uncovering bias in the PlantVillage dataset: A comparison of diseased plant leaves in isolation and within canopies](https://doi.org/10.48550/arXiv.2206.04374)," arXiv, 2022.

## Training Settings, Explained By Purpose

The config surface can look large when you are new. The easiest way to understand it is by what each section changes.

### Model identity

These decide which base model and LoRA strategy are used:

- `training.continual.backbone.model_name`
- `training.continual.adapter.target_modules_strategy`
- `training.continual.adapter.lora_r`
- `training.continual.adapter.lora_alpha`
- `training.continual.adapter.lora_dropout`

### Feature fusion

These control how features from multiple backbone layers are combined:

- `training.continual.fusion.layers`
- `training.continual.fusion.output_dim`
- `training.continual.fusion.dropout`
- `training.continual.fusion.gating`

### Optimization

These control the actual training process:

- `training.continual.learning_rate`
- `training.continual.weight_decay`
- `training.continual.num_epochs`
- `training.continual.batch_size`
- `training.continual.optimization.grad_accumulation_steps`
- `training.continual.optimization.max_grad_norm`
- `training.continual.optimization.mixed_precision`
- `training.continual.optimization.loss_name`
- `training.continual.optimization.logitnorm_tau`
- `training.continual.optimization.label_smoothing`
- `training.continual.optimization.scheduler`
- `training.continual.early_stopping`

### Data loading

These affect image loading and error tolerance:

- `training.continual.data.sampler`
- `training.continual.data.loader_error_policy`
- `training.continual.data.target_size`
- `training.continual.data.augmentation_policy`
- `training.continual.data.randaugment_num_ops`
- `training.continual.data.randaugment_magnitude`
- `training.continual.data.few_shot_research_mode`
- `training.continual.data.few_shot_min_class_samples`
- `training.continual.data.cache_size`
- `training.continual.data.cache_train_split`
- `training.continual.data.validate_images_on_init`

Current sampler behavior:

- the shipped default is `training.continual.data.sampler: "auto"`
- `auto` keeps normal shuffle training when the continual split is roughly balanced
- `auto` switches the train loader to `WeightedRandomSampler` when the largest known-class count is at least `1.5x` the smallest non-zero class count
- set `"shuffle"` or `"weighted"` explicitly when you do not want the automatic rule
- this threshold is a repo-level engineering heuristic for long-tail handling, not a paper-faithful claim that one ratio is universally optimal

Current init-validation behavior:

- the shipped default is `training.continual.data.validate_images_on_init: false`
- turn it on when you want an eager file-integrity sweep before the first batch
- leaving it off avoids a full `Image.verify()` pass across every split during loader construction

Current augmentation behavior:

- the shipped train-time online augmentation policy is `training.continual.data.augmentation_policy: "randaugment"`
- supported policies are `"randaugment"`, `"basic"`, and `"none"`
- `"basic"` keeps the earlier hand-written crop/flip/rotation/color-jitter/blur policy
- `"none"` uses deterministic resize plus normalization for ablation runs
- RandAugment is applied only to the training split; validation, test, OOD, and inference preprocessing stay deterministic
- optional materialized offline augmentation is available through `scripts/augment_runtime_train_split.py`; it must run after the runtime split exists and it only writes generated files under `continual/`
- runtime manifests may include `reference_image_count` so generated offline augmentations do not satisfy the production minimum image-count guardrail by themselves
- offline augmented image families still need the grouped split rules above, because online augmentation does not fix leakage from pre-generated variants

Current class-imbalance behavior layered on top of the sampler:

- the workflow first resolves per-class reference counts from `split_manifest.json`; if that manifest is absent, it falls back to runtime `continual` counts
- any supported class below `100` resolved images now hard-fails the run before adapter initialization unless few-shot research mode is explicitly enabled
- when all supported classes pass that floor and at least one supported class is in the `100-200` range, the training loss automatically receives class-balanced weighting based on the effective-number method from Cui et al.
- once class-balanced mode activates, weights are computed for all supported classes from the same resolved counts rather than only the `100-200` subset
- the weighted loss applies only to the training classifier loss; validation and test loss remain plain CE so early stopping and readiness artifacts stay comparable to historical runs
- the workflow records the resolved counts, activation decision, and normalized per-class weights in the training run context and summary artifacts
- the `100` floor and `100-200` trigger are repo policy for this adapter workflow; the effective-number weighting itself is literature-backed, but these exact thresholds are engineering policy rather than paper-defined universal constants

Few-shot research mode:

- in Notebook 2, set `FEW_SHOT_RESEARCH_MODE = True` in the top parameter cell to run a few-shot ablation
- `training.continual.data.few_shot_research_mode: true` bypasses the production `100` image/class floor for experiment runs only
- `FEW_SHOT_MIN_CLASS_SAMPLES` controls the notebook lower hard floor; the canonical config key is `training.continual.data.few_shot_min_class_samples` and defaults to `1`
- artifacts still record `production_under_min_classes` and `production_guardrail_bypassed` under the class-balance context
- treat few-shot research runs as ablations, not deployment-ready adapter evidence

Current loss behavior:

- the shipped default is `training.continual.optimization.loss_name: "logitnorm"`
- set `loss_name: "cross_entropy"` when you want the plain CE baseline
- `training.continual.ood.ber_enabled: true` is incompatible with LogitNorm, so BER experiments must explicitly use `loss_name: "cross_entropy"`

### OOD behavior

These control how OOD calibration and scoring behave:

- `training.continual.ood.threshold_factor`
- `training.continual.ood.primary_score_method`
- `training.continual.ood.energy_temperature_mode`
- `training.continual.ood.energy_temperature`
- `training.continual.ood.energy_temperature_range`
- `training.continual.ood.energy_temperature_steps`
- `training.continual.ood.radial_l2_enabled`
- `training.continual.ood.radial_beta_range`
- `training.continual.ood.radial_beta_steps`
- `training.continual.ood.knn_backend`
- `training.continual.ood.knn_chunk_size`
- `training.continual.ood.sure_enabled`
- `training.continual.ood.sure_semantic_percentile`
- `training.continual.ood.sure_confidence_percentile`
- `training.continual.ood.conformal_enabled`
- `training.continual.ood.conformal_alpha`
- `training.continual.ood.conformal_method`
- `training.continual.ood.conformal_raps_lambda`
- `training.continual.ood.conformal_raps_k_reg`

Current shipped default:

- the raw config surface ships `training.continual.ood.primary_score_method: "auto"`
- the trainer starts with the concrete detector path on `"ensemble"`
- when the repo only has one shared real `ood/` pool, the workflow keeps that concrete runtime method instead of auto-tuning on the same pool later used for the final readiness verdict
- the held-out fallback benchmark can still auto-select a concrete winner because it is separate proxy evidence rather than the final real-OOD deployment verdict
- fallback-only readiness can now be `provisional`, but not fully deployable
- if you want to promote `energy` or `knn` using real OOD evidence, inspect the split-local comparison artifacts and rerun with an explicit `training.continual.ood.primary_score_method`, or keep a separate dev OOD pool outside the shipped contract
- energy scoring can optionally keep a fixed temperature or auto-calibrate one from the calibration split
- kNN scoring can use `cdist`, chunked search, or optional FAISS when available
- conformal mode can be threshold conformalization, APS, or RAPS depending on whether you want rejection calibration or set-valued classification

### Readiness policy

These control how strict the final deployment verdict is:

- `training.continual.evaluation.best_metric`
- `training.continual.evaluation.emit_ood_gate`
- `training.continual.evaluation.require_ood_for_gate`
- `training.continual.evaluation.ood_benchmark_min_classes`

Practical interpretation:

- `emit_ood_gate` controls whether split-local `metric_gate.json` files are written to disk
- `require_ood_for_gate` controls whether the metric policy fails when OOD evidence is missing or insufficient
- even when the metric policy passes, the final status remains `provisional` until a real runtime `ood/` pool is part of the evidence

### Colab runtime controls

These do not change the model itself. They change the notebook runtime behavior:

- `colab.training.num_workers`
- `colab.training.pin_memory`
- `colab.training.validation_every_n_epochs`
- `colab.training.checkpoint_every_n_steps`
- `colab.training.checkpoint_on_exception`
- `colab.training.stdout_progress_batch_interval`
- `colab.training.stdout_progress_min_interval_sec`

Current Colab default tradeoffs:

- the Colab environment now ships a high-VRAM large-batch profile: `batch_size=96`, `grad_accumulation_steps=1`, `mixed_precision="bf16"`, and `num_workers=12`
- the Colab environment disables deterministic training so CuDNN can use faster kernels
- the Colab environment validates every 2 epochs by default, while still forcing validation on the final epoch
- if you do not provide a real `ood/` split, the workflow now falls back to the held-out benchmark automatically; readiness still fails when that benchmark is impossible or below target
- Notebook 2 can also validate every `N` epochs instead of every epoch; this reduces runtime but makes best-model and early-stopping decisions less responsive between validation checkpoints
- checkpointing and live batch-progress cadence are scaled for the larger per-step sample count so resume points and logs do not become too sparse
- the Colab environment now uses a much larger cache budget and can cache the continual train split too, which is intended for high-RAM A100 sessions to reduce repeated image decode and disk I/O
- live batch-status telemetry is throttled so `latest_status.json` is not rewritten on every batch

## Notebook-Only Top Cell Toggles

Notebook 2 also exposes a small set of notebook-level toggles:

- `CACHE_TRAIN_SPLIT`
- `FEW_SHOT_RESEARCH_MODE`
- `FEW_SHOT_MIN_CLASS_SAMPLES`
- `BER_ENABLED`
- `BER_LAMBDA_OLD`
- `BER_LAMBDA_NEW`
- `AUTO_PUSH_TO_GITHUB`
- `AUTO_PUSH_REMOTE_NAME`
- `AUTO_PUSH_BRANCH`
- `AUTO_DISCONNECT_RUNTIME`
- `AUTO_DISCONNECT_GRACE_SECONDS`

These are convenience controls for the notebook surface. The canonical workflow still lives in `TrainingWorkflow.run(...)`.

Notebook 2 now takes its visible training and OOD parameters directly from the top parameter cell. It no longer applies a hidden notebook-only profile and it no longer remaps those visible knobs through `NOTEBOOK_OVERRIDES`. If you want different values, edit the cell directly and rerun it.

## Token Resolution

Notebook 2 and the direct-adapter notebooks resolve the Hugging Face token from these sources:

- `HF_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- matching Colab secrets when running inside Colab

The notebook validates the token before model access.

Notebook 0 and Notebook 2 resolve the GitHub push token from:

- `GH_TOKEN`
- `GITHUB_TOKEN`
- matching Colab secrets when running inside Colab

If auto-push is enabled, Notebook 0 uses that token to push `data/prepared_runtime_datasets/<dataset_key>/`, and Notebook 2 uses it after the repo mirror step.

The same token is also used by the notebook bootstrap when it needs to clone a private GitHub repo into the Colab runtime.

## What Notebook 2 Produces

Notebook 2 writes to three locations. Each one exists for a different reason. The artifact root now also includes a `guided/` folder so a new user can start from a short human-readable index before opening raw JSON/CSV files.

### 1. Local notebook output

```text
outputs/colab_notebook_training/
  <crop>/
    <part>/
      continual_sd_lora_adapter/
  artifacts/
  telemetry_runtime/
```

Use this when you want the immediate local result of the notebook run.

`artifacts/` contains the same artifact families as the workflow:

- `guided/`
- `training/`
- `validation/`
- `test/`
- `ood_benchmark/`
- `production_readiness.json`

Recommended reading order inside `artifacts/`:

- `guided/00_start_here.md`
- `guided/01_run_overview.json`
- `guided/02_file_catalog.json`

These files organize the raw artifacts; they do not replace them.

### 2. Repo mirror

Notebook 2 mirrors non-checkpoint outputs into:

```text
runs/<RUN_ID>/
  notebooks/2_interactive_adapter_training.executed.ipynb
  outputs/colab_notebook_training/
  telemetry/
  checkpoint_state/
```

Use this when you want one self-contained record of a notebook run inside the repo workspace.

Important detail:

- `checkpoint_state/` keeps checkpoint metadata plus only the mirrored best checkpoint
- the mirrored best checkpoint is stored once; the manifests point to that directory instead of maintaining a second `best/` copy
- the actual rolling checkpoint tree stays under the repo-local telemetry checkpoint root

Optional current behavior:

- if `AUTO_PUSH_TO_GITHUB` is enabled and `GH_TOKEN` or `GITHUB_TOKEN` is available, Notebook 2 commits and pushes `runs/<RUN_ID>/` after the mirror step
- the auto-push commit is scoped to `runs/<RUN_ID>/`, so unrelated staged repo changes are not included
- the auto-push helper skips `.pt` checkpoint blobs, so large resume weights stay out of the normal GitHub history

### 3. Repo-local telemetry runtime

Current Notebook 2 telemetry runtime layout:

```text
outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/
  checkpoints/
  artifacts/
    guided/
    training/
    validation/
    test/
    ood_benchmark/
    adapter_export/
      <crop>/
        <part>/
          continual_sd_lora_adapter/
  events.jsonl
  runtime.log
  latest_status.json
  summary.json
  latest_checkpoint.json
  best_checkpoint.json
  checkpoint_index.json
```

Use this for the active run's notebook logs, adapter-export mirror, and checkpoint recovery while the Colab runtime is alive.

Important durability guardrail:

- Notebook 2 no longer requests Google Drive mount access by default
- use the repo mirror and `AUTO_PUSH_TO_GITHUB=True` when you need artifacts to survive Colab runtime reset
- the auto-push helper skips `.pt` checkpoint blobs, so rolling resume checkpoints are runtime-local unless you export them through a separate storage path

Important current-state note:

- Notebook 2 currently exports adapter assets to `artifacts/adapter_export/<crop>/<part>/continual_sd_lora_adapter/`
- telemetry manifests point to the best rolling checkpoint; they do not maintain a second duplicated best-checkpoint tree

## How To Read The Main Training Artifacts

### `training/`

This folder helps you understand how training behaved over time.

- `results.png`: visual overview of loss, accuracy, throughput, and other curves
- `history.json` and `history.csv`: per-epoch history
- `batch_metrics.csv`: batch-level record
- `summary.json`: compact summary of the run

### `validation/` and `test/`

These folders help you understand how the model performed on known-class data.

- `classification_report.txt` and `.json`: precision, recall, and F1-style breakdowns
- `per_class_metrics.csv`: per-class summary
- `confusion_matrix.csv` and `.png`: where predictions are confused
- `metric_gate.json`: split-local threshold checks
- `ood_type_breakdown.json`: optional real-OOD slice metrics keyed by the top-level folder under `ood/`
- `ood_evidence_summary.json`: pooled OOD metrics plus a compact summary of any discovered real-OOD slices

### `ood_benchmark/`

This folder appears when the fallback held-out OOD benchmark runs.

- `summary.json`: overall benchmark verdict
- `per_fold.csv`: one row per held-out class fold
- `progress.json`: latest persisted fold and stage while the benchmark is running
- `folds/<held_out_class>/failure.json` and `failure_traceback.txt`: saved Python failure details when a fold raises an exception

### `production_readiness.json`

This is the final deployment verdict. Read this before deployment.

## OOD And Readiness In Notebook 2

Notebook 2 does more than train a classifier. It also:

- calibrates OOD state after training
- evaluates real `ood/` data when present
- otherwise runs the held-out fallback benchmark automatically
- writes the final readiness artifact to `production_readiness.json`

Important guardrails:

- the best in-memory weights are restored before final calibration, evaluation, and adapter export
- if `val` was used for OOD calibration, an isolated `test/` split is needed for a final readiness verdict
- if no real `ood/` split exists, the fallback benchmark may retrain once per known class
- if your real goal is set-valued classification, prefer `conformal_method: "aps"` or `"raps"` instead of the threshold mode

Naming note:

- the repo label `SURE+` means `SURE+/DS-F1-inspired double scoring`
- it should not be read as a claim that Notebook 2 exports a paper-faithful reimplementation of a separately named SURE+ method

Read [ood_readiness_guide.md](ood_readiness_guide.md) for the full readiness logic.

## BER Rollout

BER is currently an optional training-only experiment.
The shipped default keeps BER disabled and uses LogitNorm instead.

Recommended comparison workflow:

1. keep the crop, seed, class set, and evidence source fixed
2. run the shipped BER-off LogitNorm baseline
3. run BER candidates with `loss_name: "cross_entropy"` and different lambdas
4. compare the same artifact families across runs

Most useful files for that comparison:

- `validation/metric_gate.json`
- `test/metric_gate.json`
- `ood_benchmark/summary.json`
- `production_readiness.json`

Optional helper command:

```powershell
.\scripts\python.cmd scripts/evaluate_ber_rollout.py <baseline_artifact_root> <candidate_artifact_root> [...]
```

## Notebook 3 Adapter Smoke Test

Notebook 3 is for direct adapter validation after training or export.

It does not use the router.

Use it to:

- inspect adapter metadata
- confirm the adapter bundle still loads
- run one-image prediction
- run folder-level sanity prediction with per-file error reporting

### What paths Notebook 3 accepts

Current accepted `ADAPTER_DIR` patterns include:

- direct asset dir: `.../continual_sd_lora_adapter/`
- parent export dir: `outputs/colab_notebook_training/`
- current telemetry export dir: `outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/artifacts/adapter_export/`
- direct metadata file: `.../adapter_meta.json`

Current accepted `ADAPTER_ROOT` pattern:

- parent of crop folders, usually `models/adapters/`

Important caveat:

- current helper resolution accepts the `adapter_export/` dir or the `continual_sd_lora_adapter/` folder itself for the current export layout
- `ADAPTER_ROOT` is for deployed adapters under `models/adapters/<crop>/...`, not for telemetry run roots

### Image input rules

- `IMAGE_PATH` must be one image file
- `BATCH_IMAGE_DIR` must be one directory of image files
- `CROP_NAME` can stay `None` when the adapter path itself already implies the crop or nearby metadata provides it

## Notebook 4 Minimal Smoke UI

Notebook 4 uses the same adapter discovery and prediction helpers as Notebook 3, but presents them through a small widget UI.

Use Notebook 4 when you want:

- a quick manual adapter selection dropdown
- image upload without editing path variables
- a smaller direct-adapter sanity check surface

Use Notebook 3 when you want:

- the fuller documented direct-adapter workflow
- more explicit path handling
- a surface that mirrors the maintained smoke-test helper flow step by step

## Deployment Handoff

Router inference looks for adapters here by default:

```text
models/adapters/<crop>/<part>/continual_sd_lora_adapter/
```

You can deploy from any of these outputs:

- `outputs/colab_notebook_training/<crop>/<part>/continual_sd_lora_adapter/`
- `runs/<RUN_ID>/outputs/colab_notebook_training/<crop>/<part>/continual_sd_lora_adapter/`
- `outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/artifacts/adapter_export/<crop>/<part>/continual_sd_lora_adapter/`

If you want a different storage location, pass `--adapter-root` to the inference surface.

## Common Beginner Mistakes

### Mistake 1: skipping Notebook 0 before Notebook 2

Notebook 2 expects the prepared runtime layout from Notebook 0, not a fresh flat class-root dataset.

### Mistake 2: confusing local output with deployed adapters

Training output folders are not automatically deployed. Router inference only finds adapters under the configured adapter root unless you override it.

### Mistake 3: reading only `validation/metric_gate.json`

Use `production_readiness.json` for the final deployment verdict.

### Mistake 4: pointing Notebook 3 at the wrong folder

If you are testing a telemetry export, point `ADAPTER_DIR` at the export folder or the `continual_sd_lora_adapter/` folder inside it.

## Validation Commands

Start with the maintained validation commands from [../../README.md](../../README.md). For notebook-only changes, the narrowest common subset is:

```powershell
.\scripts\python.cmd scripts/validate_notebook_imports.py
.\scripts\python.cmd scripts/validate_config_schema.py
.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root data\<your_flat_class_root>
pytest tests/colab/test_smoke_training.py -q
```

## Repo Hygiene

Keep these out of git:

- `runs/`
- `models/adapters/`
- `outputs/`

`colab_notebooks/requirements_colab.txt` should stay in the repo. It is a wrapper around the canonical root `requirements_colab.txt`.






