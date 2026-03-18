# Colab Training Manual

This guide is written for someone who may be opening the AADS notebooks for the first time.

It covers the maintained Colab surfaces:

- Notebook 0: `colab_notebooks/0_grouped_dataset_preparation.ipynb`
- Notebook 2: `colab_notebooks/2_interactive_adapter_training.ipynb`
- Notebook 3: `colab_notebooks/3_adapter_smoke_test.ipynb`

Notebook 0 is the audit-first data-preparation surface. Notebook 2 is the training surface and can materialize the grouped runtime dataset before training when you start from a flat class-root dataset. Notebook 3 checks an already exported adapter directly.

If you are brand new to the repo, read [../../README.md](../../README.md) first.

## What You Will Do In These Notebooks

### Notebook 0

Notebook 0 audits a flat class-root dataset, groups likely duplicate and augmentation families with DINOv3 and BioCLIP-2.5 similarity signals, writes review artifacts, and can materialize a prepared runtime dataset.

### Notebook 2

Notebook 2 can now do one of two things:

- start from a flat class-root dataset, run grouped prep/materialization, and then train
- train directly from a prepared runtime dataset root produced earlier

### Notebook 3

Notebook 3 loads one saved adapter directly so you can inspect it and run a quick prediction without the router.

## Important Terms

- `flat class-root dataset`: the simple folder layout you prepare by hand for Notebook 2
- `runtime dataset`: the split layout used by workflow training; it can be created by Notebook 2 from a flat dataset or consumed directly in runtime mode
- `adapter bundle`: the saved training output that contains model weights, metadata, and OOD state
- `telemetry`: the logs and mirrored artifacts saved during a notebook run
- `OOD`: "out of distribution," meaning the image may not belong to the supported disease classes
- `readiness`: the final deployment verdict written to `production_readiness.json`

## Before You Start

Make sure you have these things ready:

1. A dataset for one crop.
2. A Colab runtime that has GPU access if you intend to use the default `cuda` path.
3. A Hugging Face token if the required models need authenticated access.
4. Enough storage for outputs, telemetry, and checkpoints if you enable Drive logging.

Private repo note:

- if the notebook starts outside the repo workspace and this GitHub repo is private, set `GH_TOKEN` or `GITHUB_TOKEN` as a Colab secret before running the bootstrap cell
- alternatively, mount Drive and set `AADS_REPO_ROOT` to an existing checkout so the notebook does not need to clone

Important current behavior:

- requesting `device="cuda"` fails immediately when CUDA is unavailable
- the notebook validates the dataset before training starts
- Notebook 2 accepts either `DATASET_LAYOUT_MODE="class_root"` or `DATASET_LAYOUT_MODE="runtime"`
- `class_root` now runs grouped prep and runtime-dataset materialization before training
- `runtime` consumes an already prepared runtime dataset without re-running prep

## Notebook 2 In Plain English

This is the current Notebook 2 training flow from start to finish when you use `DATASET_LAYOUT_MODE="class_root"`:

1. find or initialize the repo workspace
2. install notebook requirements
3. mount Google Drive when available
4. resolve a Hugging Face token from environment variables or Colab secrets
5. validate a flat class-root dataset
6. run grouped duplicate-aware prep
7. materialize a runtime dataset under `data/prepared_runtime_datasets/<crop>/`
8. train the continual SD-LoRA adapter
8. restore the best model state
9. calibrate OOD
10. write validation and test artifacts
11. use real `ood/` data when present, otherwise run the held-out fallback benchmark automatically
12. write `production_readiness.json`
13. mirror outputs into `runs/<RUN_ID>/`
14. optionally auto-push the mirrored run record to GitHub
15. optionally auto-disconnect the Colab runtime after final exports succeed

Important recommendation:

- use Notebook 0 first when you want to inspect the audit artifacts before training
- use Notebook 2 `class_root` mode when you want one notebook to prepare and train in one pass
- use Notebook 2 `runtime` mode when the prepared runtime dataset already exists

## Notebook 0 In Plain English

This is the current Notebook 0 flow from start to finish:

1. find or initialize the repo workspace
2. install notebook requirements
3. mount Google Drive when available
4. resolve a Hugging Face token from environment variables or Colab secrets
5. scan a flat class-root dataset
6. normalize class names against the crop taxonomy when possible
7. audit exact duplicates, perceptual-hash neighbors, and DINOv3/BioCLIP similarity families
8. write review artifacts and a grouped split manifest
9. optionally materialize a prepared runtime dataset under `data/prepared_runtime_datasets/<crop>/` when you want Notebook 0 to complete the full prep flow itself

## The Dataset Format Notebook 2 Accepts

Notebook 2 `class_root` mode expects the simplest possible input layout:

```text
<root>/<class>/<images>
```

Example:

```text
data/tomato_flat/
  healthy/
    img001.jpg
    img002.jpg
  early_blight/
    img003.jpg
    img004.jpg
  late_blight/
    img005.jpg
```

That means:

- the root folder contains one folder per class
- each class folder contains image files
- you should not pre-create `train`, `val`, or `test` folders for Notebook 2

### Validate the dataset before training

Use:

```powershell
.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root data\tomato_flat
```

That validator checks:

- whether the root exists
- whether it is a directory
- whether it uses the flat class-root shape
- how many images exist per class
- what the estimated split counts will look like

## What Notebook 2 Builds From That Dataset

Notebook 2 `class_root` mode materializes the grouped runtime layout used by the workflow:

```text
data/runtime_notebook_datasets/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
```

If you also provide unknown examples, the runtime layout can include:

```text
data/runtime_notebook_datasets/<crop>/ood/*
```

Important distinction:

- the flat class-root notebook input is only for supported disease classes
- the real `ood/` pool is separate unknown-input evidence, not another class
- do not create an `ood` class folder inside the flat notebook input root
- if you maintain a real OOD pool, it belongs under the runtime dataset as `data/runtime_notebook_datasets/<crop>/ood/`

The generated runtime dataset includes:

- `split_manifest.json`
- `_split_metadata.json`

Contract reminder:

- Notebook 2 `class_root` mode accepts the flat class-root layout and now runs grouped prep before training
- Notebook 2 `runtime` mode and `TrainingWorkflow.run(...)` expect the runtime split layout
- Notebook 0 remains the maintained audit-first surface when you want to inspect or fix prep issues before training

## How The Split Is Created

The dataset materialization step is handled by `scripts/colab_dataset_layout.py`.

Current behavior:

- the notebook uses an 80/10/10-style split policy with small-class safeguards
- the runtime split names are `continual`, `val`, and `test`
- Notebook 2 now uses the automatic materialization strategy, which prefers links over full copies on Colab and other non-Windows systems
- on Windows, materialization defaults to copying files instead of symlinks

## Prepare Trustworthy Evaluation Data Before Auto-Splitting

Notebook 2 auto-splitting is convenient, but it is only an image-level random split after the flat class-root dataset has already been assembled.

Use the `class_root` path when you want Notebook 2 to perform grouped prep and then train in one pass. It is most appropriate when:

- each class folder contains only real, independent images
- you are not mixing multiple public sources into one flat pool
- you are not mixing offline augmentations or synthetic images into the same evaluation candidate set
- you accept a random image-level split as the benchmark

Do not treat Notebook 2 auto-splitting as the main benchmark path when your flat dataset was built by merging public sources, adding offline augmentations, including GAN-generated images, or mixing repeated captures of the same plant, session, or original photo family.

Why this matters:

- if you merge sources first and split later, source style can leak across `continual`, `val`, and `test`
- if offline augmentations or synthetic variants are present before splitting, sibling images can land in different splits
- if repeated captures of the same leaf or plant exist, the random split can produce an unrealistically easy test set

This is consistent with duplicate-aware image-benchmark guidance such as ciFAIR and with literature warning that PlantVillage-style plant-disease datasets can contain shortcut cues and domain bias instead of deployment-like variability.[^cifair] [^plantvillage-bias]

### Recommended data-preparation workflow

1. Preserve provenance before merging.

   Keep a manifest for every image with fields such as:

   - `source_dataset`
   - `source_subset`
   - `original_image_id`
   - `augmentation_family_id`
   - `synthetic_flag`
   - `capture_group_id`

2. Deduplicate before final splitting.

   Run both:

   - exact duplicate checks such as file hash comparisons
   - near-duplicate checks such as perceptual-hash or visual-family clustering

   Remove exact duplicates from the benchmark set and keep near-duplicate families together in one split.

3. Split by group, not by image.

   Keep these together:

   - one original image and all of its offline augmentations
   - one synthetic image family and its close variants
   - images from the same plant, same capture session, or same collection event when that metadata exists

4. Prefer real images for authoritative `val` and `test`.

   A practical rule is:

   - `continual/` may include curated augmentations when you intentionally want them for training
   - `val/` and `test/` should prefer real, non-synthetic images
   - GAN-generated images should be train-only or evaluated in a separate ablation, not mixed silently into the authoritative test split

5. Keep source-aware evaluation when sources are merged.

   Preferred order:

   - source-held-out test split
   - source-balanced grouped split
   - image-level random split only as a convenience baseline

   If your data comes from multiple public datasets, a held-out source or at least a source-balanced grouped test split is much more trustworthy than a random split after merge.

6. Materialize the runtime dataset through Notebook 0 when you want audit-first control.

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
   .\scripts\python.cmd -m src.app.cli training tomato data\runtime_notebook_datasets outputs\training_run --config-env colab
   ```

7. Keep OOD separate from classification splitting.

   Do not mix unknown images into the supported class-root dataset. Keep `ood/` as a separate pool under the runtime dataset and match it to the real deployment contract.

### Practical rules for merged public tomato-leaf datasets

If you combine multiple public tomato-leaf datasets, use these rules:

- never merge raw images, augmented variants, and GAN samples into one flat folder and trust the notebook random split as the main benchmark
- keep all derivatives of one original image in the same split
- keep one clean held-out test slice of real leaf images that were never used to generate training augmentations
- report whether the test slice is source-held-out, source-balanced, or only random-after-merge
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

### When source separation is uncertain

Public plant-disease datasets often do not provide enough metadata to prove that all sources are fully independent. Different dataset packages may reuse the same original photo, a resized copy, a crop, or an offline augmentation family.

Do not claim full source independence unless you can actually verify it. Instead, use a conservative protocol that reduces leakage risk as much as possible:

1. preserve whatever provenance you do have, such as dataset package, subset, filename lineage, download origin, or augmentation history
2. run exact duplicate detection
3. run near-duplicate detection
4. group suspiciously similar images together even if you are not fully certain they share one original
5. split by those groups instead of by individual images
6. describe the evaluation honestly as source-held-out, source-balanced, grouped, or random-after-merge

This is still much more trustworthy than merging everything first and relying on an image-level random split.

### A conservative protocol when lineage is incomplete

If you cannot fully reconstruct which augmented images came from which originals, use the most conservative practical workflow you can support:

1. start from the rawest available image export rather than the already mixed training folder
2. keep dataset-package boundaries visible in folder names or manifests
3. run duplicate and near-duplicate audits before finalizing the runtime split
4. keep flagged duplicate families in one split even when that lowers the apparent benchmark score
5. reserve a clean real-image test slice that excludes synthetic and likely augmented families
6. treat any remaining random-after-merge result as a convenience baseline, not as the headline claim

When provenance is weak, the goal is not to prove perfect independence. The goal is to remove obvious leakage paths and make the benchmark defensible.

### When Notebook 2 auto-splitting is still acceptable

Notebook 2 auto-splitting is still useful when you want:

- a fast smoke run to validate training and export surfaces
- an early baseline before building a stricter evaluation split
- a convenience experiment on a small, clean, single-source dataset without augmentation-family leakage

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
- `training.continual.optimization.label_smoothing`
- `training.continual.optimization.scheduler`
- `training.continual.early_stopping`

### Data loading

These affect image loading and error tolerance:

- `training.continual.data.sampler`
- `training.continual.data.loader_error_policy`
- `training.continual.data.target_size`
- `training.continual.data.cache_size`
- `training.continual.data.cache_train_split`
- `training.continual.data.validate_images_on_init`

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
- the trainer starts with the concrete detector path on `"ensemble"` until OOD evidence exists
- when real `ood/` data or the held-out fallback benchmark is available, the workflow auto-selects the concrete winning method and exports that chosen method into the adapter
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
- `require_ood_for_gate` controls whether final production readiness fails when OOD evidence is missing or insufficient

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
- live batch-status telemetry is throttled so Drive does not get a tiny `latest_status.json` rewrite on every batch

## Notebook-Only Top Cell Toggles

Notebook 2 also exposes a small set of notebook-level toggles:

- `CACHE_TRAIN_SPLIT`
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

Notebook 2 resolves the Hugging Face token from these sources:

- `HF_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- matching Colab secrets when running inside Colab

The notebook validates the token before model access.

Notebook 2 also resolves the GitHub push token from:

- `GH_TOKEN`
- `GITHUB_TOKEN`
- matching Colab secrets when running inside Colab

If auto-push is enabled, the notebook uses that token after the repo mirror step.

The same token is also used by the notebook bootstrap when it needs to clone a private GitHub repo into the Colab runtime.

## What Notebook 2 Produces

Notebook 2 writes to three locations. Each one exists for a different reason.

### 1. Local notebook output

```text
outputs/colab_notebook_training/
  continual_sd_lora_adapter/
  artifacts/
```

Use this when you want the immediate local result of the notebook run.

`artifacts/` contains the same artifact families as the workflow:

- `training/`
- `validation/`
- `test/`
- `ood_benchmark/`
- `production_readiness.json`

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
- the actual rolling checkpoint tree stays under the Drive checkpoint root

Optional current behavior:

- if `AUTO_PUSH_TO_GITHUB` is enabled and `GH_TOKEN` or `GITHUB_TOKEN` is available, Notebook 2 commits and pushes `runs/<RUN_ID>/` after the mirror step
- the auto-push commit is scoped to `runs/<RUN_ID>/`, so unrelated staged repo changes are not included
- the auto-push helper skips `.pt` checkpoint blobs, so large resume weights stay out of the normal GitHub history

### 3. Drive telemetry

Current Drive telemetry layout:

```text
<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/
  checkpoints/
  artifacts/
    training/
    validation/
    test/
    ood_benchmark/
    adapter_export/
      continual_sd_lora_adapter/
  events.jsonl
  runtime.log
  latest_status.json
  summary.json
  latest_checkpoint.json
  best_checkpoint.json
  checkpoint_index.json
```

Use this when you need durable notebook logs and checkpoint recovery across sessions.

Important durability guardrail:

- Drive telemetry is only treated as durable when the target is backed by an actual mounted Google Drive path
- if Drive is not mounted yet, telemetry continues in the local spool instead of creating a fake `/content/drive/...` tree that would disappear with the runtime

Important current-state note:

- Notebook 2 currently exports adapter assets to `artifacts/adapter_export/continual_sd_lora_adapter/`
- Drive telemetry manifests point to the best rolling checkpoint; they do not maintain a second duplicated best-checkpoint tree

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

Recommended comparison workflow:

1. keep the crop, seed, class set, and evidence source fixed
2. run a BER-off baseline
3. run BER candidates with different lambdas
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
- current Drive export dir: `.../telemetry/<RUN_ID>/artifacts/adapter_export/`
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

## Deployment Handoff

Router inference looks for adapters here by default:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

You can deploy from any of these outputs:

- `outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `runs/<RUN_ID>/outputs/colab_notebook_training/continual_sd_lora_adapter/`
- `<AADS_DRIVE_LOG_ROOT>/telemetry/<RUN_ID>/artifacts/adapter_export/continual_sd_lora_adapter/`

If you want a different storage location, pass `--adapter-root` to the inference surface.

## Common Beginner Mistakes

### Mistake 1: using a pre-split dataset as Notebook 2 input

Notebook 2 expects the flat class-root layout, not pre-made `train/val/test` folders.

### Mistake 2: confusing local output with deployed adapters

Training output folders are not automatically deployed. Router inference only finds adapters under the configured adapter root unless you override it.

### Mistake 3: reading only `validation/metric_gate.json`

Use `production_readiness.json` for the final deployment verdict.

### Mistake 4: pointing Notebook 3 at the wrong folder

If you are testing a telemetry export, point `ADAPTER_DIR` at the export folder or the `continual_sd_lora_adapter/` folder inside it.

## Validation Commands

Run these before or after notebook-related changes:

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
