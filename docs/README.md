# Documentation Map

This folder is organized for two kinds of readers:

- someone with zero project knowledge who needs a guided path
- someone who already knows the goal and needs the exact maintained contract

If you are new, start with the root [README.md](../README.md) first.

## Beginner Reading Paths

### I know nothing and want the full picture

Read in this order:

1. [../README.md](../README.md)
2. [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md)
3. [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)
4. [architecture/overview.md](architecture/overview.md)

### I only want to train an adapter in Colab

Read:

1. [../README.md](../README.md)
2. [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md)
3. [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)

### I only want to understand deployment readiness

Read:

1. [../README.md](../README.md)
2. [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)
3. [architecture/overview.md](architecture/overview.md)

### I need the code map

Read:

1. [../README.md](../README.md)
2. [architecture/overview.md](architecture/overview.md)

## Maintained Docs

### [../README.md](../README.md)

Use this when you need:

- the project purpose in plain language
- the maintained entrypoints
- the dataset formats
- the quick start commands
- the output and deployment paths
- the router-driven inference response shape, including the structured `router` summary block

### [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md)

Use this when you need:

- Notebook 0, Notebook 2, and Notebook 3 explained step by step
- the tracked Notebook 4 convenience UI explained in relation to Notebook 3
- the flat dataset input contract
- the runtime split layouts created by Notebook 0 or Notebook 2
- notebook outputs, telemetry, and adapter handoff
- Hugging Face token behavior and notebook-specific caveats

### [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)

Use this when you need:

- a beginner explanation of OOD
- the current readiness policy
- how to structure and curate a real `ood/` pool
- the difference between `metric_gate.json` and `production_readiness.json`
- the fallback held-out benchmark behavior
- the default readiness targets

### [architecture/overview.md](architecture/overview.md)

Use this when you need:

- the end-to-end training flow
- the end-to-end inference flow
- the current config flow
- the artifact writing path
- the canonical run-lineage and optimization-record artifacts
- the main file-to-responsibility map
- the canonical primary-detection rule and router retry semantics after startup failures

## Supporting Notes

These are reference notes, not the first documents a beginner should read:

- [architecture/ood_recommendation.md](architecture/ood_recommendation.md)
- [architecture/unknown_disease_rejection.md](architecture/unknown_disease_rejection.md)

## Archive

These notes are historical context only. Do not treat them as maintained methodology:

- [archive/experimental_leave_one_class_out_ood.md](archive/experimental_leave_one_class_out_ood.md)

## Fast Answers To Common Questions

### Which file tells me whether a trained adapter is deployable?

Read `production_readiness.json`. The readiness guide explains its fields.

### Which files tell me how one training run should be compared for optimization work?

Read:

- `training/experiment_manifest.json` for dataset lineage and run identity
- `training/optimization_record.json` for normalized parameters and objectives
- `runs/_index/latest_registry.json` for the local aggregate registry view

### What dataset format does Notebook 2 accept?

Flat class-root:

```text
<root>/<class>/<images>
```

### What dataset format does workflow or CLI training accept?

Runtime split layout:

```text
<data_dir>/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

### Where should deployed adapters live by default?

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```

### Which training file is canonical?

`src/workflows/training.py`

### Where is the local run registry written?

```text
runs/_index/
  trials.jsonl
  latest_registry.json
  pareto_inputs.json
```

Those files are now refreshed automatically on successful canonical training traceability writes when the run is mirrored under the repo `runs/` tree. The index script is still available for manual rebuilds.

### Which inference file is canonical?

`src/workflows/inference.py`

### Where is the router inference payload contract defined?

Read `src/shared/contracts.py` for the typed contract and [../README.md](../README.md) for the maintained user-facing explanation.

## Current Source Of Truth

When docs, older notes, or generated outputs disagree, prefer this order:

1. `src/workflows/`
2. `src/core/config_manager.py`
3. `src/shared/contracts.py`
4. `src/training/services/`
5. `scripts/`
6. `tests/`
7. archived methodology notes

## Validation Commands

Use the maintained command list from [../README.md](../README.md). That file is the canonical quick-start surface for validation commands and dependency-policy notes.

## Tracked Vs Local Files

Use [../README.md](../README.md) for the repo-wide tracked-vs-generated boundary and [../data/README.md](../data/README.md) for the maintained dataset policy.

Current checked-in dataset exceptions:

- `data/class_root_dataset/grape_fruit/`
- `data/class_root_dataset/grape_leaf/`
- `data/ood_dataset/grape_fruit_ood_trial/`

## Similar-Looking Paths That Often Cause Confusion

- `requirements_colab.txt` at the repo root is the canonical Colab dependency list.
- `colab_notebooks/requirements_colab.txt` is only a wrapper so notebook-local bootstrap can find the root file.
- Notebook 2 currently exports Drive adapter assets under `artifacts/adapter_export/continual_sd_lora_adapter/`.
