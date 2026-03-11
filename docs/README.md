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

### [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md)

Use this when you need:

- Notebook 2 and Notebook 3 explained step by step
- the flat dataset input contract
- the runtime split layout created by Notebook 2
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
- the main file-to-responsibility map

## Supporting Notes

These are reference notes, not the first documents a beginner should read:

- [architecture/ood_recommendation.md](architecture/ood_recommendation.md)
- [architecture/experimental_leave_one_class_out_ood.md](architecture/experimental_leave_one_class_out_ood.md)

## Fast Answers To Common Questions

### Which file tells me whether a trained adapter is deployable?

Read `production_readiness.json`. The readiness guide explains its fields.

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

### Which inference file is canonical?

`src/workflows/inference.py`

## Current Source Of Truth

When docs, older notes, or generated outputs disagree, prefer this order:

1. `src/workflows/`
2. `src/core/config_manager.py`
3. `src/shared/contracts.py`
4. `src/training/services/`
5. `scripts/`
6. `tests/`
7. historical architecture notes

## Validation Commands

Windows PowerShell should prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv`.

Dependency policy:

- root `requirements.txt` and `requirements_colab.txt` pin the maintained ML stack to validated minor lines
- CI checks both the pinned baseline and the newest patch releases within those same minor lines
- do not replace this with open-ended `latest` upgrades unless the repo surface has been revalidated

Use the narrow maintained validation surface first:

```powershell
.\scripts\python.cmd scripts/validate_notebook_imports.py
.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root <flat_class_root>
pytest tests/unit tests/colab/test_smoke_training.py -q
pytest tests/integration -q --runintegration
.\scripts\python.cmd scripts/benchmark_surfaces.py
```

## Tracked Vs Local Files

Tracked:

- `src/`
- `tests/`
- `scripts/`
- `config/`
- `docs/`
- root `README.md`
- `colab_notebooks/*.ipynb`
- root dependency files

Local or generated:

- `runs/<RUN_ID>/`
- `models/adapters/<crop>/continual_sd_lora_adapter/`
- `outputs/`
- `.runtime_tmp/`
- caches and virtual environments

## Similar-Looking Paths That Often Cause Confusion

- `requirements_colab.txt` at the repo root is the canonical Colab dependency list.
- `colab_notebooks/requirements_colab.txt` is only a wrapper so notebook-local bootstrap can find the root file.
- Notebook 2 currently exports Drive adapter assets under `artifacts/adapter_export/continual_sd_lora_adapter/`.
- Some older helpers also accept `artifacts/adapter/`.
