# Documentation Map

Use this file as a routing index. If you are new, start with [../README.md](../README.md).

## Quick Reading Paths

1. Full picture: [../README.md](../README.md) -> [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md) -> [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md) -> [architecture/overview.md](architecture/overview.md)
2. Colab training only: [../README.md](../README.md) -> [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md) -> [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)
3. Deployment readiness only: [../README.md](../README.md) -> [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)
4. Code map only: [../README.md](../README.md) -> [architecture/overview.md](architecture/overview.md)

## Maintained Docs

- [../README.md](../README.md): project scope, entrypoints, dataset contracts, quick-start commands, output paths
- [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md): Notebook 0/2/3 flow, Notebook 4 relation, dataset materialization, notebook caveats
- [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md): OOD and readiness policy, `metric_gate.json` vs `production_readiness.json`
- [architecture/overview.md](architecture/overview.md): training and inference flow, config flow, artifact contracts, file responsibility map

## Fast Reference

- Deployability verdict file: `production_readiness.json`
- Canonical training entrypoint: `src/workflows/training.py`
- Canonical inference entrypoint: `src/workflows/inference.py`
- Router payload contract: `src/shared/contracts.py`
- Default adapter deployment path: `models/adapters/<crop>/<part>/continual_sd_lora_adapter/`
- Local run registry path: `runs/_index/`
- Dataset-aware cohort command:

```powershell
.\scripts\python.cmd scripts/optimize_training_runs.py --dataset-lineage-key <dataset_key>::<split_manifest_sha256> --crop-name <crop> --part-name <part>
```

Optimizer status note:

- Bayesian proposals are disabled.
- Notebook 2 ignores `OPTIMIZATION_CAMPAIGN_MODE` (`continue`/`stop`) and records disabled campaign status.

Dataset formats:

Notebook 2 prepared runtime dataset:

```text
data/prepared_runtime_datasets/<dataset_key>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

Workflow and CLI runtime split layout:

```text
<data_dir>/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

## Source Of Truth Order

When docs, notes, and generated artifacts disagree, prefer:

1. `src/workflows/`
2. `src/core/config_manager.py`
3. `src/shared/contracts.py`
4. `src/training/services/`
5. `scripts/`
6. `tests/`
7. archived notes

## Additional Notes

Supporting references:

- [architecture/ood_recommendation.md](architecture/ood_recommendation.md)
- [architecture/unknown_disease_rejection.md](architecture/unknown_disease_rejection.md)

Archive (historical only):

- [archive/experimental_leave_one_class_out_ood.md](archive/experimental_leave_one_class_out_ood.md)

Tracked-vs-generated boundaries and validation commands:

- Use [../README.md](../README.md) as canonical guidance.
- Use [../data/README.md](../data/README.md) for dataset policy.

Path confusion reminders:

- `requirements_colab.txt` at repo root is the canonical Colab dependency list.
- `colab_notebooks/requirements_colab.txt` is a wrapper used by notebook-local bootstrap.
- Notebook 2 telemetry export path: `outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/artifacts/adapter_export/<crop>/<part>/continual_sd_lora_adapter/`.
