# Documentation Map

Use this file as the routing index. If you are new, start with [../README.md](../README.md).

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
- [SOTA_AUTOMATION_GUIDE.md](SOTA_AUTOMATION_GUIDE.md): continuous automation strategy grounded in literature; recommended cascade by tier and ROI

## Fast Reference

- Deployability verdict file: `production_readiness.json`
- Canonical training entrypoint: `src/workflows/training.py`
- Canonical inference entrypoint: `src/workflows/inference.py`
- Config schema versioning: `config_schema_version` in `config/base.json` and `config/colab.json`; bump it when you make breaking config changes.
- Router payload contract: `src/shared/contracts.py`
- Auto router-to-adapter notebook: `colab_notebooks/8_auto_router_adapter_prediction.ipynb`
- Presentation recording demo: `colab_notebooks/9_presentation_recording_demo.ipynb`
- ROI ablation notebooks: `colab_notebooks/10_ablation_full_image_baseline.ipynb` through `colab_notebooks/14_ablation_mixed_full_roi_training.ipynb`
- ROI ablation reports: `docs/ablation_results/<condition>/`
- Validation-only notebook surfaces: `colab_notebooks/6_train_all_continual_sd_lora_adapters.ipynb`, `colab_notebooks/7_ood_oe_quality.ipynb`
- Optional plantness input guard config: `inference.input_guard`
- Default adapter deployment path: `models/adapters/<crop>/<part>/continual_sd_lora_adapter/`
- Local run registry path: `runs/_index/`
- Dataset-aware cohort command:

```powershell
.\scripts\python.cmd scripts/optimize_training_runs.py --dataset-lineage-key <dataset_key>::<split_manifest_sha256> --crop-name <crop> --part-name <part>
```

Optimizer status note:

- Notebook 2 enables adapter/cohort-scoped Bayesian recommendations by default with `ENABLE_BAYESIAN_OPTIMIZATION=True`.
- The standalone registry and optimizer CLIs still require an explicit Bayesian flag before writing `bayesian_recommendations.json`.

Generated / local-only surfaces:

- `runs/`, `models/adapters/`, `outputs/`, `data/prepared_runtime_datasets/`, and `.runtime_tmp/` are generated or local-working directories and should not be treated as canonical source.

Dataset formats:

Notebook 2 prepared runtime dataset:

```text
data/prepared_runtime_datasets/<dataset_key>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
  oe/*
```

Runtime split-leakage check:

```powershell
.\scripts\python.cmd scripts/validate_dataset_layout.py --root data/prepared_runtime_datasets/<dataset_key> --check-leakage
```

Workflow and CLI runtime split layout:

```text
<data_dir>/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
  oe/*
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

- [architecture/production_hardening_plan.md](architecture/production_hardening_plan.md)
- [architecture/data_augmentation_leakage_prevention.md](architecture/data_augmentation_leakage_prevention.md)
- [architecture/agricultural_augmentation_domain_guide.md](architecture/agricultural_augmentation_domain_guide.md)
- [architecture/augmentation_implementation_guide.md](architecture/augmentation_implementation_guide.md)
- [architecture/router_performance_literature_review.md](architecture/router_performance_literature_review.md)
- [architecture/notebook8_router_performance_plan.md](architecture/notebook8_router_performance_plan.md)
- [architecture/router_calibration_recovery_plan.md](architecture/router_calibration_recovery_plan.md)
- [architecture/ood_recommendation.md](architecture/ood_recommendation.md)
- [architecture/unknown_disease_rejection.md](architecture/unknown_disease_rejection.md)
- [architecture/plantness_input_guard_prompt_groups.md](architecture/plantness_input_guard_prompt_groups.md)

Archive (historical only):

- [archive/experimental_leave_one_class_out_ood.md](archive/experimental_leave_one_class_out_ood.md)

Tracked-vs-generated boundaries and validation commands:

- Use [../README.md](../README.md) as canonical guidance.
- Use [../data/README.md](../data/README.md) for dataset policy.

Path confusion reminders:

- `requirements_colab.txt` at repo root is the canonical Colab dependency list.
- `requirements_presentation_colab.txt` is the lightweight Notebook 9 inference-only bootstrap profile.
- `colab_notebooks/requirements_colab.txt` is a wrapper used by notebook-local bootstrap.
- Notebook 2 telemetry export path: `outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/artifacts/adapter_export/<crop>/<part>/continual_sd_lora_adapter/`.
