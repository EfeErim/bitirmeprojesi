# Documentation Map

Use this file as the routing index. If you are new, start with [../README.md](../README.md).

This repo is intentionally narrow. The fastest handoff path is:

1. [../README.md](../README.md)
2. [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md)
3. [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)
4. [architecture/overview.md](architecture/overview.md)

## Quick Reading Paths

1. Full picture: [../README.md](../README.md) -> [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md) -> [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md) -> [architecture/overview.md](architecture/overview.md)
2. Colab training only: [../README.md](../README.md) -> [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md)
3. Deployment readiness only: [../README.md](../README.md) -> [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md)
4. Code map only: [../README.md](../README.md) -> [architecture/overview.md](architecture/overview.md)

## Maintained Docs

- [../README.md](../README.md): project scope, entrypoints, dataset contracts, quick-start commands, output paths
- [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md): Notebook 0/2/3 flow, Notebook 4 relation, dataset materialization, notebook caveats
- [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md): OOD and readiness policy, `metric_gate.json` vs `production_readiness.json`
- [architecture/overview.md](architecture/overview.md): training and inference flow, config flow, artifact contracts, file responsibility map
- [architecture/code_organization_map.md](architecture/code_organization_map.md): repo-wide source boundaries for `src/`, scripts, notebook helpers, tests, configs, and docs
- [project_completion_plan.md](project_completion_plan.md): June 2026 scope freeze, milestones, deliverables, and definition of done
- [eight_surface_completion_plan.md](eight_surface_completion_plan.md): follow-up plan to fix all eight final crop/part surfaces after the full M2 run
- [architecture/taxonomy_prototype_router_plan.md](architecture/taxonomy_prototype_router_plan.md): automatic taxonomy + prototype router plan so new plant additions do not require manual decision-surface tuning
- [demo_checklist.md](demo_checklist.md): Notebook 8 demo image matrix, expected behavior, result logging, and pass criteria
- [handoff_guide.md](handoff_guide.md): company/evaluator handoff path, supported target labels, runtime notes, and known limitations
- [final_validation_checklist.md](final_validation_checklist.md): final freeze validation commands, required checks, optional checks, and status template
- [presentation_outline.md](presentation_outline.md): Turkish narration with English technical terms, slide plan, visuals, and rehearsal checklist
- [architecture/tomato_leaf_review_gate_promotion_validation.md](architecture/tomato_leaf_review_gate_promotion_validation.md): `tomato__leaf` report-only review-gate pilot and runtime promotion gates
- [SOTA_AUTOMATION_GUIDE.md](SOTA_AUTOMATION_GUIDE.md): continuous automation strategy grounded in literature; recommended cascade by tier and ROI
- [roi_ablation_memory.md](roi_ablation_memory.md): current ROI/bbox/router/adapter ablation state, decisions, and next-step plan

## Fast Reference

- Deployability verdict file: `production_readiness.json`
- Canonical training entrypoint: `src/workflows/training.py`
- Canonical inference entrypoint: `src/workflows/inference.py`
- Config schema versioning: `config_schema_version` in `config/base.json` and `config/colab.json`; bump it when you make breaking config changes.
- Router payload contract: `src/shared/contracts.py`
- Auto router-to-adapter notebook: `colab_notebooks/8_auto_router_adapter_prediction.ipynb`
- ROI/bbox evidence notebook: `colab_notebooks/16_ablation_dual_view_inference.ipynb` (multi-adapter target discovery)
- ROI/bbox ablation memory: `docs/roi_ablation_memory.md`
- ROI ablation reports: `docs/ablation_results/<condition>/`
- Validation-only notebook surfaces: `colab_notebooks/6_train_all_continual_sd_lora_adapters.ipynb`, `colab_notebooks/7_ood_oe_quality.ipynb`
- Optional plantness input guard config: `inference.input_guard`
- Default adapter deployment path: `models/adapters/<crop>/<part>/continual_sd_lora_adapter/`
- Local run registry path: `runs/_index/`
- Repo-wide source-boundary audit: `.\scripts\python.cmd scripts/audit_code_organization.py`
- Dataset-aware cohort command:

```powershell
.\scripts\python.cmd scripts/optimize_training_runs.py --dataset-lineage-key <dataset_key>::<split_manifest_sha256> --crop-name <crop> --part-name <part>
```

Optimizer status note:

- Notebook 2 enables adapter/cohort-scoped Bayesian recommendations by default with `ENABLE_BAYESIAN_OPTIMIZATION=True`.
- The standalone registry and optimizer CLIs still require an explicit Bayesian flag before writing `bayesian_recommendations.json`.

## Generated And Local-Only Surfaces

The following paths are generated, transient, or local-working directories and should not be treated as canonical source:

- `runs/`
- `models/adapters/`
- `outputs/`
- `data/prepared_runtime_datasets/`
- `.runtime_tmp/`
- `.codex/`
- `.tmp.drivedownload/`
- `.tmp.driveupload/`

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
- [architecture/code_organization_map.md](architecture/code_organization_map.md)
- [architecture/data_augmentation_leakage_prevention.md](architecture/data_augmentation_leakage_prevention.md)
- [architecture/agricultural_augmentation_domain_guide.md](architecture/agricultural_augmentation_domain_guide.md)
- [architecture/augmentation_implementation_guide.md](architecture/augmentation_implementation_guide.md)
- [architecture/router_performance_literature_review.md](architecture/router_performance_literature_review.md)
- [architecture/notebook8_router_performance_plan.md](architecture/notebook8_router_performance_plan.md)
- [architecture/router_calibration_recovery_plan.md](architecture/router_calibration_recovery_plan.md)
- [architecture/tomato_leaf_review_gate_promotion_validation.md](architecture/tomato_leaf_review_gate_promotion_validation.md)
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
- `colab_notebooks/requirements_colab.txt` is a wrapper used by notebook-local bootstrap.
- Notebook 2 telemetry export path: `outputs/colab_notebook_training/telemetry_runtime/telemetry/<RUN_ID>/artifacts/adapter_export/<crop>/<part>/continual_sd_lora_adapter/`.
