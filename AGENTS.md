# AADS v6 Agent Guide

This repo is intentionally narrow. The maintained product surface is audit-first dataset preparation, continual SD-LoRA adapter training, OOD readiness, router-driven inference, and Colab notebook wrappers for those same flows. This file is for coding agents working inside this repo.

## Source Of Truth

Read these before making repo-wide assumptions:

- `README.md`
- `docs/README.md`
- `docs/architecture/overview.md`
- `docs/user_guide/colab_training_manual.md`
- `docs/user_guide/ood_readiness_guide.md`

## Evidence Standard

- Treat repo docs and code as the source of truth for current product behavior.
- Back method choices, evaluation policy, threshold logic, data-curation guidance, and other substantive ML claims with literature when credible sources exist.
- Prefer primary sources: peer-reviewed papers, official standards, and first-party technical documentation. Reuse literature already cited in maintained docs before adding new references.
- Separate current repo behavior, literature-backed rationale, and engineering inference made to fit repo constraints. Do not present an implementation as paper-faithful unless the citation and code actually support that claim.
- If strong literature is not available, say so briefly and fall back to repo docs, tests, benchmarks, and measured behavior.
- When adding new literature-backed rationale, place it in maintained docs or succinct code comments near non-obvious logic instead of ad hoc notes.

## SOTA Practice Lens

Use current best practice as a decision filter, not as a license for broad rewrites:

- For ML reliability, prefer measured calibration, risk-coverage, false-accept, slice, and held-out OOD evidence over one-image anecdotes or confidence-looking scores.
- For OOD and reject behavior, preserve train/dev/test evidence separation, keep Outlier Exposure separate from final OOD evidence, and report the deployment-relevant failure mode, especially FPR on realistic unknowns.
- For router and adapter thresholds, calibrate on repo-local evaluation surfaces before changing defaults. A higher forced-label accuracy is not an improvement if abstention quality or negative false accepts regress.
- For software changes, follow secure and reproducible engineering practice: narrow diffs, explicit contracts, dependency caution, generated-artifact boundaries, deterministic validation, and traceable artifacts.
- For tests, prefer the smallest executable check that proves the contract. Use property-based, metamorphic, or mutation-style reasoning when exact-output examples are weak for the touched logic.

Literature and standard anchors already reflected in repo docs include calibration, OOD detection, Outlier Exposure, selective prediction, conformal prediction, router risk-coverage, benchmark hygiene, and testing practice. Good starting points are:

- `docs/architecture/ood_recommendation.md`
- `docs/architecture/unknown_disease_rejection.md`
- `docs/architecture/router_performance_literature_review.md`
- `docs/architecture/data_augmentation_leakage_prevention.md`
- `skills/aads-bugfix-debugging/references/bugfix_practices.md`
- [NIST SSDF SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final) for secure/reproducible development process guidance.

## Maintained Entrypoints

- Grouped dataset preparation: `colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb`
- Training: `src/workflows/training.py` via `TrainingWorkflow.run(...)`
- Inference: `src/workflows/inference.py` via `InferenceWorkflow.predict(...)`
- Notebook training surface: `colab_notebooks/2_train_continual_sd_lora_adapter.ipynb`
- Notebook inference surface: `colab_notebooks/1_identify_crop_part_with_router.ipynb`
- Router calibration surface: `colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb`
- Direct adapter validation: `colab_notebooks/3_validate_exported_adapter_directly.ipynb` and `scripts/colab_adapter_smoke_test.py`
- Auxiliary direct adapter UI: `colab_notebooks/4_simple_direct_adapter_test_ui.ipynb` and `scripts/colab_simple_adapter_smoke_ui.py`
- Repo validation and automation: `.github/workflows/ci.yml`, `scripts/validate_notebook_imports.py`, `scripts/validate_config_schema.py`, `scripts/benchmark_surfaces.py`

## Tracked Vs Local-Generated

Tracked source of truth:

- `src/`, `tests/`, `scripts/`, `config/`
- `docs/` and `README.md`
- `colab_notebooks/*.ipynb`
- Root dependency files

Local or generated only:

- `runs/<RUN_ID>/`
- `models/adapters/<crop>/<part>/continual_sd_lora_adapter/`
- `outputs/`
- `.runtime_tmp/`, caches, temp folders, virtualenvs

Do not treat generated outputs as tracked implementation files unless the user explicitly asks about generated artifacts. Do not vendor or edit Codex-home global skills from this repo.

## Repo-Local Skills

Project-local skills live under `skills/` and should be preferred for repo-specific work:

- `aads-training-ood`: `skills/aads-training-ood/SKILL.md`
- `aads-colab-notebooks`: `skills/aads-colab-notebooks/SKILL.md`
- `aads-inference-runtime`: `skills/aads-inference-runtime/SKILL.md`
- `aads-bugfix-debugging`: `skills/aads-bugfix-debugging/SKILL.md`
- `aads-repo-hygiene`: `skills/aads-repo-hygiene/SKILL.md`

Use the smallest set that covers the task.

## Skill Maintenance

- When changing `AGENTS.md` or any repo-local skill, keep the matching `SKILL.md` guidance and `agents/*.yaml` metadata aligned in the same edit.
- Keep shared repo facts in canonical docs when possible. Skills should focus on routing, guardrails, and the narrowest inspect-first set needed for the task.
- After changing repo-local skill routing or metadata, sanity-check at least 2 representative requests so the right skill would trigger and inspect the right canonical files.

## Routing Rules

- Use `aads-training-ood` for `TrainingWorkflow.run(...)`, continual SD-LoRA config, OOD calibration, readiness artifacts, BER comparisons, and training-side adapter export semantics.
- Use `aads-colab-notebooks` for Notebook 0, 1, 2, 3, 4, or 5 changes, grouped dataset preparation, dataset materialization, Hugging Face token handling, Drive telemetry, notebook output mirroring, and notebook-specific troubleshooting.
- Use `aads-inference-runtime` for router inference, adapter lookup and deployment handoff, lazy adapter loading, direct adapter smoke testing, and inference payload behavior.
- Use `aads-bugfix-debugging` for regressions, silent failures, invariant violations, unexpected fallback paths, boundary-validation gaps, and root-cause-driven bug fixes on maintained surfaces.
- Use `aads-repo-hygiene` for CI, tests, benchmark capture, docs consistency, and tracked-vs-generated repo boundaries.

## Overlap Rules

- Use `aads-training-ood` plus `aads-colab-notebooks` for Notebook 0 or Notebook 2 changes that affect dataset materialization, training contracts, or notebook/export mismatches.
- Use `aads-training-ood` plus `aads-bugfix-debugging` for training-side bugs, silent metric or artifact drift, config-normalization regressions, or OOD/readiness contract breaks.
- Use `aads-training-ood` plus `aads-repo-hygiene` for training-side code changes that also affect tests, docs, metrics, or CI coverage.
- Use `aads-inference-runtime` plus `aads-bugfix-debugging` for router/runtime regressions, payload contract bugs, silent misrouting, adapter lookup failures, or degraded fallback behavior.
- Use `aads-inference-runtime` plus `aads-repo-hygiene` for runtime bugfixes, adapter lookup regressions, or inference-facing docs and tests.
- Use `aads-colab-notebooks` plus `aads-bugfix-debugging` for notebook wrapper failures, stale notebook state, dataset-materialization bugs, or notebook-only regressions.
- Use `aads-bugfix-debugging` plus `aads-repo-hygiene` when the fix should add regression tests, validation commands, benchmark checks, or contributor-facing debugging guidance.
- Use `aads-colab-notebooks` plus `aads-inference-runtime` for Notebook 1, Notebook 3, or Notebook 4 tasks that stay on inference and adapter-validation surfaces.
- If a task spans training and inference through the saved adapter contract, anchor on the canonical workflow and runtime entrypoints rather than notebook-only behavior.

## Default Validation Commands

Start with the narrowest relevant subset:

- On Windows PowerShell, prefer `.\scripts\python.cmd ...` so commands resolve the repo `.venv` before any global launcher.
- `.\scripts\python.cmd scripts/validate_notebook_imports.py`
- `.\scripts\python.cmd scripts/validate_config_schema.py`
- `.\scripts\python.cmd scripts/evaluate_dataset_layout.py --root <flat_class_root>`
- `pytest tests/unit tests/colab/test_smoke_training.py -q`
- `pytest tests/integration -q --runintegration`
- `.\scripts\python.cmd scripts/evaluate_router_part_surface.py --root <router_part_eval_root> --config-env colab --output .runtime_tmp/router_part_eval.json`
- `.\scripts\python.cmd scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json`

Run benchmark capture when orchestration interfaces, workflow entrypoints, or router runtime behavior change.

## Non-Goals

- This repo does not define autonomous runtime agents.
- GitHub Actions jobs are automation surfaces, not the "agents" managed by this file.
- Keep project-local skills lean and route to existing docs and scripts instead of copying long guides into skill bodies.
