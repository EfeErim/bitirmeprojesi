# Provenance-Aware Evidence, LogitNorm, and Slice-Aware OOD Selection

## Summary

Implement the three changes as maintained workflow upgrades with minimal contract churn:

- Make stronger evidence and slice-aware reporting part of the default training/readiness flow.
- Keep LogitNorm fully supported but opt-in; default loss remains cross-entropy.
- Keep Notebook 0 focused on supported-class ID prep; OOD remains a separate curated input and existing top-level `ood/` folders remain the canonical OOD slice surface.
- Preserve the existing adapter bundle and inference payload contracts; only training metadata and artifacts expand.

## Implementation Changes

### 1. Provenance-Aware Evaluation Evidence

Add optional provenance metadata to the ID prep flow and use it for report-only shift diagnostics.

- Add an optional Notebook 0 and Notebook 2 `class_root` parameter: `PROVENANCE_MANIFEST_PATH`.
- Auto-discover `<class_root>/provenance_manifest.csv` when the parameter is empty and the file exists.
- Define the initial sidecar contract as CSV-only for v1.
  - Required column: `relative_path`
  - Optional columns: `source_dataset`, `source_subset`, `capture_group_id`, `domain_tag`
  - `relative_path` is matched against the class-root image path using normalized POSIX-style relative paths.
- Extend grouped prep records and manifests to carry the four provenance fields plus the existing `source_hint`.
- Copy provenance fields into the materialized runtime `split_manifest.json` rows so `TrainingWorkflow.run(...)` can consume them without any new workflow arguments.
- If the manifest is missing, partial, or has unmatched rows, do not block prep. Emit warnings in prep artifacts and continue.
- Add authoritative ID-split provenance analysis during training evaluation.
  - Evaluate only on the final authoritative in-distribution split already used for readiness (`test` preferred, `val` fallback only under current policy).
  - Build slice summaries for each populated dimension among: `source_dataset`, `source_subset`, `capture_group_id`, `domain_tag`, and `source_hint`.
  - Only assert or report a slice when it has at least 5 examples in the authoritative split.
- Emit a new report-only artifact on the authoritative split: `provenance_slice_breakdown.json`.
  - Include per-dimension pooled sample count, per-slice sample count, `accuracy`, `balanced_accuracy`, `macro_f1`, and worst-slice values.
  - Include simple deltas from the pooled authoritative split metrics.
- Extend `production_readiness.json` context with a report-only provenance summary and warning list.
  - This must not change `status`, `passed`, or `missing_deployment_requirements` in v1.
  - The warnings are informational only.

### 2. LogitNorm Training Option

Add LogitNorm as a first-class training option without changing the default loss.

- Extend the normalized training config under `training.continual.optimization` with:
  - `loss_name`: `"cross_entropy"` or `"logitnorm"`
  - `logitnorm_tau`: float, default `1.0`
- Keep `loss_name: "cross_entropy"` as the maintained default in shipped config and notebook defaults.
- Validate the new fields in config normalization and trainer config parsing.
- Implement LogitNorm in the trainer classification loss path.
  - Apply it only to the closed-set classification loss.
  - Keep validation and test artifact metrics comparable to the current path.
- For v1, make `ber_enabled=true` incompatible with `loss_name="logitnorm"`.
  - Fail config validation early with a clear error.
  - This keeps the first rollout decision-isolated and benchmarkable.
- Persist `loss_name` and `logitnorm_tau` in adapter metadata and training summary artifacts.
- Surface the option in Notebook 2 with explicit parameters and no hidden overrides.

### 3. Slice-Aware OOD Score Selection

Upgrade method comparison from pooled AUROC/FPR-only to pooled-plus-slice-aware analysis while keeping the current real-OOD guardrail.

- Add a new training artifact on validation/test splits when OOD metrics exist: `ood_method_comparison.json`.
  - Include pooled metrics for `ensemble`, `energy`, and `knn`.
  - Include worst-slice FPR and worst-slice AUROC derived from existing `ood_type_breakdown.json` when real OOD slices exist.
  - Include worst-fold metrics when the evidence source is the held-out benchmark.
- Preserve the current real-OOD guardrail.
  - If `primary_score_method="auto"` and real `ood/` data exists, keep the concrete runtime detector on `ensemble`.
  - Still emit the new comparison artifact so users can inspect slice-aware evidence and rerun with an explicit method if desired.
- Change auto-selection logic only for selection contexts that are already allowed to pick a method from proxy evidence.
  - On held-out benchmark selection, rank methods by:
    1. pooled gate eligibility
    2. best worst-slice or worst-fold FPR
    3. best pooled AUROC
    4. existing deterministic method preference
- Extend the comparison and reporting context in metric artifacts and readiness context with:
  - pooled method metrics
  - worst-slice summary
  - selected method
  - selection source
- Do not change OOD pool curation or folder semantics in this plan. Top-level folders under `ood/` remain the canonical real-OOD slice IDs.

## Public Interfaces and Artifact Additions

Add or change only these public-facing surfaces:

- Notebook params:
  - `PROVENANCE_MANIFEST_PATH` in Notebook 0
  - `PROVENANCE_MANIFEST_PATH`, `LOSS_NAME`, and `LOGITNORM_TAU` in Notebook 2
- Training config:
  - `training.continual.optimization.loss_name`
  - `training.continual.optimization.logitnorm_tau`
- Prep and runtime manifest rows:
  - `source_dataset`, `source_subset`, `capture_group_id`, `domain_tag`
- New artifacts:
  - `provenance_slice_breakdown.json`
  - `ood_method_comparison.json`
- Expanded readiness context:
  - provenance report-only warnings and summary
  - slice-aware OOD method-comparison summary

## Test Plan

Add or update tests for the following scenarios:

- Prep and manifest handling
  - provenance CSV loads and merges by `relative_path`
  - missing or unmatched provenance rows emit warnings but do not block `runtime_ready`
  - provenance fields survive grouped prep and runtime materialization manifest roundtrip
- Evaluation and reporting
  - authoritative split provenance slice artifact is emitted only when metadata exists
  - slices below 5 samples are skipped or non-asserted
  - readiness context includes report-only provenance warnings without changing readiness status
- LogitNorm
  - config normalization accepts new fields and defaults correctly
  - trainer uses LogitNorm when configured
  - `ber_enabled=true` plus `loss_name="logitnorm"` is rejected
  - adapter metadata and training summary persist the new loss settings
- OOD method selection
  - `ood_method_comparison.json` is emitted with pooled and worst-slice metrics
  - held-out benchmark auto-selection uses worst-slice or worst-fold FPR before pooled AUROC
  - real-OOD `auto` still resolves to `ensemble`
- Notebook contract validation
  - Notebook 0 and Notebook 2 parameter cells include the new explicit parameters
  - Notebook 2 keeps `LOSS_NAME` defaulted to cross-entropy and does not hide the new surface

## Assumptions and Defaults

- Notebook 0 remains ID-only; OOD prep remains a separate curated step.
- Real `ood/` folder structure remains the source of truth for OOD slice IDs.
- Provenance sidecar is CSV-only in v1 to avoid dual parser complexity.
- `provenance_manifest.csv` is optional and non-blocking.
- Provenance and domain slice results are report-only in v1 and must not fail readiness by themselves.
- LogitNorm is implemented completely but stays opt-in until benchmark evidence justifies changing the default.
- Existing adapter export layout, inference payloads, and `production_readiness.json` top-level status semantics remain unchanged.
