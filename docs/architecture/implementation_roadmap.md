# LogitNorm and Slice-Aware OOD Selection

## Summary

Planned workflow upgrades should stay focused on two maintained directions:

- keep LogitNorm supported as an opt-in training loss
- keep OOD method comparison slice-aware without changing the current real-OOD guardrail

The training workflow, adapter bundle, and inference payload contracts should remain stable.

## Implementation Changes

### 1. LogitNorm Training Option

- Keep `training.continual.optimization.loss_name` limited to `"cross_entropy"` or `"logitnorm"`.
- Keep `training.continual.optimization.logitnorm_tau` as the opt-in LogitNorm parameter.
- Keep cross-entropy as the shipped default.
- Keep `ber_enabled=true` incompatible with `loss_name="logitnorm"` so the first comparison stays decision-isolated.
- Persist `loss_name` and `logitnorm_tau` in training summary and adapter export metadata.
- Surface the option directly in Notebook 2 with no hidden notebook-only override layer.

### 2. Slice-Aware OOD Score Selection

- Keep `ensemble`, `energy`, and `knn` as the maintained OOD score methods.
- Emit `ood_method_comparison.json` when OOD metrics exist.
- Include pooled metrics plus worst-slice or worst-fold summaries when available.
- Preserve the current real-OOD guardrail:
  - if `primary_score_method="auto"` and real `ood/` data exists, keep the runtime detector on `ensemble`
  - treat method comparison as analysis unless the user reruns with an explicit method
- Allow held-out benchmark auto-selection to rank methods using pooled gate eligibility, worst-fold or worst-slice FPR, then pooled AUROC.

## Public Interfaces

The maintained public surfaces are:

- `training.continual.optimization.loss_name`
- `training.continual.optimization.logitnorm_tau`
- `ood_method_comparison.json`

No additional dataset metadata sidecars or report-only split-diagnostic artifacts are part of the maintained plan.

## Test Plan

- LogitNorm config normalization accepts defaults and explicit values.
- Trainer uses LogitNorm when configured.
- `ber_enabled=true` plus `loss_name="logitnorm"` fails fast.
- Training summary and adapter metadata persist the selected loss settings.
- `ood_method_comparison.json` is emitted with pooled and worst-slice or worst-fold summaries.
- Real-OOD `auto` still resolves to `ensemble`.

## Defaults

- Notebook 0 remains ID-only prep.
- Real `ood/` folder structure remains the maintained OOD slice surface.
- LogitNorm stays opt-in until benchmark evidence justifies changing the default.
- Existing adapter export layout, inference payloads, and `production_readiness.json` status semantics remain unchanged.
