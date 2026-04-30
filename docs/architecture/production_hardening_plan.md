# Production Hardening Plan For AADS v6

## Summary

Harden the system around deployment claims, not research ablations. The change is intentionally strict-breaking: marginal old runs may fail readiness, `crop_hint` no longer bypasses routing unless explicitly trusted, and the config schema moves to v2.

Defer LoRA target, DINOv3 pooling, and loss/fusion ablations to a later research plan. This plan fixes readiness, OOD selection, router evaluation, crop-hint safety, and imbalance double-correction.

## Implementation Status

Implemented in the maintained code path:

- schema-v2 config defaults for stricter readiness, real-OOD-dev selection, auxiliary gating, and sampler/loss double-correction control
- stricter `production_readiness.json` policy with `readiness_tier`, hard sample floors, per-OOD-type checks, and auxiliary SURE/conformal diagnostics
- real `ood_dev` primary-score and threshold selection before final held-out OOD evaluation
- full router handoff evaluation through `scripts/evaluate_router_surface.py`
- untrusted `crop_hint` behavior plus explicit `trust_crop_hint` bypass support
- weighted-sampler versus effective-number-loss double-correction guardrails

## Key Changes

### 1. Readiness policy

- Bump config schema to `2`.
- Replace the 5/5 OOD evidence floor with deployable defaults: `min_in_distribution_samples=30`, `min_ood_samples=30`, `min_ood_samples_per_type=5`.
- Hard OOD readiness checks become OOD AUROC, OOD FPR, ID sample count, OOD sample count, and per-OOD-type sample count.
- Move SURE and conformal checks into `auxiliary_checks`; report them without blocking `ready` unless `gate_auxiliary_ood_diagnostics=true`.
- Keep top-level readiness statuses `ready`, `provisional`, and `failed`, but add `readiness_tier` and explicit missing requirements for sample-floor failures.

### 2. Real OOD dev selection

- When real splitting provides `ood_dev`, use it by default to select the OOD primary score method and runtime threshold before final evaluation.
- Never use held-out `ood` for selection; it remains final readiness evidence only.
- Persist `selection_source="real_ood_dev"`, selected method, selected threshold, target FPR, and dev metrics in adapter metadata, summary, and readiness context.
- If `ood_dev` is unavailable, keep the current ensemble guardrail and record `selection_source="real_ood_guardrail_no_dev"`.

### 3. Router hardening

- Add `scripts/evaluate_router_surface.py` for a new `data/router_eval` surface:
  - `id/<crop>/<part>/*` expects crop and part.
  - `negatives/off_crop/<label>/*` expects no supported crop handoff.
  - `negatives/non_plant/<label>/*` expects no supported crop handoff.
  - `ambiguous/<label>/*` expects `unknown_crop` or `router_uncertain`.
  - `wrong_part/<crop>/<unsupported_part>/*` expects crop retained but part rejected or marked unknown.
- Report crop accuracy, negative false-accept rate, abstention rate, risk-coverage curve, part precision/recall, unsupported-part emissions, mean latency, p95 latency, and threshold-sweep recommendations.
- Keep the existing router-part eval script for backward compatibility.

### 4. Crop hint safety

- Add `trust_crop_hint: bool = False` through `InferenceWorkflow`, `RouterAdapterRuntime`, CLI, and notebook/script wrappers.
- Treat `crop_hint` alone as an untrusted hint: the router still runs and must agree before adapter loading.
- `trust_crop_hint=True` preserves the old bypass behavior and emits router status `trusted_hint_skipped`.
- If an untrusted hint disagrees with the router or the router is unavailable, return `router_uncertain` or `router_unavailable`; do not load the adapter.

### 5. Class imbalance policy

- Prevent default double compensation: if the train loader resolves to weighted sampling, effective-number class-balanced loss is disabled unless `allow_sampler_and_loss_rebalance=true`.
- Record the disable reason in class-balance runtime artifacts.

## Public Interfaces

- Config schema: `config_schema_version=2`.
- New config defaults:
  - `training.continual.evaluation.min_in_distribution_samples=30`
  - `training.continual.evaluation.min_ood_samples=30`
  - `training.continual.evaluation.min_ood_samples_per_type=5`
  - `training.continual.evaluation.gate_auxiliary_ood_diagnostics=false`
  - `training.continual.ood.real_dev_selection_enabled=true`
  - `training.continual.class_balance.allow_sampler_and_loss_rebalance=false`
- Inference API adds `trust_crop_hint=False`.
- CLI keeps `--crop` as an untrusted hint and adds `--trust-crop-hint` for bypass.
- Readiness artifacts add `readiness_tier`, `auxiliary_checks`, `ood_type_sample_checks`, and real-OOD-dev selection metadata.

## Test Plan

- Readiness fails with fewer than 30 ID/OOD samples or fewer than 5 samples in any OOD type.
- SURE/conformal failures appear under `auxiliary_checks` and do not block `ready` by default.
- `gate_auxiliary_ood_diagnostics=true` restores hard auxiliary gating.
- Real `ood_dev` selection changes method/threshold and final readiness still uses held-out `ood`.
- No `ood_dev` keeps the ensemble guardrail.
- `crop_hint` no longer bypasses by default; trusted hints do bypass.
- Untrusted hint disagreement returns `router_uncertain`; untrusted hint with router failure returns `router_unavailable`.
- Weighted sampling disables class-balanced loss unless explicitly overridden.
- New router eval script parses every `data/router_eval` group and computes false-accept and risk-coverage metrics.
- Update offline workflow golden snapshots for schema v2 and stricter readiness payloads.
- Run router/runtime unit tests, workflow unit tests, `scripts/validate_config_schema.py`, `scripts/validate_notebook_imports.py`, and `scripts/benchmark_surfaces.py`.

## Assumptions

- Scope is production hardening only.
- Strict-breaking behavior is acceptable.
- Moderate deployable evidence floor is fixed at 30 ID, 30 OOD, and 5 per OOD type.
- Real `ood_dev` is allowed for method and threshold selection; held-out `ood` remains untouched final evidence.
- `crop_hint` bypass is allowed only through explicit trusted opt-in.
