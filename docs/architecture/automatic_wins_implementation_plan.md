# Single-Batch Plan For Automatic Wins Across Training, OOD, Data, and Reporting

## Summary

Implement all seven items in one branch, but separate them into two delivery classes inside that branch:

- Default-on low-risk improvements: switch energy temperature calibration to auto and extend experiment/reporting surfaces for label smoothing.
- Default-off research features with full plumbing: ReAct-style activation rectification, classifier-only rebalance, AugMix, OE with a separate auxiliary unknown pool, and Notebook 0 label-quality surfacing.

This keeps the public `TrainingWorkflow.run(...)` contract intact, avoids contaminating final readiness evidence, and makes every change measurable through existing artifact and benchmark surfaces before any later default flips.

## Key Changes

### 1. OOD calibration and scoring

- Change the shipped default `training.continual.ood.energy_temperature_mode` from `fixed` to `auto`.
- Keep the existing real-OOD guardrail: method or detector selection must not tune on final real `ood/` evidence.
- Add a new optional rectification block under `training.continual.ood`:
  - `react_enabled: false`
  - `react_percentile: 0.99`
  - `react_apply_during_calibration: true`
  - `react_apply_during_inference: true`
- Implement ReAct as post-encoding feature clipping before classifier/OOD scoring so it reuses the current materialized feature path and affects `ensemble`, `energy`, and `knn` consistently.
- Persist rectification metadata into adapter OOD calibration state and readiness/reporting artifacts so exported adapters remain self-describing.
- Extend OOD comparison artifacts so runs report base-vs-ReAct method metrics separately; auto-selection continues to use only allowed proxy evidence.

### 2. Training objective and rebalance work

- Keep `optimization.loss_name="logitnorm"` as the default.
- Do not change the shipped `label_smoothing` default; keep `0.0` and implement it as a first-class experiment dimension.
- Extend the optimization/search/reporting surfaces so `training.optimization.label_smoothing` is sweepable and appears in run registry, Pareto/frontier summaries, and Colab recommendations.
- Add a new optional `training.continual.classifier_rebalance` block:
  - `enabled: false`
  - `epochs: 3`
  - `learning_rate: 5e-5`
  - `weight_decay: 0.0`
  - `sampler: weighted`
  - `objective: logit_adjusted_cross_entropy`
  - `logit_adjustment_tau: 1.0`
- Implement rebalance as a second training stage after the main session:
  - freeze backbone, LoRA adapter weights, and fusion
  - train only the classifier head
  - use the same continual split with the configured rebalanced sampler/objective
  - recalibrate OOD only after the rebalance stage finishes
- Record both pre-rebalance and post-rebalance validation/test metrics in artifacts, with the final export using post-rebalance weights when the stage is enabled.

### 3. Data augmentation and OE data contract

- Extend `training.continual.data.augmentation_policy` to allow `augmix` in addition to `none`, `basic`, and `randaugment`.
- Keep `randaugment` as the default shipped policy.
- Add AugMix-specific config only as needed for a minimal stable implementation:
  - `augmix_severity`
  - `augmix_width`
  - `augmix_depth`
  - `augmix_alpha`
- Add a separate OE contract under `training.continual.ood`:
  - `oe_enabled: false`
  - `oe_loss_weight: 0.5`
  - `oe_target: uniform`
  - `oe_root: ""`
- Define auxiliary OE data as a separate unknown pool, not the final readiness `ood/` pool:
  - runtime contract: optional `ood_aux/` tree under the crop runtime root, or explicit `oe_root`
  - Notebook/data-prep support: optional reusable repo-local auxiliary pool mirroring the existing reusable OOD staging pattern
- Use OE only during training loss computation; never use `ood_aux/` as final readiness evidence.
- When OE is enabled, report auxiliary-OE sample counts and source path in traceability and readiness context.

### 4. Notebook 0 label-quality surfacing

- Add a Notebook 0 audit-time review artifact family that builds on the existing human-in-the-loop prep gate and does not auto-relabel:
  - `label_review_candidates.csv`
  - `label_review_summary.json`
  - guided artifact entry for notebook/user navigation
- Keep the signal source audit-time and metadata-free, prioritizing the existing grouped-prep evidence:
  - cross-class duplicate/conflict findings
  - borderline same-class review clusters
  - representation-neighbor disagreement against local class neighborhoods
  - source-style and train-only routing evidence already used by Notebook 0
- Keep this as a one-way review surface in v1:
  - no automatic relabeling
  - no Notebook 2 ownership of label-quality review
  - use Notebook 0's existing human review packet as the confirmation point before materialization

### 5. Docs, notebook wrappers, and reporting

- Update canonical docs and notebook guidance together:
  - training manual
  - OOD readiness guide
  - architecture overview / unknown-disease rejection note
- Document the new separation between:
  - final real `ood/` evidence
  - auxiliary `ood_aux/` OE training data
  - Notebook 0 audit-time label-review artifacts
- Expose new knobs in Notebook 2 only where they are part of the maintained workflow surface; keep defaults conservative and notebook wrappers thin.
- Extend benchmark and traceability outputs so the new dimensions are visible in run comparison:
  - energy temperature mode
  - ReAct enabled/percentile
  - label smoothing
  - augmentation policy
  - classifier rebalance enabled/objective
  - OE enabled and auxiliary sample counts

## Public Interfaces And Artifact Additions

- Add config keys under `training.continual.ood` for ReAct and OE.
- Add `augmix` to the allowed `training.continual.data.augmentation_policy` enum.
- Add `training.continual.classifier_rebalance` as a new optional block.
- Add Notebook 0 label-review artifacts and guided-artifact references as maintained outputs.
- Preserve backward compatibility:
  - existing configs remain valid
  - all new feature blocks default to off except `energy_temperature_mode: "auto"`

## Test Plan

- Config normalization and schema tests for all new keys and enum values.
- Unit tests for ReAct feature clipping, persistence, and score/report propagation.
- Unit tests for AugMix transform construction and validation.
- Loader/workflow tests proving `ood_aux/` is isolated from final readiness `ood/`.
- Training tests for classifier-rebalance stage:
  - only classifier params are trainable in stage 2
  - OOD recalibration happens after stage 2
  - pre/post rebalance metrics are both emitted
- Notebook 0 artifact/reporting tests for label-review outputs and guided-artifact visibility.
- Colab/notebook import validation and narrow smoke training coverage for the new config surface.
- Benchmark-surface regression run to confirm the workflow entrypoint still emits comparable artifacts.

## Assumptions And Defaults

- Single implementation batch means one integrated branch, not one simultaneous default flip for every research feature.
- Only `energy_temperature_mode` becomes default-on in this batch.
- Label smoothing is implemented as a measured experiment surface, not a shipped default change.
- ReAct, classifier rebalance, AugMix, and OE ship fully wired but default-off until repo-local evidence supports promotion.
- OE uses a separate auxiliary unknown pool and must never be counted as final readiness evidence.
- Notebook 0 label-quality work is surfacing-only in v1; it does not create an automated audit feedback loop.
