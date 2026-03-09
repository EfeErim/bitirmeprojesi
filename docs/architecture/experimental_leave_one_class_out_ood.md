# Experimental Prototype: Leave-One-Class-Out OOD Evaluation

Status: historical design note. The supported workflow now includes an automatic held-out benchmark fallback when no real `data/<crop>/ood/` split exists. For the current user-facing behavior, see [OOD Readiness Guide](../user_guide/ood_readiness_guide.md).

## Goal

Capture a benchmarking protocol for out-of-distribution (OOD) behavior when the dataset has only known classes and no separate unknown set.

## Idea

Given a dataset with `N` classes:

1. Select one class as the temporary OOD class.
2. Train the adapter on the remaining `N-1` classes only.
3. Calibrate OOD using only the seen classes.
4. Evaluate on a mixed set:
   - samples from the `N-1` seen classes have `ood_label = 0`
   - samples from the held-out class have `ood_label = 1`
   - model `ensemble_score` is used as `ood_score`
5. Record classification metrics on seen classes and OOD metrics on the mixed set.
6. Repeat for each class.
7. After benchmarking, train the production adapter on all `N` classes.

## Why This Exists

The current v6 training path calibrates OOD after normal training on all detected classes. That is appropriate for production adapter creation, but it does not by itself measure how well the adapter rejects unseen classes.

This prototype provides a controlled evaluation protocol for that question.

## Intended Outputs

Per held-out class:

- seen-class classification metrics
- OOD AUROC
- OOD false positive rate
- calibration metadata
- summary of the held-out class and seen class set

Aggregate:

- mean and variance across all held-out-class runs
- per-class failure cases
- final comparison against the production adapter trained on all classes

## Non-Goals

- Do not replace `TrainingWorkflow.run(...)`.
- Do not change the current notebook dataset contract.
- Do not make OOD benchmarking mandatory for normal adapter training.

## Suggested Implementation Shape

Possible future entrypoints:

- a standalone script under `scripts/`
- an opt-in workflow helper under `src/workflows/`
- a notebook-side experimental cell path, clearly separated from the supported flow

Suggested high-level loop:

1. Enumerate dataset classes.
2. For each held-out class:
   - materialize a train/val split containing only seen classes
   - materialize an OOD eval split mixing seen validation samples with held-out samples
   - train adapter on seen classes
   - calibrate OOD on seen train/val data
   - collect `ood_labels` and `ood_scores` from the mixed OOD eval split
   - persist metrics and artifacts
3. Train a final production adapter on all classes.

## Current Repo Mapping

- Train/evaluate canonical adapter: `src/workflows/training.py`
- OOD detector and `ensemble_score`: `src/ood/continual_ood.py`
- OOD calibration: `src/training/services/ood_calibration.py`
- Metric computation for `ood_labels` / `ood_scores`: `src/training/services/metrics.py`
- Validation artifact persistence: `src/training/services/reporting.py`

## Important Constraint

This protocol is an evaluation strategy, not a production training requirement.

The supported default remains:

1. train on all known classes
2. calibrate OOD on known-class data
3. save adapter and metadata for inference
