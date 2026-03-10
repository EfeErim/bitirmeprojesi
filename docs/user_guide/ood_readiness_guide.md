# OOD Readiness Guide

This guide describes the current OOD and readiness flow implemented by the training workflow and used by Notebook 2.

## Current Workflow

The supported path is:

1. train the adapter on the known classes
2. calibrate OOD on known-class data
3. save the adapter with its OOD state
4. look for real OOD evidence under `data/<crop>/ood/` or the runtime equivalent
5. if real OOD data exists, score against it
6. if real OOD data does not exist, run the held-out fallback benchmark
7. write the final verdict to `production_readiness.json`

The final deployment decision does not come from `validation/metric_gate.json` or `test/metric_gate.json` alone. The authoritative artifact is `production_readiness.json`.

## What OOD Means Here

For one crop adapter, known classes are the diseases that adapter was trained to predict.

Example for tomato:

- known: `healthy`, `early_blight`, `late_blight`
- OOD: anything outside that label set

That includes:

- unseen diseases for the same crop
- other plant leaves
- bad crops, blur, damage, or background clutter
- non-plant images

The practical goal is to reduce confident wrong predictions on unsupported inputs.

## Dataset Layout

The runtime training layout is:

```text
data/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
```

Real OOD evidence is optional and uses one shared pool:

```text
data/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

`ood/` may also contain nested folders:

```text
data/tomato/ood/
  other_plants/*
  background/*
  random_objects/*
```

Current behavior:

- images under `ood/` are loaded recursively
- subfolder names inside `ood/` are not treated as class labels
- one shared `ood/` pool can be reused across runs for the same crop

## When Real OOD Data Is Missing

The current fallback is a leave-one-class-out benchmark.

Example with classes `a`, `b`, `c`:

1. train on `a` and `b`, treat `c` as temporary OOD
2. train on `a` and `c`, treat `b` as temporary OOD
3. train on `b` and `c`, treat `a` as temporary OOD

These fold models are benchmarking artifacts only. The deployable adapter is still the normal final model trained on all classes.

If fewer than 3 classes are available, the fallback is considered too weak and readiness fails.

## How The Final Verdict Is Chosen

The readiness artifact combines:

- classification evidence from the authoritative in-distribution split
- OOD evidence from either a real `ood/` split or the held-out benchmark

Current authoritative split selection:

- prefer `test` when a `test/metric_gate.json` artifact exists
- otherwise fall back to `val` only when `val` was not also used for OOD calibration

Important isolation rule:

- if `val` was used for OOD calibration and no isolated `test/` split exists, production readiness fails instead of reusing calibration data as evaluation evidence

That chosen split appears in:

```text
production_readiness.json -> classification_evidence.split_name
```

## Artifact Layout

Workflow and CLI training write under:

```text
<output_dir>/training_metrics/
```

Key files:

- `training/results.png`
- `training/results.csv`
- `training/history.json`
- `training/history.csv`
- `training/batch_metrics.csv`
- `training/summary.json`
- `validation/metric_gate.json`
- `test/metric_gate.json`
- `ood_benchmark/summary.json`
- `ood_benchmark/per_fold.csv`
- `production_readiness.json`

Notebook 2 writes the same artifact families locally under:

```text
outputs/colab_notebook_training/artifacts/
```

and mirrors them into:

```text
runs/<RUN_ID>/outputs/colab_notebook_training/artifacts/
runs/<RUN_ID>/telemetry/artifacts/
```

## How To Read The Artifacts

- `validation/metric_gate.json` and `test/metric_gate.json`
  Split-local classification plus OOD metric checks for that split.

- `ood_benchmark/summary.json`
  Aggregate fallback evidence when no real `ood/` split exists.

- `production_readiness.json`
  Final deployment verdict. This is the artifact to use for go/no-go decisions.

## Readiness Targets

The current default targets come from `DEFAULT_PLAN_TARGETS` in `src/training/services/metrics.py`:

- `accuracy >= 0.93`
- `ood_auroc >= 0.92`
- `ood_false_positive_rate <= 0.05`
- `sure_ds_f1 >= 0.90`
- `conformal_empirical_coverage >= 0.95`

Current implementation detail:

- production readiness follows `require_ood_for_gate`
- split-local metric gates are still computed for internal readiness logic
- `emit_ood_gate` controls whether `validation/metric_gate.json` and `test/metric_gate.json` are written to disk
- the final readiness artifact fails if required OOD evidence is unavailable or below target

## Configuration

The current readiness policy is controlled by `training.continual.evaluation`:

- `best_metric`
- `emit_ood_gate`
- `require_ood_for_gate`
- `ood_fallback_strategy`
- `ood_benchmark_auto_run`
- `ood_benchmark_min_classes`

Current defaults in shipped config:

- `require_ood_for_gate: true`
- `ood_fallback_strategy: "held_out_benchmark"`
- `ood_benchmark_auto_run: true`
- `ood_benchmark_min_classes: 3`

Current benchmark planning detail:

- when fallback benchmarking runs, the workflow reports the estimated fold count before the benchmark starts
- for `N` known classes, the held-out fallback retrains `N` fold models, so exploratory runs can become expensive quickly

## BER Note

`training.continual.ood.ber_enabled` is currently a training-only regularizer.

It does not change:

- the readiness artifact schema
- the inference payload shape
- the readiness threshold definitions

Evaluate BER by comparing the same artifact set on the same crop, split layout, seed, and OOD evidence source.

## Recommended Practice

1. Train on the disease classes you actually support.
2. Add a realistic shared `ood/` pool when possible.
3. Keep an isolated `test/` split if `val` is used for calibration and you want a final readiness verdict.
4. Let the fallback benchmark run only when that pool is missing.
5. Read `production_readiness.json` before deployment.
6. Treat split-local metric gates as diagnostics.
7. Compare BER or other experiments on identical evidence sources.

## Related Files

- [../../README.md](../../README.md)
- [colab_training_manual.md](colab_training_manual.md)
- [../architecture/overview.md](../architecture/overview.md)
- [../architecture/ood_recommendation.md](../architecture/ood_recommendation.md)
- [../../src/workflows/training.py](../../src/workflows/training.py)
- [../../src/training/services/metrics.py](../../src/training/services/metrics.py)
- [../../src/training/services/ood_benchmark.py](../../src/training/services/ood_benchmark.py)
