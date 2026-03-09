# OOD Readiness Guide

This guide explains how out-of-distribution (OOD) handling works in the current adapter-training workflow, what data you need, and how to read the final readiness result.

## What OOD Means Here

For one crop adapter, the known classes are the diseases you trained on.

Example for tomato:

- known classes: `healthy`, `early_blight`, `late_blight`
- OOD: anything outside that label set

That can include:

- tomato diseases not in the adapter labels
- other plant leaves
- damaged or badly cropped images
- non-plant images

The goal is simple:

- if the image matches a known class, predict that class
- if the image does not look like any known class strongly enough, return an unknown / OOD decision

## Current Workflow

The supported training path does this automatically:

1. Train the adapter on all known classes.
2. Calibrate OOD statistics on known-class data.
3. Save the adapter with its OOD state.
4. Look for real OOD evidence under `data/<crop>/ood/`.
5. If real OOD data exists, evaluate against it.
6. If real OOD data does not exist, run the held-out-class fallback benchmark.
7. Write the final deployment verdict to `production_readiness.json`.

The final readiness decision is not taken from `validation/metric_gate.json` or `test/metric_gate.json` alone. The authoritative artifact is `production_readiness.json`.

## Dataset Layout

Known classes still use the normal supervised splits:

```text
data/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
```

Real OOD data is optional and uses one shared folder:

```text
data/<crop>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

The `ood/` folder can also contain subfolders:

```text
data/tomato/ood/
  other_plants/*
  background/*
  random_objects/*
```

Important details:

- `ood/` is a shared pool of unknown examples, not one folder per known disease class.
- Images under `ood/` are loaded recursively.
- Subfolder names inside `ood/` are not treated as class labels.
- The same `ood/` folder can be reused across multiple runs for the same crop.

## What To Put In `ood/`

Best choices are realistic hard negatives:

- unseen diseases for the same crop
- related plant leaves not in the adapter label set
- non-disease leaf damage
- bad crops, blur, partial leaves, background clutter

Allowed but weaker choices:

- random cars, houses, animals, indoor objects

Those are valid OOD samples, but they are often too easy. A detector can look strong on them while still failing on plant-like unknowns.

## If `ood/` Does Not Exist

The workflow falls back to a held-out-class benchmark.

Simple example with 3 classes `a`, `b`, `c`:

1. Train on `a` and `b`, then test whether `c` is rejected as unknown.
2. Train on `a` and `c`, then test whether `b` is rejected as unknown.
3. Train on `b` and `c`, then test whether `a` is rejected as unknown.

Those temporary fold models are only for benchmarking. The deployed adapter is still the normal final model trained on all available classes.

This fallback is useful when you have no real unknown-image set, but it is still proxy evidence rather than direct real-world OOD proof.

## How Readiness Is Decided

The repo combines two kinds of evidence:

- classification evidence from the normal validation/test path
- OOD evidence from either a real `ood/` split or the held-out benchmark

The final artifact is:

```text
<artifact_root>/production_readiness.json
```

Main fields:

- `status`: `ready` or `failed`
- `passed`: boolean mirror of the status
- `ood_evidence_source`: `real_ood_split`, `held_out_benchmark`, or `unavailable`
- `classification_evidence`: accuracy evidence from the authoritative split
- `ood_evidence`: OOD metrics and their checks
- `missing_requirements`: metrics that blocked readiness
- `targets`: threshold values used for gating
- `context`: run metadata

Default targets are currently:

- `accuracy >= 0.93`
- `ood_auroc >= 0.92`
- `ood_false_positive_rate <= 0.05`
- `sure_ds_f1 >= 0.90`
- `conformal_empirical_coverage >= 0.95`

## Important Artifacts

Workflow / CLI training writes these under:

```text
<output_dir>/training_metrics/
```

Most useful files:

- `validation/metric_gate.json`
- `test/metric_gate.json`
- `ood_benchmark/summary.json`
- `ood_benchmark/per_fold.csv`
- `production_readiness.json`
- `training/summary.json`

Notebook 2 writes the same artifact names under:

```text
outputs/colab_notebook_training/artifacts/
```

and mirrors them into the matching `runs/<RUN_ID>/...` export tree.

How to read them:

- `validation/metric_gate.json` and `test/metric_gate.json`: split-local diagnostics
- `ood_benchmark/summary.json`: aggregate fallback results when no real `ood/` split exists
- `production_readiness.json`: final deployment decision

## Configuration

The OOD readiness policy lives under `training.continual.evaluation`.

Current defaults:

- `require_ood_for_gate: true`
- `ood_fallback_strategy: "held_out_benchmark"`
- `ood_benchmark_auto_run: true`
- `ood_benchmark_min_classes: 3`

Practical meaning:

- OOD evidence is required for readiness by default.
- If no real `ood/` split exists, the workflow auto-runs the fallback benchmark.
- If the crop has fewer than 3 classes, fallback evidence is considered too weak and readiness fails.

## What This Does And Does Not Guarantee

What the repo does automatically:

- OOD calibration
- OOD metric computation when evidence exists
- fallback benchmarking when real OOD data is missing
- final readiness reporting

What still requires judgment:

- whether your `ood/` set is realistic enough
- whether fallback evidence is strong enough for your deployment risk

Important limitation:

- no benchmark can prove rejection for every unseen disease in the world

The practical goal is not perfect certainty. It is to reduce confident wrong predictions on things the adapter was never trained to recognize.

## Recommended Practice For Plant Adapters

Use this order of preference:

1. Train normally on the disease classes you support.
2. Add a small shared `ood/` pool with realistic plant-like unknowns if you can.
3. Let the workflow use the fallback benchmark when you do not have that pool yet.
4. Inspect `production_readiness.json` instead of relying on one split-local metric file.
5. Expand the shared `ood/` pool over time using real failure cases and hard negatives.

## Related Files

- [README.md](../../README.md)
- [Colab Training Manual](colab_training_manual.md)
- [Architecture Overview](../architecture/overview.md)
- [OOD Recommendation](../architecture/ood_recommendation.md)
- [Training Workflow](../../src/workflows/training.py)
- [OOD Benchmark Service](../../src/training/services/ood_benchmark.py)
