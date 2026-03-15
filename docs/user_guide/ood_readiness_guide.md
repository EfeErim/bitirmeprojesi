# OOD Readiness Guide

This guide explains how AADS v6 decides whether a trained adapter is ready for deployment.

It is written for readers who may not already know what OOD means.

If you are new to the repo, read [../../README.md](../../README.md) first.

## The Short Version

The workflow does not ask only:

- "Is the model accurate on known classes?"

It also asks:

- "Can the model recognize inputs that do not belong to the classes it was trained to support?"

The final answer is written to:

```text
production_readiness.json
```

That file is the authoritative deployment verdict.

## What OOD Means In This Repo

`OOD` stands for `out of distribution`.

For one crop adapter, the known classes are the disease labels that adapter was trained on. OOD means the input does not belong to that supported label set.

Example for a tomato adapter trained on:

- `healthy`
- `early_blight`
- `late_blight`

Possible OOD inputs include:

- a tomato disease not in that label set
- a potato or wheat image
- a non-leaf plant image that does not match training coverage
- a blurred or heavily damaged image
- a background or random object image
- a non-plant image

The practical goal is to reduce confident wrong predictions on unsupported inputs.

Deployment assumption note:

- if an upstream router or trusted `crop_hint` guarantees that the tomato adapter only receives tomato images, then off-crop inputs like potato become lower-priority adapter negatives rather than the main deployment risk
- in that crop-gated setup, the highest-value OOD evidence is usually unsupported tomato diseases plus realistic tomato-image failure cases such as blur, occlusion, clutter, or unusual viewpoints
- this does not remove the OOD requirement entirely because the adapter can still be overconfident on unsupported tomato inputs

## The Current Readiness Workflow

The supported path is:

1. train the adapter on known classes
2. calibrate OOD on known-class data
3. save the adapter together with its OOD state
4. look for real OOD evidence under the runtime dataset
5. if real OOD data exists, evaluate against it
6. if real OOD data does not exist, run the held-out fallback benchmark automatically
7. combine classification evidence and OOD evidence into a final verdict
8. write `production_readiness.json`

## Known Classes Vs Unknown Inputs

Think of readiness as a two-part check.

### Part 1: classification quality

Can the model correctly separate the classes it is supposed to support?

### Part 2: unknown-input handling

Can the model avoid acting overconfident on images outside that supported set?

Both matter for deployment.

## Dataset Layout For OOD Evidence

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

The `ood/` folder may also contain nested folders:

```text
data/tomato/ood/
  other_plants/*
  background/*
  random_objects/*
```

Current behavior:

- images under `ood/` are loaded recursively
- nested folder names inside `ood/` are not treated as class labels
- one shared `ood/` pool can be reused across runs for the same crop
- when real `ood/` data is evaluated, the top-level folder under `ood/` is emitted as `ood_type` in validation and test artifacts so you can inspect near/far/non-plant/blur-style slices without turning them into supported labels

## How To Build A Real `ood/` Pool

Treat `data/<crop>/ood/` as one shared pool of unsupported inputs for that crop adapter.

These images are used as unknown-input evidence during readiness evaluation. They are not another supported disease class and they are not part of the known-class training split.

### What belongs in `ood/`

Good candidates include:

- diseases of the same crop that are not in the adapter's supported label set
- realistic same-crop failure cases such as unusual viewpoints, heavy blur, occlusion, clutter, or abiotic damage
- images from other crops or other plant species
- plant parts that fall outside the supported training coverage
- a minority slice of clearly non-plant or random-object images

### What does not belong in `ood/`

Do not put these into `ood/`:

- images from the exact supported classes already used under `continual/`, `val/`, or `test/`
- another copy of the normal validation or test split
- images you intend the adapter to support for this run
- an `ood` class label inside the known-class train or notebook input layout

If a category should become a supported class in the next experiment, move it into the normal known-class split layout for that run instead of keeping it in `ood/`.

### Practical curation rules

Use these rules when building the pool:

1. Prefer realistic hard negatives over easy random negatives.
2. Keep the pool clean and reusable even if it starts small.
3. Reuse the same `ood/` pool across comparable runs so readiness comparisons stay meaningful.
4. Use nested folders only for organization, not for labels.
5. Keep random non-plant images as a minority slice rather than the whole pool.
6. Match the pool to the deployment contract. If the adapter is crop-gated upstream, make most of the pool same-crop unknowns and same-crop failure cases first, then add off-crop images as secondary evidence.

Important current repo behavior:

- the repo does not require a fixed class-balance ratio, but the readiness gate now requires at least 5 in-distribution and 5 OOD evaluation examples before OOD metrics can satisfy the final target
- a small clean `ood/` pool is better than a large noisy one
- plant-like unknowns are usually more valuable than only easy random objects

### Example organization

One practical pattern is:

```text
data/tomato/ood/
  unsupported_tomato_diseases/*
  off_coverage_views/*
  abiotic_damage/*
  blur_occlusion_clutter/*
  other_crops_optional/*
  non_plant_misc/*
```

The folder names above are only for human organization. The workflow evaluates everything under `ood/` as one shared unknown pool.

Reporting detail:

- the workflow still computes the main OOD gate on the pooled `ood/` split
- additional `ood_type_breakdown.json` artifacts summarize AUROC and FPR by top-level `ood/` folder when that structure exists

## What Happens When Real OOD Data Exists

This is the strongest current evidence path.

The workflow:

1. trains the final model on the known classes
2. calibrates OOD
3. evaluates the model on a known-class split plus the real `ood/` pool
4. uses those results in the final readiness decision

When this path is available, the readiness artifact records the OOD evidence source as:

- `real_ood_split`

## What Happens When Real OOD Data Is Missing

The fallback is a leave-one-class-out benchmark.

Example with known classes `a`, `b`, and `c`:

1. train on `a` and `b`, then treat `c` as temporary unknown data
2. train on `a` and `c`, then treat `b` as temporary unknown data
3. train on `b` and `c`, then treat `a` as temporary unknown data

Why this exists:

- it gives the workflow some evidence about how the system behaves on unseen classes
- it is better than having no OOD evidence at all

What it does not mean:

- these fold models are not the deployment model
- the final deployable adapter is still the main model trained on all classes

If fewer than 3 classes are available, the fallback is considered too weak and readiness fails.

When this path is used, the readiness artifact records the OOD evidence source as:

- `held_out_benchmark`

Current pass rule for the fallback benchmark:

- the selected aggregate OOD metrics must meet their targets
- every completed held-out fold must also meet those same OOD targets

This second rule is a repo-local guardrail motivated by OOD-evaluation literature that warns against trusting only pooled averages when failure behavior is heterogeneous across slices. See [In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation](https://proceedings.mlr.press/v202/bitterwolf23a.html).

Diagnostic note:

- while the fallback benchmark is running, the workflow persists `ood_benchmark/progress.json`
- if a fold raises a Python exception, the workflow also writes `ood_benchmark/folds/<held_out_class>/failure.json` and `failure_traceback.txt`
- these files are for troubleshooting interrupted or failed benchmark runs; the final deployment verdict still comes from `production_readiness.json`

## How The Final Verdict Is Chosen

The final readiness artifact combines:

- classification evidence from the authoritative in-distribution split
- OOD evidence from either the real `ood/` split or the fallback benchmark

### Which classification split is authoritative

Current selection order:

1. prefer `test` when `test/metric_gate.json` exists
2. otherwise fall back to `val` only when `val` was not also used for OOD calibration

Important isolation rule:

- if `val` was used for OOD calibration and there is no isolated `test/` split, production readiness fails instead of reusing calibration data as final evidence

That chosen split appears in:

```text
production_readiness.json -> classification_evidence.split_name
```

## Why `production_readiness.json` Matters More Than `metric_gate.json`

The repo writes multiple metric files, but they answer different questions.

### `validation/metric_gate.json` and `test/metric_gate.json`

These are split-local diagnostics.

They tell you whether one specific split met the configured thresholds.

### `production_readiness.json`

This is the final deployment artifact.

It combines:

- the selected classification evidence
- the available OOD evidence
- the configured targets
- the final missing requirements list

Use this file for go/no-go deployment decisions.

## What `production_readiness.json` Contains

The current payload shape is conceptually:

```json
{
  "status": "ready or failed",
  "passed": true,
  "ood_evidence_source": "real_ood_split | held_out_benchmark | unavailable",
  "classification_evidence": {
    "split_name": "test or val",
    "metrics": {},
    "evaluation": {
      "checks": {
        "accuracy": {}
      }
    }
  },
  "ood_evidence": {
    "source": "real_ood_split | held_out_benchmark | unavailable",
    "metrics": {},
    "evaluation": {}
  },
  "missing_requirements": [],
  "targets": {},
  "context": {}
}
```

### Fields to focus on first

- `status`: `ready` or `failed`
- `passed`: boolean version of the same outcome
- `missing_requirements`: which required targets were not satisfied
- `classification_evidence.split_name`: which in-distribution split was used
- `ood_evidence.source`: where OOD evidence came from
- `targets`: the thresholds the run was judged against

## How To Read `metric_gate.json`

The split-local metric gate stores threshold checks in a machine-readable form.

Important ideas:

- `value`: the measured metric
- `target`: the required threshold
- `operator`: whether the metric must be above or below the threshold
- `asserted`: whether the metric was available at all
- `passed`: whether that metric met its threshold

The gate also contains a `gating` section that explains whether missing metrics were treated as a hard failure or a softer status.

## Current Default Targets

The current default targets come from `DEFAULT_PLAN_TARGETS` in `src/training/services/metrics.py`:

- `accuracy >= 0.93`
- `ood_auroc >= 0.92`
- `ood_false_positive_rate <= 0.05`
- `ood_samples >= 5`
- `in_distribution_samples >= 5`
- `sure_ds_f1 >= 0.90`
- `conformal_empirical_coverage >= 0.95`
- `conformal_avg_set_size <= 2.0`

Important interpretation note:

- `ood_samples` and `in_distribution_samples` are repo-local evidence-sufficiency floors, not universal thresholds copied from one paper. They are conservative engineering safeguards motivated by small-sample ROC/AUC instability studies such as [Confidence intervals for the receiver operating characteristic area in studies with small samples](https://pubmed.ncbi.nlm.nih.gov/9702267/), [Confidence bounds when the estimated ROC area is 1.0](https://pubmed.ncbi.nlm.nih.gov/12458878/), and [A comparison of confidence/credible interval methods for the area under the ROC curve for continuous diagnostic tests with small sample size](https://pubmed.ncbi.nlm.nih.gov/26323286/).
- `conformal_avg_set_size` is also a repo-local utility bar. The exact threshold is an engineering choice for this project, but the reason it exists is literature-backed: conformal prediction quality depends on efficiency as well as coverage. See [Entropy Reweighted Conformal Classification](https://proceedings.mlr.press/v230/luo24a.html).

## Conformal Modes

The repo now separates two different conformal use cases.

### Mode 1: `threshold`

This conformalizes the detector's OOD residual score. It helps stabilize the rejection threshold, but it is not the same thing as standard set-valued classification.

Guarantee conditions to keep in mind:

- calibration and evaluation samples still need to be exchangeable for the split-conformal quantile argument to make sense
- the guarantee is about the calibrated score residual, not about APS/RAPS-style label-set coverage
- if you calibrate on `val`, do not treat the same `val` split as final deployment evidence

### Mode 2: `aps`

This is standard Adaptive Prediction Sets over class probabilities.

### Mode 3: `raps`

This is Regularized Adaptive Prediction Sets with an explicit set-size penalty.

APS/RAPS are the standard choice when your real goal is set-valued classification rather than OOD-threshold conformalization.

Current repo configuration keys:

- `training.continual.ood.conformal_method`
- `training.continual.ood.conformal_alpha`
- `training.continual.ood.conformal_raps_lambda`
- `training.continual.ood.conformal_raps_k_reg`

These targets are the default bar used unless a different target spec is loaded.

## Configuration That Controls Readiness

The current readiness policy lives under:

```text
training.continual.evaluation
```

Important keys:

- `best_metric`
- `emit_ood_gate`
- `require_ood_for_gate`
- `ood_benchmark_min_classes`

Current shipped defaults include:

- `require_ood_for_gate: true`
- `ood_benchmark_min_classes: 3`

Practical meaning:

- if OOD evidence is required and missing, the final readiness artifact fails
- if no real `ood/` split exists and the fallback is possible, the workflow runs the held-out benchmark automatically

## Typical Readiness Outcomes

### Outcome 1: ready

This usually means:

- the chosen classification split met the accuracy target
- the required OOD metrics were present
- the OOD metrics met their targets

### Outcome 2: failed because classification quality is not enough

Common reasons:

- accuracy missed the target
- the isolated final evaluation split performed poorly

### Outcome 3: failed because OOD evidence is missing

Common reasons:

- no real `ood/` split exists
- fewer than 3 classes were available for the fallback benchmark

### Outcome 4: failed because OOD behavior is below target

Common reasons:

- AUROC too low
- false positive rate too high
- too few in-distribution or OOD evaluation samples to treat the OOD metrics as sufficient evidence
- SURE or conformal coverage below threshold
- conformal prediction sets are too large on average to be useful
- one completed held-out benchmark fold fell below the required OOD target even if the fold mean looked acceptable

## Method Naming Note

The repo label `SURE+` should be read as `SURE+/DS-F1-inspired double scoring`.

That means:

- the code combines a semantic OOD score and a confidence rejection score
- DS-F1 is the joint reporting metric for that path
- the repo does not advertise this implementation as an exact reproduction of a separately versioned "SURE+" paper unless that citation is pinned explicitly in downstream reporting

## Common Questions

### Why can validation look good while readiness still fails?

Because readiness requires more than good known-class accuracy. It also requires acceptable OOD evidence.

### Why can a run fail when there is no `test/` split?

If `val` was already used for OOD calibration, the workflow refuses to reuse it as final deployment evidence.

### Why is the fallback benchmark expensive?

For `N` known classes, the held-out fallback retrains `N` fold models. That can be costly for exploratory runs.

### When should I add a real `ood/` folder?

Whenever possible. Real unknown examples are the strongest current evidence source.

## BER Note

`training.continual.ood.ber_enabled` is currently a training-only regularizer.

It does not change:

- the readiness artifact schema
- the inference payload shape
- the readiness threshold definitions

Evaluate BER by comparing runs on the same:

- crop
- seed
- class set
- split layout
- OOD evidence source

## Recommended Practice

1. Train on the exact disease classes you intend to support.
2. Keep an isolated `test/` split for final classification evidence.
3. Add a realistic shared `ood/` pool whenever possible.
4. If deployment is crop-gated upstream, prioritize same-crop unknowns and same-crop failure cases inside that pool.
5. Use the fallback benchmark only when real OOD data is missing.
6. Read `production_readiness.json` before deployment.
7. Treat split-local gates as diagnostics, not final approval.

## Related Files

- [../../README.md](../../README.md)
- [colab_training_manual.md](colab_training_manual.md)
- [../architecture/overview.md](../architecture/overview.md)
- [../architecture/ood_recommendation.md](../architecture/ood_recommendation.md)
- [../../src/workflows/training.py](../../src/workflows/training.py)
- [../../src/training/services/metrics.py](../../src/training/services/metrics.py)
- [../../src/training/services/ood_benchmark.py](../../src/training/services/ood_benchmark.py)
