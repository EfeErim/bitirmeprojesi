# Unknown Disease Rejection Research Note

This note summarizes the current repo position and the most relevant literature for one narrow problem:

- can a crop adapter say `unknown` or reject an image that belongs to a disease class it was not trained to support?

Important scope note:

- this is not exactly the same as generic far-OOD detection on random non-plant images
- the harder and more relevant case is semantically near unknowns, such as unsupported diseases of the same crop
- the recommendations below are an engineering synthesis from the cited sources plus the current repo design; they are not a claim that the repo already implements every cited method faithfully

## Problem Framing

For this repo, the important deployment question is usually:

- can the adapter avoid a confident wrong disease prediction when it sees an unsupported disease or a realistic failure case for the same crop?

That places the problem between three related literatures:

1. Open-set recognition:
   known classes at train time, unknown semantic classes at test time
2. OOD detection:
   score whether an input is outside the training distribution
3. Selective prediction:
   abstain when confidence is not trustworthy

In practice, the repo needs all three ideas, but the core problem is best treated as semantic open-set rejection rather than only generic far-OOD filtering.

## Where The Repo Is Now

Current maintained behavior:

- the adapter trains a closed-set classifier on supported disease classes
- after training, the repo calibrates an adapter-side OOD detector and saves that state with the exported adapter
- the detector calibrates three candidate score families together:
  - `ensemble`: Mahalanobis-style feature distance plus energy z-score
  - `energy`: raw energy score
  - `knn`: class-local nearest-neighbor distance
- the shipped config keeps `primary_score_method: "auto"` at the public surface, but under real `ood/` evidence the workflow keeps the runtime detector on `ensemble` instead of auto-tuning on the same final evidence
- if no real `ood/` split exists, the workflow falls back to a leave-one-class-out held-out benchmark and can use that proxy evidence for score selection
- the repo also exposes:
  - SURE+/DS-F1-inspired double scoring as diagnostic evidence
  - conformal thresholding or APS/RAPS set-valued prediction
  - LogitNorm as the default training loss
  - BER as an optional incremental-training regularizer when explicitly enabled with cross-entropy

Relevant current files:

- `src/ood/continual_ood.py`
- `src/training/services/ood_benchmark.py`
- `src/training/services/metrics.py`
- `config/base.json`

Short assessment:

- this is already beyond a simple MSP or raw-softmax threshold baseline
- the repo currently sits in the "strong classifier plus post-hoc detector" family
- that is a reasonable modern baseline, especially with a strong pretrained backbone
- the remaining weakness is not that the repo lacks an OOD method entirely; the remaining weakness is mostly about semantic unknown evidence quality and method selection on that evidence

## What The Literature Says

### 1. Strong closed-set classifiers matter more than many people expect

Vaze et al. argue that open-set recognition performance is strongly tied to closed-set classification quality and that a strong closed-set classifier can already be highly competitive for OSR tasks.

Source:

- Vaze et al., "Open-Set Recognition: a Good Closed-Set Classifier is All You Need?", ICLR 2022
  https://openreview.net/forum?id=5hLP5JY9S2d

Implication for this repo:

- the pretrained DINOv3-based adapter stack is a strength, not a liability
- representation quality and fine-tuning style can matter as much as the reject score itself

### 2. MSP is a baseline, not the endpoint

The standard maximum softmax probability baseline remains historically important, but more robust score families generally outperform it on realistic OOD settings.

Sources:

- Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks", ICLR 2017
  https://openreview.net/forum?id=Hkg4TI9xl
- Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018
  https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html
- Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020
  https://nips.cc/virtual/2020/public/poster_f5496252609c43eb8a3d147ab9b9c006.html
- Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors", ICML 2022
  https://proceedings.mlr.press/v162/sun22d.html

Implication for this repo:

- keeping `ensemble`, `energy`, and `knn` as the maintained candidates is well aligned with mainstream literature
- `energy` and `deep kNN` deserve first-class evaluation on semantic unknown disease cases, not only generic OOD slices

### 3. Evaluation quality is critical

Bitterwolf et al. show that OOD rankings can move substantially when benchmark construction changes, and that flawed OOD evaluation can create misleading conclusions.

Source:

- Bitterwolf et al., "In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation", ICML 2023
  https://proceedings.mlr.press/v202/bitterwolf23a.html

Implication for this repo:

- the current real-OOD guardrail is justified
- the final real `ood/` split should not double as free model-selection evidence
- the most important benchmark slice is not random far-OOD; it is near-OOD unknown diseases and realistic same-crop failure cases

### 4. Auxiliary outlier data can help even when it is small

Outlier Exposure showed that training with auxiliary outliers can improve rejection of unseen anomalies, and later work argues that even small auxiliary outlier sets can be useful.

Sources:

- Hendrycks et al., "Deep Anomaly Detection with Outlier Exposure", ICLR 2019
  https://openreview.net/forum?id=HyxCxhRcY7
- Liznerski et al., "Exposing Outlier Exposure: What Can Be Learned From Few, One, and Zero Outlier Images", TMLR/OpenReview 2022
  https://openreview.net/forum?id=3v78awEzyB

Implication for this repo:

- a small curated auxiliary unknown set is worth testing
- it does not need to be exhaustive to be useful
- this is a better next experiment than trying to construct a perfect universal unknown dataset

### 5. Calibration and abstention are product-relevant

If the product goal is "speak only when reliable", selective prediction and conformal prediction are directly relevant.

Sources:

- Geifman and El-Yaniv, "SelectiveNet: A Deep Neural Network with an Integrated Reject Option", ICML 2019
  https://proceedings.mlr.press/v97/geifman19a.html
- Angelopoulos et al., "Uncertainty Sets for Image Classifiers using Conformal Prediction", ICLR 2021
  https://openreview.net/forum?id=eNdiU_DbM9

Implication for this repo:

- `conformal_method: "aps"` or `"raps"` is most useful when the desired behavior is abstention or small prediction sets
- conformal prediction improves uncertainty presentation, but it does not remove the need for semantic unknown evaluation

### 6. Overconfidence control is a realistic training lever

LogitNorm directly targets overconfidence and is relatively low-risk in a standard classification pipeline.

Source:

- Wei et al., "Mitigating Neural Network Overconfidence with Logit Normalization", ICML 2022
  https://proceedings.mlr.press/v162/wei22d.html

Implication for this repo:

- LogitNorm is now the shipped default; BER and plain cross-entropy remain controlled comparison paths

### 7. Plant-disease-specific evidence suggests fine-tuning style matters

Recent plant-disease work focused on unknown disease recognition indicates that the fine-tuning paradigm itself can materially affect unknown disease rejection.

Source:

- Dong et al., "The impact of fine-tuning paradigms on unknown plant diseases recognition", Scientific Reports, 2024
  https://www.nature.com/articles/s41598-024-66958-2

Implication for this repo:

- the main research axis is not only "which score threshold?"
- representation and fine-tuning strategy should be treated as first-class variables in unknown disease rejection studies

## What Seems Most Defensible For This Repo

Given the current architecture, the most defensible near-term path is:

1. Keep the maintained post-hoc detector stack.
2. Prioritize semantic unknown evaluation slices.
3. Compare `ensemble`, `energy`, and `knn` explicitly on those slices.
4. Keep a controlled cross-entropy or BER ablation if LogitNorm hurts a crop-specific run.
5. If needed, add small-scale auxiliary outlier exposure.
6. Use APS/RAPS when the desired product behavior is abstention rather than a binary OOD-only flag.

This is better aligned with the current repo than a full jump to:

- OpenMax-style explicit open-set heads
- one-class methods such as Deep SVDD
- a large redesign toward a separate reject model family

Those approaches are literature-backed, but they carry more implementation and product risk than the current repo likely needs first.

## Recommended Evaluation Protocol

The most useful evaluation protocol for unknown disease rejection in this repo is:

### ID splits

- one clean `continual/` training split for supported classes
- one calibration split for OOD thresholding and conformal calibration
- one isolated final in-distribution classification split

### Unknown slices

Use separate unknown evidence slices and report them separately:

1. same-crop unsupported diseases
2. same-crop non-disease failure cases
3. nearby off-crop plant images
4. far non-plant or random images

Priority order:

- same-crop unsupported diseases are the headline metric
- same-crop failure cases are next
- far-OOD random images are only secondary diagnostics

### Metrics

At minimum report:

- closed-set accuracy, balanced accuracy, macro-F1
- OOD AUROC
- FPR at high TPR
- abstention or coverage metrics when using conformal or selective prediction
- worst-slice performance

### Selection discipline

- do not select the final primary score method on the same final real unknown evidence that is later used as the headline deployment claim
- if method selection is needed, use separate proxy evidence or rerun explicitly after analysis

## Concrete Repo-Level Recommendations

### Low-risk next experiments

1. Run the same crop and split with explicit:
   - `primary_score_method="ensemble"`
   - `primary_score_method="energy"`
   - `primary_score_method="knn"`
2. Use the same semantic unknown disease slice across all three.
3. Compare worst-slice behavior, not only pooled AUROC.

### Medium-risk experiments

1. Compare the LogitNorm default against an explicit `loss_name="cross_entropy"` baseline.
2. Add a small auxiliary outlier set for OE-style training.
3. Re-run the same method comparison on the same unknown slices.

### Higher-risk experiments

Consider these only if the previous steps are insufficient:

- explicit reject-head architectures
- OpenMax-style open-set heads
- one-class redesigns

## Bottom Line

Current repo position:

- already stronger than simple softmax-threshold OOD handling
- well aligned with modern post-hoc OOD literature
- not obviously missing a mainstream score family

Main gap:

- semantic unknown disease evidence quality and disciplined comparison on that evidence

Best next move:

- focus on unknown disease rejection as a semantic near-OOD benchmark
- compare `ensemble`, `energy`, and `knn` explicitly
- compare the LogitNorm default against cross-entropy and small auxiliary outlier ablations

If the repo needs one sentence of guidance, it is this:

- do not spend effort chasing a perfect universal OOD split; spend effort building a clean same-crop unknown disease benchmark and comparing the current maintained score families on it.
