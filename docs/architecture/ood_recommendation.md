# OOD Recommendation for This Repo

This document captures my recommendation for improving OOD handling in this repository based on the current codebase and the cited primary literature.

Important: the recommendation below is an engineering inference from the cited sources plus the repo's current design constraints. It is not a direct claim made by any single paper.

## Bottom Line

The best next step is not to replace the current OOD stack. The best next step is to strengthen it with better evidence and a slightly better scoring stack.

Recommended direction:

1. Keep the current production-readiness flow and held-out fallback benchmark.
2. Add a small reusable real `data/<crop>/ood/` pool with realistic hard negatives.
3. Upgrade the scoring path to evaluate an energy-centric score plus a non-parametric nearest-neighbor score alongside the current detector.
4. Add a low-risk overconfidence control such as LogitNorm during training.
5. Only after that, consider OE-style training or a learned reject option.

If only one thing can be done next, it should be step 2: add a small real OOD pool and use it consistently across runs.

## Why This Is The Recommendation

### 1. Real OOD evidence matters more than a more complicated benchmark

The repo already has a good fallback when no real unknown-image set exists:

- train on known classes
- calibrate OOD on known data
- if `data/<crop>/ood/` is missing, run the held-out benchmark
- write the final verdict to `production_readiness.json`

That is a solid baseline, but it is still proxy evidence. A held-out known class is not the same thing as a real unseen disease or a real deployment mistake.

The literature supports the idea that evaluation quality matters a lot. Bitterwolf et al. show that flawed OOD datasets can badly distort conclusions, which implies the converse as well: even a modest but clean OOD set is valuable if it matches the deployment problem well.  
Source: [Bitterwolf et al., 2023, ICML](https://proceedings.mlr.press/v202/bitterwolf23a.html)

Inference for this repo:

- a small clean `ood/` set is more valuable than endlessly extending only the held-out benchmark
- that `ood/` set should emphasize plant-like unknowns and realistic failure cases, not only easy random objects
- if deployment is crop-gated upstream, most of that set should be same-crop unknowns and same-crop failure cases first, with other plant species used as secondary negatives

### 2. A small auxiliary outlier set is literature-backed

Outlier Exposure (OE) is the clearest result here: exposing the model to auxiliary outliers can improve detection of unseen anomalies.  
Source: [Hendrycks et al., 2018, arXiv](https://arxiv.org/abs/1812.04606)

Later work argues that the amount of auxiliary outlier data needed may be much smaller than people assume.  
Source: [Liznerski et al., 2022, OpenReview](https://openreview.net/forum?id=3v78awEzyB)

Inference for this repo:

- you do not need a full catalog of every unseen tomato disease
- a small shared OOD pool can already improve both evaluation quality and, later, OE-style training if desired

### 3. Better OOD scores are available with manageable implementation cost

Energy-based scoring is a strong, simple baseline and improves on raw softmax confidence as an OOD signal.  
Source: [Liu et al., 2020, NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)

Deep nearest neighbors improves OOD detection without assuming a Gaussian feature model, which is relevant because the current repo already uses feature-space statistics and an ensemble score.  
Source: [Sun et al., 2022, ICML](https://proceedings.mlr.press/v162/sun22d.html)

Inference for this repo:

- keep the current Mahalanobis-plus-energy path for continuity
- add energy-only and kNN-style scores as first-class evaluation candidates
- compare them on the same `production_readiness.json` gates and on the same real/fallback OOD evidence

### 4. Reducing overconfidence during training is a cheap improvement

LogitNorm targets overconfidence directly and is simple to add to a cross-entropy training loop.  
Source: [Wei et al., 2022, ICML](https://proceedings.mlr.press/v162/wei22d.html)

Inference for this repo:

- LogitNorm is a better near-term training experiment than a large redesign
- it is likely cheaper to integrate than a full new OOD model family

### 5. Strong pretrained features still matter

Hendrycks et al. show that pretraining improves robustness and uncertainty, including OOD behavior.  
Source: [Hendrycks et al., 2019, ICML](https://proceedings.mlr.press/v97/hendrycks19a.html)

Inference for this repo:

- the current frozen pretrained backbone is a strength, not a weakness
- OOD work should build on that representation instead of replacing the whole stack with a one-class method

### 6. A reject option is useful, but not the first thing to do

SelectiveNet shows value in learning rejection explicitly rather than only thresholding confidence after the fact.  
Source: [Geifman and El-Yaniv, 2019, ICML](https://proceedings.mlr.press/v97/geifman19a.html)

Inference for this repo:

- reject-option modeling is worth considering later
- it should follow better evidence and better scores, not precede them

## Recommended Plan For This Repo

### Phase 1: Improve OOD evidence first

Recommendation:

- keep the current hard gate and held-out fallback exactly as they are
- create a small reusable `data/<crop>/ood/` pool for each crop
- prefer realistic hard negatives:
  - unseen diseases for the crop
  - non-disease damage and same-crop off-coverage views
  - other plant species as secondary negatives when the deployment contract is crop-gated
  - poor crops, blur, occlusion, background clutter
- continue allowing random non-plant images, but only as a minority slice

Why this phase is first:

- it improves confidence in the current system immediately
- it needs no fundamental model redesign
- it makes future model comparisons meaningful

### Phase 2: Upgrade the scoring stack

Recommendation:

- keep the current calibrated ensemble path
- add two evaluation paths:
  - energy-only OOD score
  - deep-kNN OOD score on fused features
- compare all three:
  - current ensemble
  - energy
  - kNN

Success criterion:

- whichever score wins should do so on the same readiness targets and the same OOD evidence source

Why this phase is second:

- it is likely lower risk than training-objective changes
- it fits the repo's current calibration-and-artifact architecture well

### Phase 3: Add low-risk training regularization

Recommendation:

- run a controlled experiment with LogitNorm in the classifier loss path
- accept it only if it preserves or improves known-class accuracy while improving OOD rejection

Why this phase is third:

- it is easier than introducing full OE training immediately
- it addresses overconfidence, which is one of the central failure modes in OOD detection

### Phase 4: Add OE-style training only if needed

Recommendation:

- if Phase 1 and Phase 2 are still not strong enough, add OE-style auxiliary outlier exposure
- start with the same small shared `ood/` pool
- if real negatives are scarce, consider synthetic outliers as a supplement rather than a replacement

Relevant source for synthetic/constructed calibration data:

- [Narayanaswamy et al., 2024, MIDL](https://proceedings.mlr.press/v227/narayanaswamy24a.html)

Why this phase is later:

- OE is powerful, but it changes training behavior, not just evaluation
- the repo should first establish a clean real-OOD evaluation surface

### Phase 5: Consider a learned reject option

Recommendation:

- evaluate an explicit reject head or selective-classification variant only after the score/data path is stable

Why this phase is last:

- it adds product and training complexity
- it is easier to reason about once the base detector and evidence pipeline are already strong

## What I Do Not Recommend Right Now

### Do not try to build an exhaustive unseen-disease dataset

That is not realistic. The literature does not require coverage of every unknown class in the world. What matters is representative unknowns.

### Do not rely only on random easy negatives

Cars, houses, and pets are valid OOD samples, but by themselves they can make the detector look better than it really is.

### Do not replace the current classifier with a pure one-class method

For this repo, the main task is still multi-class disease recognition with rejection. A one-class-only redesign would be a larger departure than the evidence supports.

### Do not rely forever on held-out-class benchmarking alone

The held-out benchmark is good and worth keeping, but it should remain fallback evidence, not the only long-term evidence source.

## Recommended Success Criteria

I recommend treating the following as the decision framework:

1. Known-class accuracy must remain at or above the current target.
2. Real OOD evidence should be preferred over fallback evidence whenever `data/<crop>/ood/` exists.
3. Score upgrades must beat the current detector on the same OOD set, not on a different benchmark.
4. Training changes must not improve OOD by materially damaging closed-set disease accuracy.

In repo terms, the relevant artifacts remain:

- `validation/metric_gate.json`
- `test/metric_gate.json`
- `ood_benchmark/summary.json`
- `production_readiness.json`

## Final Recommendation

If I had to choose one concrete roadmap for this repo, it would be:

1. Keep the current hard-gated readiness flow.
2. Build a small reusable real `ood/` pool for each crop.
3. Add energy and deep-kNN as competing OOD scores.
4. Trial LogitNorm.
5. Add OE only if the earlier steps are still insufficient.

That path is the best balance of:

- literature support
- engineering cost
- compatibility with the current repo
- and practical improvement in deployment confidence

## References

- Bitterwolf, J., Muller, M., and Hein, M. (2023). *In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation*. ICML. https://proceedings.mlr.press/v202/bitterwolf23a.html
- Geifman, Y., and El-Yaniv, R. (2019). *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. ICML. https://proceedings.mlr.press/v97/geifman19a.html
- Hendrycks, D., Lee, K., and Mazeika, M. (2019). *Using Pre-Training Can Improve Model Robustness and Uncertainty*. ICML. https://proceedings.mlr.press/v97/hendrycks19a.html
- Hendrycks, D., Mazeika, M., and Dietterich, T. (2018). *Deep Anomaly Detection with Outlier Exposure*. arXiv. https://arxiv.org/abs/1812.04606
- Liu, W., Wang, X., Owens, J., and Li, Y. (2020). *Energy-based Out-of-distribution Detection*. NeurIPS. https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
- Liznerski, P., Ruff, L., Vandermeulen, R. A., Franks, B. J., Muller, K.-R., and Kloft, M. (2022). *Exposing Outlier Exposure: What Can Be Learned From Few, One, and Zero Outlier Images*. OpenReview. https://openreview.net/forum?id=3v78awEzyB
- Narayanaswamy, V., Mubarka, Y., Anirudh, R., Rajan, D., Spanias, A., and Thiagarajan, J. J. (2024). *Know Your Space: Inlier and Outlier Construction for Calibrating Medical OOD Detectors*. MIDL. https://proceedings.mlr.press/v227/narayanaswamy24a.html
- Sun, Y., Ming, Y., Zhu, X., and Li, Y. (2022). *Out-of-Distribution Detection with Deep Nearest Neighbors*. ICML. https://proceedings.mlr.press/v162/sun22d.html
- Wei, H., Xie, R., Cheng, H., Feng, L., An, B., and Li, Y. (2022). *Mitigating Neural Network Overconfidence with Logit Normalization*. ICML. https://proceedings.mlr.press/v162/wei22d.html
