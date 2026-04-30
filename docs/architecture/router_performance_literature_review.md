# Router Performance Literature Review

This note reviews literature relevant to improving the maintained AADS v6 router. It is scoped to the current repo design: router-driven inference first identifies crop and plant part, then `RouterAdapterRuntime` loads one crop adapter. The current implementation is not a paper-faithful reproduction of any single method; it is an engineering pipeline that combines SAM3 region proposals, BioCLIP-2.5 prompt scoring, crop-part compatibility, and conservative abstention.

## Current Repo Behavior

The maintained router path is:

```text
src/workflows/inference.py
-> src/pipeline/router_adapter_runtime.py
-> src/router/router_pipeline.py
-> src/router/sam3_runtime.py
-> src/router/roi_pipeline.py
```

Important current choices:

- `RouterAdapterRuntime` rejects weak crop handoffs with `inference.router_min_confidence` and `inference.router_min_margin` before adapter load.
- `RouterPipeline` uses SAM3 plus BioCLIP-2.5. The older `VLM` name remains a compatibility surface, not a separate maintained method.
- SAM3 prompts default to `plant` plus supported part prompts, capped by `sam3_prompt_limit`.
- BioCLIP scoring uses prompt ensembles and caches OpenCLIP text embeddings.
- Crop choice fuses ROI crop scores with whole-image crop context through `global_crop_context_weight`.
- Part choice is restricted to the configured compatible part surface and may abstain to `unknown`.
- `scripts/evaluate_router_part_surface.py` evaluates crop accuracy, part precision/recall, abstention rate, unsupported part emissions, and part-threshold sweeps on `data/router_part_eval/<crop>/<part>/*`.
- `scripts/evaluate_router_surface.py` evaluates the full crop/part handoff on `data/router_eval/`, including ID crop/part samples, off-crop and non-plant negatives, ambiguous inputs, and wrong-part cases.

## Literature Signals

### 1. Keep The Router Object-Centric, But Calibrate It On Repo Data

SAM3 is designed for promptable concept segmentation: it detects and segments objects from short noun phrases, exemplars, or both, and its architecture separates recognition from localization. This supports the repo choice to use SAM3 as a region-proposal front end rather than treating the whole image as the only evidence source. See [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719).

Open-vocabulary localization literature points in the same direction. GLIP unifies object detection and phrase grounding, showing strong zero-shot and few-shot object-level transfer from language-aware pretraining. Grounding DINO similarly frames open-set detection as language-guided object detection, where user-provided category names or referring expressions drive localization. See [GLIP](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html) and [Grounding DINO](https://arxiv.org/abs/2303.05499).

Engineering inference for this repo: do not replace the router with a single whole-image crop classifier as the first improvement. The better near-term path is to keep object-centric routing and add better evaluation, calibration, and prompt evidence around it.

### 2. Global Plus Local Evidence Is The Right Plant-Vision Bias

Plant disease recognition literature repeatedly finds that local symptom evidence and global plant context carry complementary information. LGNet explicitly motivates dual local/global branches because CNN-like local perception and transformer-like global perception each miss part of the signal. The glocal tomato anomaly work similarly separates local anomaly regions from whole-scene context. See [LGNet](https://pubmed.ncbi.nlm.nih.gov/39130161/) and [Deep Learning-Based Phenotyping System With Glocal Description of Plant Anomalies and Symptoms](https://www.frontiersin.org/articles/10.3389/fpls.2019.01321/full).

Engineering inference for this repo: the current `global_crop_context_weight` is conceptually justified, but its value should be data-calibrated. The repo should sweep this weight alongside `router_min_confidence`, `router_min_margin`, SAM3 threshold, prompt limits, and part-abstention thresholds.

### 3. BioCLIP-2.5 Is A Good Backbone, But Its Scores Still Need Local Calibration

BioCLIP and BioCLIP 2 are trained specifically for biological imagery and hierarchical biological labels, which is a better fit for plant routing than generic CLIP alone. The BioCLIP 2 paper reports that scaling hierarchical contrastive learning on TreeOfLife-200M yields biologically meaningful embeddings. The BioCLIP-2.5 model card reports improved zero-shot species and broader biological-task performance over BioCLIP 2, and specifically lists PlantDoc as one evaluation task. See [BioCLIP](https://arxiv.org/abs/2311.18803), [BioCLIP 2](https://arxiv.org/abs/2505.23883), and the [BioCLIP-2.5 model card](https://huggingface.co/imageomics/bioclip-2.5-vith14).

CLIP-style models support zero-shot classification through natural-language prompts, and prompt ensembling is a standard part of practical CLIP usage. See [Learning Transferable Visual Models From Natural Language Supervision](https://huggingface.co/papers/2103.00020).

Engineering inference for this repo: BioCLIP-2.5 should remain the main scoring model, but its softmax scores should not be treated as calibrated probabilities. The router should calibrate score thresholds on repo-local held-out router data.

### 4. Abstention Is A Feature, But It Must Be Measured As Risk-Coverage

The current router already abstains through `unknown_crop`, `router_uncertain`, and part `unknown`. This is aligned with selective prediction: systems should reject when evidence is weak or ambiguous instead of forcing a label. SelectiveNet shows that explicitly optimizing reject behavior can improve the risk-coverage trade-off over simple confidence thresholding, though the repo currently uses post-hoc gates rather than training a reject head. See [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).

Calibration literature also matters because the router gates on confidence-like scores. Guo et al. show that modern neural network confidence can be poorly calibrated and that post-hoc temperature scaling is a practical baseline. See [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).

Conformal prediction is relevant if the desired product behavior becomes "return a small set of plausible crops/parts" rather than "return one label or abstain." Conformal methods provide distribution-free set guarantees under the usual exchangeability conditions, but they require a representative calibration split. See [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).

Engineering inference for this repo: the next router evaluation should report risk-coverage curves, not only top-1 crop accuracy. A router that improves top-1 accuracy by forcing labels at low confidence may be worse for this product.

### 5. Evaluation Quality Comes Before Method Complexity

OOD evaluation literature warns that benchmark construction can distort conclusions when evaluation samples overlap with in-distribution classes or fail to represent real failure modes. See [In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation](https://proceedings.mlr.press/v202/bitterwolf23a.html).

Engineering inference for this repo: the same warning applies to router evaluation. The repo should add a router evaluation set with clean crop/part labels, realistic off-crop and non-plant negatives, and known ambiguous cases. Without that surface, changing thresholds or adding another model is likely to optimize anecdotal examples rather than deployment behavior.

## Recommended Performance Plan

### Phase 1: Measure Router Performance More Completely

Extend the current part-only evaluation surface into a full router surface:

```text
data/router_eval/
  id/<crop>/<part>/*
  negatives/off_crop/<label>/*
  negatives/non_plant/<label>/*
  ambiguous/<reason>/*
  wrong_part/<crop>/<unsupported_part>/*
```

Metrics to report:

- crop top-1 accuracy on ID samples
- crop risk-coverage curve using `router_uncertain` as rejection
- crop false-accept rate on off-crop and non-plant negatives
- part non-unknown precision and recall
- part abstention rate
- unsupported part emission count
- mean and p95 router latency
- SAM3 instance count, ROI count, and BioCLIP scoring calls

This phase should also persist raw router evidence needed for analysis: top-k crop scores, top-k part scores, global crop scores, ROI crop scores, SAM3 prompt provenance when available, quality score, and rejection reasons.

### Phase 2: Calibrate Existing Gates Before Adding Models

Use `scripts/calibrate_router_surface.py` on the eval surface to sweep multiple parameters against the same dataset while reusing one loaded SAM3/BioCLIP router instance:

```powershell
.\scripts\python.cmd scripts/calibrate_router_surface.py --root data/router_eval --config-env colab --preset quick --output .runtime_tmp/router_calibration.json
```

The script supports built-in `handoff`, `quick`, and `docs` presets plus explicit `--sweep PARAM=v1,v2` overrides. It ranks variants by negative false-accept rate, unsupported part emissions, wrong-part rejection, crop accuracy, part precision/recall, abstention, and latency while enforcing configurable accuracy and precision drop guardrails.

The sweepable config paths include:

- `inference.router_min_confidence`
- `inference.router_min_margin`
- `router.vlm.confidence_threshold`
- `part_open_set_min_confidence`
- `part_open_set_margin`
- `global_crop_context_weight`
- `sam3_mask_threshold`
- `sam3_prompt_limit`
- `crop_num_prompts`
- `part_num_prompts`
- `max_rois_for_classification`

Accept a configuration only if it improves risk-coverage or false-accept behavior without materially damaging crop ID accuracy or part precision. This should be a benchmarked config update, not a hand-tuned one-image fix.

### Phase 3: Improve Prompt Evidence

Low-risk prompt work:

- complete crop aliases and scientific names for all configured crops, not only the current partial alias map
- add organ-conditioned crop prompts such as `{crop} leaf`, `{crop} fruit`, `{crop} stem`
- evaluate part prompts with and without disease/symptom terms
- keep prompt templates small enough that latency remains acceptable

Use ablation tables because prompt changes can shift both accuracy and calibration.

### Phase 4: Add Few-Shot BioCLIP Prototype Evidence If Data Supports It

BioCLIP-2.5 supports zero-shot and few-shot classification use cases. If the repo has clean reference images for supported crops and parts, add an optional generated router prototype cache:

```text
.runtime_tmp/router_prototypes/<config_hash>.json
```

or a tracked config-described path outside generated runtime outputs if the user wants reproducible deployment artifacts.

The crop decision would then fuse:

- prompt-score evidence
- whole-image crop context
- ROI crop evidence
- optional image-embedding nearest-prototype evidence

This should remain optional until it beats the calibrated prompt-only router on the same eval set.

### Phase 5: Consider A Learned Reject Head Only After The Data Surface Is Stable

Selective prediction literature supports learned rejection, but this repo currently has no router-training workflow. A learned reject head would add training data, artifact, and deployment complexity. Keep it as a later research step after calibrated thresholds, risk-coverage metrics, and prompt/prototype evidence are stable.

## Near-Term Recommendation

The best next implementation path is not a new model. It is a router benchmark and calibration pass:

1. Use `scripts/evaluate_router_surface.py` to evaluate the full `data/router_eval/` handoff surface.
2. Persist raw crop and part evidence per sample.
3. Use `scripts/calibrate_router_surface.py` to sweep thresholds and evidence weights.
4. Update router defaults only when the sweep improves crop risk-coverage, part precision/recall, and negative false-accept behavior on the same eval set.

This fits the current repo architecture, is supported by the literature above, and avoids replacing a reasonable SAM3 + BioCLIP-2.5 design before measuring where it actually fails.

## References

- Carion et al. (2025). [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719).
- Gu et al. (2025). [BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](https://arxiv.org/abs/2505.23883).
- Stevens et al. (2024). [BioCLIP: A Vision Foundation Model for the Tree of Life](https://arxiv.org/abs/2311.18803).
- Imageomics (2026). [BioCLIP-2.5 Huge model card](https://huggingface.co/imageomics/bioclip-2.5-vith14).
- Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://huggingface.co/papers/2103.00020).
- Li et al. (2022). [Grounded Language-Image Pre-Training](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html).
- Liu et al. (2023). [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499).
- Lin et al. (2024). [Local and Global Feature-Aware Dual-Branch Networks for Plant Disease Recognition](https://pubmed.ncbi.nlm.nih.gov/39130161/).
- Fuentes et al. (2019). [Deep Learning-Based Phenotyping System With Glocal Description of Plant Anomalies and Symptoms](https://www.frontiersin.org/articles/10.3389/fpls.2019.01321/full).
- Guo et al. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- Geifman and El-Yaniv (2019). [SelectiveNet: A Deep Neural Network with an Integrated Reject Option](https://proceedings.mlr.press/v97/geifman19a.html).
- Angelopoulos and Bates (2021). [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).
- Bitterwolf et al. (2023). [In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation](https://proceedings.mlr.press/v202/bitterwolf23a.html).
