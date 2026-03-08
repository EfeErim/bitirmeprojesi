# Architecture

The repo is intentionally narrow.

## Training

- `colab_notebooks/2_interactive_adapter_training.ipynb`
- `src/workflows/training.py`
- `src/adapter/independent_crop_adapter.py`
- `src/training/continual_sd_lora.py`
- `src/training/session.py`
- `src/training/validation.py`
- `src/training/types.py`
- `src/data/`
- `scripts/colab_checkpointing.py`
- `scripts/colab_dataset_layout.py`
- `scripts/colab_live_telemetry.py`
- `scripts/colab_notebook_helpers.py`

Training is split into four layers:

1. `TrainingWorkflow.run(...)` is the canonical orchestration entrypoint for the supported adapter-training flow.
2. `ContinualSDLoRATrainer` owns model initialization, LoRA wrapping, optimizer setup, batch stepping, snapshot/restore, OOD calibration, and adapter save/load, while `src/training/services/` holds shared metric/runtime/persistence helpers.
3. `ContinualTrainingSession` owns epoch/batch orchestration, resume, validation timing, checkpoint requests, best-metric updates, early stopping, observer events, and history accumulation.
4. `evaluate_model(...)` computes validation metrics outside the trainer loop, while notebook helpers persist reports, confusion matrices, and the metric gate artifact (`accuracy`, OOD AUROC, FPR@95TPR, and optional SURE+/conformal checks when supplied).

## LoRA Training Details

The supported training path uses one backbone and one adapter strategy only: continual SD-LoRA.

At a high level:

1. `ContinualSDLoRATrainer` loads a pretrained backbone and freezes its base parameters.
2. The trainer resolves target transformer modules and applies PEFT LoRA wrappers to those linear layers.
3. Intermediate backbone layers are sampled and fused by `MultiScaleFeatureFusion`.
4. A classifier head is trained on top of the fused representation for the current crop classes.
5. When new classes are added, the classifier is expanded with old weights copied forward and the optimizer is rebuilt around the new trainable head.
6. The saved adapter bundle includes LoRA weights, classifier weights, fusion weights, config metadata, and serialized OOD state.

Configuration for this path lives in `ContinualSDLoRAConfig` and includes:

- backbone model name
- LoRA rank / alpha / dropout
- target module selection strategy
- fusion layer selection and output dimension
- optimizer, scheduler, mixed precision, and early stopping controls

The implementation surface for these details is `src/training/continual_sd_lora.py`, while adapter packaging and runtime metadata are handled by `src/adapter/independent_crop_adapter.py` and `src/training/services/persistence.py`.

Notebook 2 now uses a two-stage dataset contract:

1. User input is a flat class-root directory `<root>/<class>/<images>`.
2. `prepare_runtime_dataset_layout(...)` materializes `data/runtime_notebook_datasets/<crop>/{continual,val,test}/<class>` and writes `split_manifest.json`.

The session emits stable observer payloads. `batch_end` exposes `loss`, and epoch/validation payloads carry the validation summary plus the history snapshot used for artifact persistence and best-checkpoint decisions.

Checkpoint payloads persist the normalized trainer contract, optimizer state, scheduler state, scaler state, best-metric state, optimizer-step counters, RNG state, and serialized OOD calibration state. Adapter bundles persist LoRA weights, classifier/fusion weights, and the public metadata contract in one place.

## OOD Details

OOD behavior is part of the canonical adapter runtime, not a separate model family.

The current production path works like this:

1. Train the adapter on all known classes (optionally with BER loss wrapping CE).
2. Run OOD calibration on known-class data after training.
3. Persist per-class and detector-level calibration statistics with the adapter.
4. During inference, score the predicted class with an ensemble OOD detector and optional conformal set output.

The detector base ensemble combines two signals:

- Mahalanobis distance in fused feature space, normalized as a z-score per class
- energy score from classifier logits, also normalized as a z-score per class

The runtime ensemble is `0.6 * mahalanobis_z + 0.4 * energy_z`. Each class gets its own threshold derived from the calibrated ensemble distribution.

Extended OOD stack details:

- Radially scaled L2 normalization can auto-tune $\beta$ during calibration and normalize features before Mahalanobis statistics are estimated.
- SURE+ double scoring computes semantic OOD score (ensemble-based) and confidence rejection score (`1 - max softmax`), then applies calibrated percentile thresholds.
- Conformal prediction calibrates a global nonconformity quantile $\hat{q}$ and produces `conformal_set` at inference time.
- BER (Bi-directional Energy Regularization) is training-only and applies separate old/new energy penalties via a composable loss wrapper.
- The fully extended calibration path materializes one feature/logit snapshot and reuses it across radial, SURE+, and conformal phases to avoid repeated loader rescans.

When SURE+ is enabled, the final OOD decision uses combined semantic/confidence rejection; conformal output is returned as an additional prediction-set field, not as a replacement for top-1 diagnosis.

The relevant repo mapping is:

- detector logic: `src/ood/continual_ood.py`
- BER loss wrapper: `src/training/ber_loss.py`
- radial normalization: `src/ood/radial_normalization.py`
- SURE+ scoring: `src/ood/sure_scoring.py`
- conformal prediction: `src/ood/conformal_prediction.py`
- calibration orchestration: `src/training/services/ood_calibration.py`
- OOD metric computation and gate checks: `src/training/services/metrics.py`
- inference payload normalization: `src/pipeline/inference_payloads.py`

Experimental ideas that are intentionally not part of the canonical path are tracked separately. The current repository note for held-out-class OOD benchmarking lives in [experimental_leave_one_class_out_ood.md](experimental_leave_one_class_out_ood.md).

Output locations are surface-specific:

- Notebook 2 writes adapter + notebook artifacts under `outputs/colab_notebook_training/` and writes rolling checkpoints to Drive telemetry (`.../telemetry/<RUN_ID>/checkpoints/`).
- `TrainingWorkflow.run(...)` (including `python -m src.app.cli training ...`) writes adapter and metrics under the provided `<output_dir>` (`<output_dir>/continual_sd_lora_adapter/` and `<output_dir>/training_metrics/`).

## Inference

- `src/workflows/inference.py`
- `src/pipeline/router_adapter_runtime.py`
- `src/pipeline/inference_payloads.py`
- `src/router/vlm_pipeline.py`
- `scripts/colab_router_adapter_inference.py`
- `colab_notebooks/1_router_adapter_inference.ipynb`

Inference is one path only:

1. `InferenceWorkflow.predict(...)` is the canonical entrypoint.
2. Router resolves the crop.
3. Runtime loads that crop adapter lazily.
4. Adapter returns diagnosis and OOD payload through a typed inference contract.

By default inference resolves adapters from `models/adapters/<crop>/continual_sd_lora_adapter/`, unless `adapter_root` / `--adapter-root` is provided.

## Adapter Validation

- `scripts/colab_adapter_smoke_test.py`
- `colab_notebooks/3_adapter_smoke_test.ipynb`

This direct-adapter surface is separate from router inference:

1. Resolve one crop adapter from either an explicit export path or a deployed adapter root.
2. Load the adapter bundle directly through `IndependentCropAdapter`.
3. Inspect the saved metadata contract.
4. Run a one-image smoke prediction or a small folder sanity pass without router involvement.
