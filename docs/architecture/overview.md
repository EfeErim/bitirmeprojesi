# AADS v6 Architecture Overview

## Runtime Flow

1. Router inference via `src/router/vlm_pipeline.py` and `src/router/simple_crop_router.py`.
2. Per-crop continual adapter execution via `src/adapter/independent_crop_adapter.py`.
3. Continual training engine via `src/training/continual_sd_lora.py`.
4. OOD scoring using `src/ood/continual_ood.py` with ensemble score.
5. Multi-crop orchestration via `src/pipeline/independent_multi_crop_pipeline.py`.

## Training Contract

- `training.continual.backbone.model_name = facebook/dinov3-giant`
- `training.continual.quantization.mode = int8_hybrid`
- `training.continual.adapter.target_modules_strategy = all_linear_transformer`
- `training.continual.fusion.layers = [2,5,8,11]`
