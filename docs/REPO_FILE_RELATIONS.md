# Repository File Relations (v6)

- Training engine: `src/training/continual_sd_lora.py`
- Adapter lifecycle: `src/adapter/independent_crop_adapter.py`
- OOD engine: `src/ood/continual_ood.py`
- Pipeline: `src/pipeline/independent_multi_crop_pipeline.py`
- Colab scripts: `scripts/colab_auto_orchestrator.py`
- Colab entrypoint: `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb` (includes repo-root auto-discovery and auto-clone fallback)
