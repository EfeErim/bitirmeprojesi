# Repository File Relations (v6)

- Training engine: `src/training/continual_sd_lora.py`
- Low-bit policy guardrails: `src/training/quantization.py`
- Fusion module: `src/adapter/multi_scale_fusion.py`
- Adapter lifecycle: `src/adapter/independent_crop_adapter.py`
- OOD engine: `src/ood/continual_ood.py`
- Pipeline: `src/pipeline/independent_multi_crop_pipeline.py`
- Router core: `src/router/vlm_pipeline.py`
- Router support utilities: `src/router/roi_pipeline.py`, `src/router/roi_helpers.py`, `src/router/policy_taxonomy_utils.py`
- Colab orchestration: `scripts/colab_auto_orchestrator.py`
- Colab bootstrap helper: `scripts/colab_repo_bootstrap.py`
- Dataset layout contract checker (notebook 2): `scripts/evaluate_dataset_layout.py`
- Modular suite runner: `scripts/run_test_suites.py`
- Integration smoke (real HF backbone, heavy-model marker): `tests/integration/test_continual_trainer_real_backbone.py`
- Colab notebook 1 (router): `colab_notebooks/1_crop_router_pipeline.ipynb`
- Colab notebook 2 (training): `colab_notebooks/2_interactive_adapter_training.ipynb`
- Colab notebook archive (superseded): `colab_notebooks/archive/v6_superseded_2026-03-02/`
