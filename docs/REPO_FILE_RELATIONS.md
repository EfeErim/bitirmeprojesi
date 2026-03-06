# Repository File Relations (v6)

## Governance Model (P2-B)

- Canonical human-maintained summary: `docs/REPO_FILE_RELATIONS.md` (this file).
- Generated detailed artifact: `docs/REPO_FILE_RELATIONS_DETAILED.md` (generated; do not hand-edit).
- Canonical generation owner script: `scripts/generate_repo_relationships.py`.
- Canonical regeneration command:

```bash
python scripts/generate_repo_relationships.py --output docs/REPO_FILE_RELATIONS_DETAILED.md --json-output docs/reports/repository_relationships_snapshot.json
```

- Operational guardrail: avoid regenerating the detailed artifact during small governance/documentation updates unless explicitly requested.

## Canonical Active Relationships Summary

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
- Colab live telemetry helper: `scripts/colab_live_telemetry.py`
- Colab checkpoint manager: `scripts/colab_checkpointing.py`
- Colab notebook artifact/checkpoint helpers: `scripts/colab_notebook_helpers.py`
- Colab reconnect investigation helper: `scripts/colab_reconnect_probe.py`
- Dataset layout contract checker (notebook 2): `scripts/evaluate_dataset_layout.py`
- Modular suite runner: `scripts/run_test_suites.py`
- Integration smoke (real HF backbone, heavy-model marker): `tests/integration/test_continual_trainer_real_backbone.py`
- Colab notebook 1 (router): `colab_notebooks/1_crop_router_pipeline.ipynb`
- Colab notebook 2 (training): `colab_notebooks/2_interactive_adapter_training.ipynb`
- Colab notebook archive (superseded): `colab_notebooks/archive/v6_superseded_2026-03-02/`
