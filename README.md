# AADS v6

This repo now keeps only two supported flows:

- `colab_notebooks/2_interactive_adapter_training.ipynb` for Colab training
- `colab_notebooks/1_router_adapter_inference.ipynb` or `scripts/colab_router_adapter_inference.py` for router-driven inference

The inference runtime is `src/pipeline/router_adapter_runtime.py`. The training runtime is `src/training/continual_sd_lora.py` plus `src/adapter/independent_crop_adapter.py`.

## Quick Start

Install minimal local dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the lightweight validation surface:

```powershell
python scripts/validate_notebook_imports.py
pytest tests/unit tests/colab/test_smoke_training.py -q
pytest tests/integration -q --runintegration
```

## Colab

- Root Colab dependencies live in `requirements_colab.txt`
- Notebook bootstrap helpers live in `scripts/colab_repo_bootstrap.py`
- Telemetry and checkpoints live in `scripts/colab_live_telemetry.py` and `scripts/colab_checkpointing.py`

## Adapter Layout

Inference expects adapters under:

```text
models/adapters/<crop>/continual_sd_lora_adapter/
```
