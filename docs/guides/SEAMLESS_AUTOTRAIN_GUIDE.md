# Seamless Auto-Train Guide (v6)

Use the active two-notebook Colab flow:

- `colab_notebooks/1_crop_router_pipeline.ipynb`
- `colab_notebooks/2_interactive_adapter_training.ipynb`

Superseded v6 notebook flow remains archived under `colab_notebooks/archive/v6_superseded_2026-03-02/`.

## Bootstrap Compatibility

- Notebook bootstrap auto-detects repository root from common `/content` and Drive paths.
- If root is still missing, bootstrap can auto-clone the repo.
- Shared bootstrap helpers now live in `scripts/colab_repo_bootstrap.py` and are consumed by both active notebooks.
- Optional environment variables:
  - `AADS_REPO_ROOT` or `REPO_ROOT`
  - `AADS_REPO_CLONE_TARGET`
  - `AADS_REPO_URL`
  - `AADS_DISABLE_AUTO_CLONE=1`
