# Seamless Auto-Train Guide (v6)

Use `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb` for the canonical continual run.

## Bootstrap Compatibility

- Notebook bootstrap auto-detects repository root from common `/content` and Drive paths.
- If root is still missing, bootstrap can auto-clone the repo.
- Optional environment variables:
  - `AADS_REPO_ROOT` or `REPO_ROOT`
  - `AADS_REPO_CLONE_TARGET`
  - `AADS_REPO_URL`
  - `AADS_DISABLE_AUTO_CLONE=1`
