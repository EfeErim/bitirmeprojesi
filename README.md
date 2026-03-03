# AADS v6

AADS v6 is a continual-only SD-LoRA training and inference stack.

## Runtime Contract

- Backbone: `facebook/dinov3-vitl16-pretrain-lvd1689m` (frozen)
- Quantization: `int8_hybrid`
- Adapter targeting: `all_linear_transformer`
- Fusion layers: `[2, 5, 8, 11]`
- OOD score: `0.6 * mahalanobis_z + 0.4 * energy_z`

## Canonical Entrypoints

- Colab Notebook 1 (router pipeline): `colab_notebooks/1_crop_router_pipeline.ipynb`
- Colab Notebook 2 (interactive adapter training): `colab_notebooks/2_interactive_adapter_training.ipynb`
- Superseded notebook archive: `colab_notebooks/archive/v6_superseded_2026-03-02/`
- Local validation: `python scripts/validate_notebook_imports.py`

## Colab Bootstrap Behavior

- Notebook bootstrap cells auto-detect repository root from common `/content` and Drive paths.
- Optional overrides:
  - `AADS_REPO_ROOT` or `REPO_ROOT` for explicit repo path.
  - `AADS_REPO_CLONE_TARGET` for clone destination when auto-clone runs.
  - `AADS_REPO_URL` for alternate repository URL.
  - `AADS_DISABLE_AUTO_CLONE=1` to disable auto-clone fallback.

## Test Commands

```powershell
python scripts/run_test_suites.py --suite unit --suite colab --suite integration/core
python scripts/check_markdown_links.py --root .
```

## Legacy Archive

v5.5 materials are archived under:

- `docs/archive/v5_legacy/`
- `colab_notebooks/archive/v5_legacy/`
- `src/archive/v5_legacy/`

