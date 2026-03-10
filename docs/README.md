# Documentation Map

Use this index to find the maintained docs and to keep generated artifacts out of version control.

## Start Here

- [../README.md](../README.md): repo overview, quick start, and entrypoints
- [user_guide/colab_training_manual.md](user_guide/colab_training_manual.md): Notebook 2 and Notebook 3 usage, runtime dataset layout, and Colab-specific outputs
- [user_guide/ood_readiness_guide.md](user_guide/ood_readiness_guide.md): how `production_readiness.json` is produced and how to prepare `data/<crop>/ood/`
- [architecture/overview.md](architecture/overview.md): training, inference, and OOD component map
- [architecture/ood_recommendation.md](architecture/ood_recommendation.md): next-step OOD improvement notes

## Tracked Vs Local Files

Tracked source-of-truth files:

- `src/`, `tests/`, `scripts/`, `config/`
- `docs/` and the root `README.md`
- `colab_notebooks/*.ipynb`
- root dependency files such as `requirements.txt` and `requirements_colab.txt`

Local/generated files that should stay out of git:

- `runs/<RUN_ID>/`: mirrored notebook run exports, telemetry copies, plots, logs, and checkpoint manifests
- `models/adapters/<crop>/continual_sd_lora_adapter/`: deployed adapter bundles used by inference
- `outputs/`: local notebook and workflow outputs
- `.runtime_tmp/`, `.tmp*/`, caches, and virtualenvs

## Notes On Similar-Looking Files

- `requirements_colab.txt` is the canonical Colab dependency list.
- `colab_notebooks/requirements_colab.txt` is an intentional one-line wrapper so notebook-local bootstrap code can install dependencies relative to the notebook directory.
- `runs/` is a local export workspace, not a dataset or a reproducible fixture directory.
- `models/adapters/` is the default deployment target for trained adapters, not a place to version model weights.
