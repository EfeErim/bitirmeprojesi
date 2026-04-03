# Data Workspace

This repo keeps Notebook 0 and Notebook 2 dataset staging paths under `data/`, but the actual dataset contents are local working files, not tracked source.

Canonical local paths:

- `data/class_root_dataset/`: repo-local flat class-root dataset for Notebook 0 audit and Notebook 2 `class_root` mode
- `data/prepared_class_root_datasets/`: Notebook 0 cleaned working copies created from audit reports
- `data/prepared_runtime_datasets/`: Notebook 0 materialized runtime datasets
- `data/runtime_notebook_datasets/`: Notebook 2 one-pass runtime datasets

Rules:

- Keep only the placeholder `.gitkeep` files tracked in these folders.
- Do not commit real images, runtime splits, manifests, or notebook-generated artifacts under `data/`.
- If you need a different local dataset location, override the notebook parameter instead of changing the tracked scaffold.

Expected Notebook 0 input shape:

```text
data/class_root_dataset/
  <class>/
    <images>
```
