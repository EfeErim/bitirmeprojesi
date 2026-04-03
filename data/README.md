# Data Workspace

This repo keeps Notebook 0 and Notebook 2 dataset staging paths under `data/`. Most dataset contents remain local working files rather than tracked source, but this repo currently includes two checked-in Notebook 0 class-root datasets:

- `data/class_root_dataset/grape_fruit/`
- `data/class_root_dataset/grape_leaf/`

Canonical local paths:

- `data/class_root_dataset/`: repo-local flat class-root dataset for Notebook 0 audit and Notebook 2 `class_root` mode
- `data/prepared_class_root_datasets/`: Notebook 0 cleaned working copies created from audit reports
- `data/prepared_runtime_datasets/`: Notebook 0 materialized runtime datasets
- `data/runtime_notebook_datasets/`: Notebook 2 one-pass runtime datasets

Rules:

- Keep `data/prepared_class_root_datasets/`, `data/prepared_runtime_datasets/`, and `data/runtime_notebook_datasets/` as local-only working areas unless you explicitly intend to change repo policy.
- Do not commit runtime splits, manifests, or notebook-generated artifacts under `data/`.
- If you need a different local dataset location, override the notebook parameter instead of changing the tracked scaffold.

Expected Notebook 0 input shape:

```text
data/class_root_dataset/
  <class>/
    <images>
```

For the checked-in grape datasets, point Notebook 0 `DATASET_ROOT` at one of these exact roots:

- `data/class_root_dataset/grape_fruit`
- `data/class_root_dataset/grape_leaf`
