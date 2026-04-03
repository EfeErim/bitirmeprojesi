# Data Workspace

This repo keeps the maintained notebook dataset paths under `data/`. Most dataset contents are local working files rather than tracked source, but the repo currently includes two checked-in Notebook 0 example datasets:

- `data/class_root_dataset/grape_fruit/`
- `data/class_root_dataset/grape_leaf/`

Use these paths:

- `data/class_root_dataset/`: flat class-root datasets for Notebook 0 audit and Notebook 2 `class_root` mode
- `data/prepared_class_root_datasets/`: Notebook 0 cleaned working copies created from audit reports
- `data/prepared_runtime_datasets/`: prepared runtime datasets used by Notebook 0 materialization and Notebook 2 `runtime` mode

Practical flow:

1. Put raw flat datasets under `data/class_root_dataset/<dataset_name>/`.
2. Run Notebook 0 if you want audit-first cleanup and optional materialization.
3. Train from `data/prepared_runtime_datasets/<dataset_key>/` when you already have a prepared runtime layout.

Rules:

- Keep `data/prepared_class_root_datasets/` and `data/prepared_runtime_datasets/` as local-only working areas unless you explicitly intend to change repo policy.
- Do not commit runtime splits, manifests, or notebook-generated artifacts under `data/`.
- If you need a different local dataset location, override the notebook parameter instead of changing the tracked scaffold.

Expected class-root input shape:

```text
data/class_root_dataset/<dataset_name>/
  <class>/
    <images>
```

Expected prepared runtime shape:

```text
data/prepared_runtime_datasets/<dataset_key>/
  continual/<class>/*
  val/<class>/*
  test/<class>/*
  ood/*
```

For the checked-in grape datasets, point Notebook 0 or Notebook 2 `class_root` mode at one of these roots:

- `data/class_root_dataset/grape_fruit`
- `data/class_root_dataset/grape_leaf`
