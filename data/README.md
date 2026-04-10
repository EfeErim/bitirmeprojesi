# Data Workspace

This repo keeps the maintained notebook dataset paths under `data/`. Most dataset contents are local working files rather than tracked source, but the repo currently includes three checked-in example dataset trees:

- `data/class_root_dataset/grape_fruit/`
- `data/class_root_dataset/grape_leaf/`
- `data/ood_dataset/grape_fruit_ood_trial/`

Use these paths:

- `data/class_root_dataset/`: flat class-root datasets for Notebook 0 audit and Notebook 2 `class_root` mode
- `data/ood_dataset/`: repo-local OOD pools that Notebook 0 and Notebook 2 can materialize into runtime `ood/`
- `data/prepared_class_root_datasets/`: Notebook 0 cleaned working copies created from audit reports
- `data/prepared_runtime_datasets/`: prepared runtime datasets used by Notebook 0 materialization and Notebook 2 `runtime` mode

Practical flow:

1. Put raw flat datasets under `data/class_root_dataset/<dataset_name>/`.
2. Optionally put reusable OOD pools under `data/ood_dataset/<dataset_name>/`.
3. Run Notebook 0 if you want audit-first cleanup and optional materialization.
4. Train from `data/prepared_runtime_datasets/<dataset_key>/` when you already have a prepared runtime layout.

Rules:

- Keep `data/ood_dataset/` for reusable OOD pools only. Nested folders are for organization, not class labels.
- Keep `data/prepared_class_root_datasets/` as a local-only working area.
- `data/prepared_runtime_datasets/` is still ignored by default, but Notebook 0 can force-add and push a ready runtime dataset when `SAVE_RUNTIME_DATASET_TO_GITHUB=True`.
- Do not commit other runtime splits, manifests, or notebook-generated artifacts under `data/` unless the notebook is explicitly pushing the prepared runtime dataset for reuse.
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

Expected repo-local OOD pool shape:

```text
data/ood_dataset/<dataset_name>/
  unsupported_same_crop/*
  blur_or_occlusion/*
  other_crops_optional/*
```

For the checked-in grape class-root datasets, point Notebook 0 or Notebook 2 `class_root` mode at one of these roots:

- `data/class_root_dataset/grape_fruit`
- `data/class_root_dataset/grape_leaf`

For the checked-in reusable OOD example pool, point Notebook 0 or Notebook 2 at:

- `data/ood_dataset/grape_fruit_ood_trial`
