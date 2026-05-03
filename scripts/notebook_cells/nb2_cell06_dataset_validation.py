# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 6.
# Runs Notebook 2 dataset validation in the notebook namespace.

with TELEMETRY.capture_cell_output("Cell 4: Dataset Validation"):
    from scripts.colab_dataset_layout import list_repo_dataset_directories, resolve_direct_repo_dataset_root, resolve_repo_relative_root
    from scripts.colab_training_recommendations import inspect_runtime_dataset

    crop_key = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(CROP_NAME).strip())
    while "__" in crop_key:
        crop_key = crop_key.replace("__", "_")
    crop_key = crop_key.strip("_")
    if not crop_key:
        raise RuntimeError("CROP_NAME bos olmayan bir crop anahtarina cozulmeli.")

    runtime_parent = resolve_repo_relative_root(repo_root=ROOT, repo_relative_root=RUNTIME_DATASET_ROOT)
    direct_runtime_dataset = resolve_direct_repo_dataset_root(
        repo_root=ROOT,
        repo_relative_root=RUNTIME_DATASET_ROOT,
    )
    runtime_dirs = [] if direct_runtime_dataset is not None else list_repo_dataset_directories(
        repo_root=ROOT,
        repo_relative_root=RUNTIME_DATASET_ROOT,
    )
    candidates = []
    if direct_runtime_dataset is not None:
        direct_runtime_name, direct_runtime_path = direct_runtime_dataset
        candidates.append({"name": direct_runtime_name, "path": direct_runtime_path, "parent": runtime_parent})
    else:
        candidates.extend(
            {"name": path.name, "path": path, "parent": runtime_parent}
            for path in runtime_dirs
        )
    if not candidates:
        raise RuntimeError("No prepared runtime datasets were found under RUNTIME_DATASET_ROOT. Notebook 0'u once calistirin.")

    requested_dataset_name = str(DATASET_NAME).strip()
    if requested_dataset_name:
        matches = [item for item in candidates if item["name"] == requested_dataset_name]
        if not matches:
            available_options = [item['name'] for item in candidates]
            raise RuntimeError(
                f"Requested dataset '{requested_dataset_name}' was not found. Available options: {available_options}"
            )
        selected = matches[0]
    elif len(candidates) == 1:
        selected = candidates[0]
        print(f"[DATASET] Yalnizca bir runtime dataset bulundu, otomatik secildi: {selected['name']}")
    else:
        print("[DATASET] Kullanilabilir runtime dataset secenekleri:")
        for index, item in enumerate(candidates, start=1):
            print(f"  [{index}] {item['name']} ({item['path']})")
        raw_choice = str(input(f"Kullanilacak dataset icin numara ya da isim girin (1-{len(candidates)}): ")).strip()
        if not raw_choice:
            raise RuntimeError("Dataset secimi bos birakilamaz.")
        if raw_choice.isdigit():
            selected_index = int(raw_choice) - 1
            if selected_index < 0 or selected_index >= len(candidates):
                raise RuntimeError(f"Dataset secim index'i aralik disi: {raw_choice}")
            selected = candidates[selected_index]
        else:
            matches = [item for item in candidates if item['name'] == raw_choice]
            if not matches:
                raise RuntimeError(f"Dataset secimi bulunamadi: {raw_choice}")
            selected = matches[0]

    selected_dataset_name = str(selected['name'])
    selected_dataset_root = Path(selected['path'])
    if not selected_dataset_name.startswith(crop_key):
        raise RuntimeError(
            f"Secilen runtime dataset CROP_NAME ile uyusmuyor: {selected_dataset_name} vs {crop_key}"
        )
    missing_splits = [name for name in ("continual", "val", "test") if not (selected_dataset_root / name).is_dir()]
    if missing_splits:
        raise RuntimeError(f"Prepared runtime dataset is missing split folder(s): {missing_splits}")
    class_names = sorted(d.name for d in (selected_dataset_root / "continual").iterdir() if d.is_dir())
    if not class_names:
        raise RuntimeError(f"No class subdirectories in prepared runtime split: {selected_dataset_root / 'continual'}")
    runtime_root = selected_dataset_root.parent
    default_ood_root = selected_dataset_root / "ood"
    default_oe_root = selected_dataset_root / "oe"
    requested_ood_root = str(OOD_ROOT or "").strip()
    requested_oe_root = str(OE_ROOT or "").strip()
    oe_enabled = bool(OE_ENABLED)
    if bool(ASK_FOR_OOD_ROOT) and not requested_ood_root:
        default_hint = str(default_ood_root) if default_ood_root.is_dir() else ""
        prompt = "OOD klasoru yolunu girin"
        if default_hint:
            prompt += f" [Enter={default_hint}]"
        requested_ood_root = str(input(prompt + ": ")).strip()
        if not requested_ood_root and default_hint:
            requested_ood_root = default_hint
    if oe_enabled and bool(ASK_FOR_OE_ROOT) and not requested_oe_root:
        default_hint = str(default_oe_root) if default_oe_root.is_dir() else ""
        prompt = "OE klasoru yolunu girin"
        if default_hint:
            prompt += f" [Enter={default_hint}]"
        requested_oe_root = str(input(prompt + ": ")).strip()
        if not requested_oe_root and default_hint:
            requested_oe_root = default_hint

    if requested_ood_root:
        resolved_ood_root = Path(requested_ood_root).expanduser()
        if not resolved_ood_root.is_absolute():
            resolved_ood_root = (ROOT / resolved_ood_root).resolve()
        if not resolved_ood_root.is_dir():
            raise RuntimeError(f"OOD klasoru bulunamadi veya klasor degil: {resolved_ood_root}")
        print(f"[OOD] explicit ood root={resolved_ood_root}")
        resolved_ood_root_value = str(resolved_ood_root)
    elif default_ood_root.is_dir():
        print(f"[OOD] runtime ood root={default_ood_root}")
        resolved_ood_root_value = str(default_ood_root)
    else:
        print("[OOD] Gercek OOD split secilmedi; fallback held-out benchmark kullanilabilir.")
        resolved_ood_root_value = ""
    if not oe_enabled:
        if requested_oe_root:
            print("[OE] OE_ENABLED=False; OE_ROOT yok sayildi.")
        resolved_oe_root_value = ""
    elif requested_oe_root:
        resolved_oe_root = Path(requested_oe_root).expanduser()
        if not resolved_oe_root.is_absolute():
            resolved_oe_root = (ROOT / resolved_oe_root).resolve()
        if not resolved_oe_root.is_dir():
            raise RuntimeError(f"OE klasoru bulunamadi veya klasor degil: {resolved_oe_root}")
        print(f"[OE] explicit oe root={resolved_oe_root}")
        resolved_oe_root_value = str(resolved_oe_root)
    elif default_oe_root.is_dir():
        print(f"[OE] runtime oe root={default_oe_root}")
        resolved_oe_root_value = str(default_oe_root)
    else:
        raise RuntimeError(
            "OE_ENABLED=True ama OE klasoru cozulmedi. OE_ROOT girin, runtime dataset altina oe/ ekleyin, "
            "veya OE_ENABLED=False yapin."
        )
    print(f"[DATASET] runtime root={selected_dataset_root} classes={len(class_names)}: {class_names}")

    dataset_inspection = inspect_runtime_dataset(selected_dataset_root, ood_root=resolved_ood_root_value or None)
    effective_params = dict(STATE.get("effective_params") or _collect_notebook_base_params())

    split_totals = dict(dataset_inspection.get("split_totals", {}))
    print(
        f"[DATASET][CHECK] scale={dataset_inspection.get('dataset_scale_bucket', 'unknown')} "
        f"continual={split_totals.get('continual', 0)} val={split_totals.get('val', 0)} "
        f"test={split_totals.get('test', 0)} ood={split_totals.get('ood', 0)} "
        f"classes={dataset_inspection.get('class_count', 0)}"
    )

    blockers = list(dataset_inspection.get("blockers", []))
    warnings = list(dataset_inspection.get("warnings", []))
    for item in warnings:
        print(f"[DATASET][WARN] {item}")
    for item in blockers:
        print(f"[DATASET][BLOCK] {item}")
    if MANUAL_PARAM_OVERRIDES:
        print(f"[PARAMS] Manual overrides uygulandi: {MANUAL_PARAM_OVERRIDES}")
    print(
        f"[PARAMS][FINAL] epochs={effective_params['EPOCHS']} bs={effective_params['BATCH_SIZE']} "
        f"lr={effective_params['LEARNING_RATE']} lora={effective_params['LORA_R']}/{effective_params['LORA_ALPHA']} "
        f"dropout={effective_params['LORA_DROPOUT']} accum={effective_params['GRAD_ACCUM_STEPS']} "
        f"workers={effective_params['NUM_WORKERS']} prefetch={effective_params['PREFETCH']}"
    )

    STATE["class_names"] = class_names
    STATE["validated"] = True
    STATE["runtime_dataset_root"] = runtime_root
    STATE["runtime_dataset_key"] = selected_dataset_name
    STATE["selected_dataset_name"] = selected_dataset_name
    STATE["selected_dataset_root"] = selected_dataset_root
    STATE["resolved_ood_root"] = resolved_ood_root_value
    STATE["resolved_oe_root"] = resolved_oe_root_value
    STATE["dataset_inspection"] = dataset_inspection
    STATE["hardware_inspection"] = {}
    STATE["recommendation_report"] = {}
    STATE["recommendation_decision"] = "disabled"
    STATE["effective_params"] = effective_params
    TELEMETRY.update_latest(
        {
            "phase": "dataset_validated",
            "dataset_root": str(selected_dataset_root),
            "runtime_dataset_root": str(runtime_root),
            "runtime_dataset_key": selected_dataset_name,
            "selected_dataset_name": selected_dataset_name,
            "resolved_ood_root": resolved_ood_root_value,
            "resolved_oe_root": resolved_oe_root_value,
            "class_count": len(class_names),
            "recommendation_decision": "disabled",
            "recommendation_change_count": 0,
        }
    )
