# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 6.
# Runs Notebook 2 dataset validation in the notebook namespace.

with TELEMETRY.capture_cell_output("Cell 4: Dataset Validation"):
    from scripts.colab_dataset_layout import list_repo_dataset_directories, resolve_direct_repo_dataset_root, resolve_repo_relative_root
    from scripts.colab_training_recommendations import (
        inspect_runtime_dataset,
        inspect_runtime_hardware,
        recommend_notebook_training_params,
        resolve_effective_notebook_params,
    )

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
    requested_ood_root = str(OOD_ROOT or "").strip()
    if bool(ASK_FOR_OOD_ROOT) and not requested_ood_root:
        default_hint = str(default_ood_root) if default_ood_root.is_dir() else ""
        prompt = "OOD klasoru yolunu girin"
        if default_hint:
            prompt += f" [Enter={default_hint}]"
        requested_ood_root = str(input(prompt + ": ")).strip()
        if not requested_ood_root and default_hint:
            requested_ood_root = default_hint

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
    print(f"[DATASET] runtime root={selected_dataset_root} classes={len(class_names)}: {class_names}")

    base_params = _collect_notebook_base_params()
    dataset_inspection = inspect_runtime_dataset(selected_dataset_root, ood_root=resolved_ood_root_value or None)
    hardware_inspection = inspect_runtime_hardware(DEVICE)
    recommendation_report = recommend_notebook_training_params(base_params, dataset_inspection, hardware_inspection)

    split_totals = dict(dataset_inspection.get("split_totals", {}))
    print(
        f"[RECOMMEND][DATASET] scale={dataset_inspection.get('dataset_scale_bucket', 'unknown')} "
        f"continual={split_totals.get('continual', 0)} val={split_totals.get('val', 0)} "
        f"test={split_totals.get('test', 0)} ood={split_totals.get('ood', 0)} "
        f"classes={dataset_inspection.get('class_count', 0)}"
    )
    print(
        f"[RECOMMEND][HW] device={hardware_inspection.get('effective_device', 'cpu')} "
        f"gpu={hardware_inspection.get('gpu_name') or 'none'} "
        f"vram_gb={hardware_inspection.get('total_vram_gb')} cpu_count={hardware_inspection.get('cpu_count', 0)}"
    )

    blockers = list(recommendation_report.get("blockers", []))
    warnings = list(recommendation_report.get("warnings", []))
    for item in warnings:
        print(f"[RECOMMEND][WARN] {item}")
    for item in blockers:
        print(f"[RECOMMEND][BLOCK] {item}")

    accepted_recommendations = False
    recommendation_decision = "no_changes"
    if recommendation_report.get("has_changes"):
        print("[RECOMMEND] Onerilen degisiklikler:")
        for key in sorted(recommendation_report.get("changes", {})):
            change = recommendation_report["changes"][key]
            print(f"  - {key}: {change['current']} -> {change['recommended']} | {change['reason']}")
        if blockers:
            recommendation_decision = "blocked"
            print("[RECOMMEND] Blocker oldugu icin otomatik uygulama sorusu atlandi. Gerekirse MANUAL_PARAM_OVERRIDES ile acik override girin.")
        else:
            raw_confirm = str(input("Apply recommended parameters? [y/N]: ")).strip().lower()
            accepted_recommendations = raw_confirm in {"y", "yes"}
            recommendation_decision = "accepted" if accepted_recommendations else "rejected"
    elif blockers:
        recommendation_decision = "blocked"
        print("[RECOMMEND] Blocker var; notebook ham parametrelerle devam eder. Gerekirse MANUAL_PARAM_OVERRIDES guncellenmeli.")
    else:
        print("[RECOMMEND] Mevcut notebook parametreleri onerilen degerlerle zaten uyumlu.")

    effective_params = resolve_effective_notebook_params(
        base_params,
        recommendation_report,
        MANUAL_PARAM_OVERRIDES,
        accepted=accepted_recommendations,
    )
    if MANUAL_PARAM_OVERRIDES:
        print(f"[RECOMMEND] Manual overrides uygulandi: {MANUAL_PARAM_OVERRIDES}")
    print(
        f"[RECOMMEND][FINAL] epochs={effective_params['EPOCHS']} bs={effective_params['BATCH_SIZE']} "
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
    STATE["dataset_inspection"] = dataset_inspection
    STATE["hardware_inspection"] = hardware_inspection
    STATE["recommendation_report"] = recommendation_report
    STATE["recommendation_decision"] = recommendation_decision
    STATE["effective_params"] = effective_params
    TELEMETRY.update_latest(
        {
            "phase": "dataset_validated",
            "dataset_root": str(selected_dataset_root),
            "runtime_dataset_root": str(runtime_root),
            "runtime_dataset_key": selected_dataset_name,
            "selected_dataset_name": selected_dataset_name,
            "resolved_ood_root": resolved_ood_root_value,
            "class_count": len(class_names),
            "recommendation_decision": recommendation_decision,
            "recommendation_change_count": int(recommendation_report.get('change_count', 0)),
        }
    )
