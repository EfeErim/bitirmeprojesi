# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 5.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 3b: Guncelleme ve Erisim Kontrolu"):
    from scripts.colab_repo_bootstrap import collect_notebook_access_report, print_notebook_access_report

    backbone_model = str(dict(BASE_CONFIG.get("training", {}).get("continual", {})).get("backbone", {}).get("model_name", "")).strip()
    access_report = collect_notebook_access_report(
        repo_root=ROOT,
        hf_model_ids=[backbone_model] if str(backbone_model).strip() else [],
    )
    STATE["access_report"] = access_report
    print_notebook_access_report(access_report, print_fn=print)
    TELEMETRY.update_latest(
        {
            "phase": "access_checked",
            "github_read_access": access_report.get("github", {}).get("read_access_mode"),
            "repo_update_relation": access_report.get("repo_updates", {}).get("relation"),
            "hf_access_mode": access_report.get("huggingface", {}).get("access_mode"),
        }
    )
