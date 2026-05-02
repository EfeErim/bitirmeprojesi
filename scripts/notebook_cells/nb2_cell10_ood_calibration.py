# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 10.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 7: OOD Calibration"):
    if STATE.get("adapter") is None or STATE.get("loaders") is None:
        raise RuntimeError("Once engine init hucresini calistirin.")

    adapter = STATE["adapter"]
    val_loader = STATE["loaders"].get("val")
    if val_loader is None or len(val_loader.dataset) == 0:
        raise RuntimeError("Validation loader bos; OOD kalibrasyonu yapilamaz.")

    calibration = adapter.calibrate_ood(val_loader)
    STATE["calibration"] = calibration

    num_classes = calibration.get("ood_calibration", {}).get("num_classes", 0)
    version = calibration.get("ood_calibration", {}).get("version", 0)
    print(f"[OOD] Kalibrasyon tamamlandi. classes={num_classes} version={version}")
    TELEMETRY.update_latest(
        {
            "phase": "ood_calibrated",
            "ood_num_classes": num_classes,
            "ood_version": version,
        }
    )