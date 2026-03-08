from pathlib import Path

from scripts import recalibrate_adapter_ood as recalibrate


class _FakeAdapter:
    def __init__(self, crop_name: str, device: str = "cpu"):
        self.crop_name = crop_name
        self.device = device
        self.loaded_from = ""
        self.calibrated_with = None
        self.saved_to = ""

    def load_adapter(self, adapter_dir: str) -> None:
        self.loaded_from = adapter_dir

    def calibrate_ood(self, loader):
        self.calibrated_with = loader
        return {"status": "calibrated", "ood_calibration": {"version": 1, "num_classes": 11}}

    def save_adapter(self, output_dir: str) -> Path:
        self.saved_to = output_dir
        return Path(output_dir) / "continual_sd_lora_adapter"


def test_run_recalibration_prefers_val_loader_and_saves_to_export_parent(monkeypatch, tmp_path: Path):
    asset_dir = tmp_path / "telemetry" / "run_1" / "artifacts" / "adapter" / "continual_sd_lora_adapter"
    asset_dir.mkdir(parents=True)
    adapter = _FakeAdapter("tomato", device="cpu")
    train_loader = [{"images": 1, "labels": 1}, {"images": 2, "labels": 2}]
    val_loader = [{"images": 3, "labels": 3}]

    monkeypatch.setattr(recalibrate, "get_config", lambda environment=None: {
        "training": {"continual": {"batch_size": 8, "seed": 42, "data": {}}},
        "colab": {"training": {"num_workers": 0, "pin_memory": False}},
    })
    monkeypatch.setattr(recalibrate, "_resolve_adapter_dir", lambda crop_name, adapter_dir=None, config_env=None: asset_dir)
    monkeypatch.setattr(recalibrate, "_resolve_crop_name", lambda crop_name, adapter_dir=None: "tomato")
    monkeypatch.setattr(recalibrate, "create_training_loaders", lambda **kwargs: {"train": train_loader, "val": val_loader})
    monkeypatch.setattr(recalibrate, "IndependentCropAdapter", lambda crop_name, device: adapter)
    monkeypatch.setattr(recalibrate, "_save_adapter_in_place", lambda adapter_obj, resolved_dir: resolved_dir)

    result = recalibrate.run_recalibration(
        adapter_ref=asset_dir,
        data_dir=tmp_path / "dataset",
        crop_name="tomato",
        config_env="colab",
        device="cpu",
    )

    assert result["calibration_split"] == "val"
    assert adapter.calibrated_with is val_loader
    assert result["adapter_output_dir"] == str(asset_dir)
    assert result["overwritten_in_place"] is True


def test_run_recalibration_falls_back_to_train_loader(monkeypatch, tmp_path: Path):
    asset_dir = tmp_path / "adapter_export" / "continual_sd_lora_adapter"
    asset_dir.mkdir(parents=True)
    adapter = _FakeAdapter("tomato", device="cpu")
    train_loader = [{"images": 1, "labels": 1}]

    monkeypatch.setattr(recalibrate, "get_config", lambda environment=None: {
        "training": {"continual": {"batch_size": 8, "seed": 42, "data": {}}},
        "colab": {"training": {"num_workers": 0, "pin_memory": False}},
    })
    monkeypatch.setattr(recalibrate, "_resolve_adapter_dir", lambda crop_name, adapter_dir=None, config_env=None: asset_dir)
    monkeypatch.setattr(recalibrate, "_resolve_crop_name", lambda crop_name, adapter_dir=None: "tomato")
    monkeypatch.setattr(recalibrate, "create_training_loaders", lambda **kwargs: {"train": train_loader, "val": []})
    monkeypatch.setattr(recalibrate, "IndependentCropAdapter", lambda crop_name, device: adapter)
    monkeypatch.setattr(recalibrate, "_save_adapter_in_place", lambda adapter_obj, resolved_dir: resolved_dir)

    result = recalibrate.run_recalibration(
        adapter_ref=asset_dir,
        data_dir=tmp_path / "dataset",
        crop_name="tomato",
        config_env="colab",
        device="cpu",
    )

    assert result["calibration_split"] == "train"
    assert adapter.calibrated_with is train_loader


def test_run_recalibration_searches_folder_for_adapter(monkeypatch, tmp_path: Path):
    search_root = tmp_path / "downloads"
    search_root.mkdir()
    discovered_dir = search_root / "telemetry" / "run_2" / "artifacts" / "adapter"
    discovered_dir.mkdir(parents=True)
    adapter = _FakeAdapter("tomato", device="cpu")
    val_loader = [{"images": 3, "labels": 3}]

    monkeypatch.setattr(recalibrate, "get_config", lambda environment=None: {
        "training": {"continual": {"batch_size": 8, "seed": 42, "data": {}}},
        "colab": {"training": {"num_workers": 0, "pin_memory": False}},
    })

    def _raise_direct(*args, **kwargs):
        raise FileNotFoundError("direct resolve failed")

    monkeypatch.setattr(recalibrate, "_resolve_adapter_dir", _raise_direct)
    monkeypatch.setattr(
        recalibrate,
        "discover_adapter_candidates",
        lambda search_roots=None, crop_name=None: [
            {
                "adapter_dir": str(discovered_dir),
                "crop_name": "tomato",
                "ood_calibration_version": 0,
            }
        ],
    )
    monkeypatch.setattr(recalibrate, "_resolve_crop_name", lambda crop_name, adapter_dir=None: "tomato")
    monkeypatch.setattr(recalibrate, "create_training_loaders", lambda **kwargs: {"train": [], "val": val_loader})
    monkeypatch.setattr(recalibrate, "IndependentCropAdapter", lambda crop_name, device: adapter)
    monkeypatch.setattr(recalibrate, "_save_adapter_in_place", lambda adapter_obj, resolved_dir: resolved_dir)

    result = recalibrate.run_recalibration(
        adapter_ref=search_root,
        data_dir=tmp_path / "dataset",
        crop_name="tomato",
        config_env="colab",
        device="cpu",
    )

    assert result["adapter_input_dir"] == str(discovered_dir)
    assert adapter.loaded_from == str(discovered_dir)


def test_save_adapter_in_place_overwrites_non_nested_adapter_dir(tmp_path: Path):
    target_dir = tmp_path / "telemetry" / "run_1" / "artifacts" / "adapter"
    target_dir.mkdir(parents=True)
    (target_dir / "adapter_meta.json").write_text('{"old": true}', encoding="utf-8")

    class _SavingAdapter:
        def save_adapter(self, output_dir: str) -> Path:
            bundle = Path(output_dir) / "continual_sd_lora_adapter"
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / "adapter_meta.json").write_text('{"new": true}', encoding="utf-8")
            (bundle / "classifier.pth").write_text("weights", encoding="utf-8")
            return bundle

    saved_dir = recalibrate._save_adapter_in_place(_SavingAdapter(), target_dir)

    assert saved_dir == target_dir
    assert (target_dir / "classifier.pth").exists()
    assert (target_dir / "adapter_meta.json").read_text(encoding="utf-8") == '{"new": true}'
