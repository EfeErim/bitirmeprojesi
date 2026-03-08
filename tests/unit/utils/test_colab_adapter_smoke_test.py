from pathlib import Path

import torch
from PIL import Image

from scripts import colab_adapter_smoke_test as smoke


class _FakeAdapter:
    def __init__(self, crop_name: str, device: str = "cpu"):
        self.crop_name = crop_name
        self.device = device
        self.loaded_paths: list[str] = []

    def load_adapter(self, adapter_dir: str) -> None:
        self.loaded_paths.append(adapter_dir)

    def get_summary(self):
        return {
            "crop_name": self.crop_name,
            "engine": "continual_sd_lora",
            "schema_version": "v6",
            "is_trained": True,
            "num_classes": 2,
            "class_to_idx": {"healthy": 0, "blight": 1},
            "ood_calibration_version": 3,
        }

    def predict_with_ood(self, image):
        return {
            "status": "success",
            "disease": {"class_index": 1, "name": "blight", "confidence": 0.91},
            "ood_analysis": {
                "is_ood": False,
                "ensemble_score": 0.2,
                "class_threshold": 0.8,
                "calibration_version": 3,
            },
        }


def _write_adapter_export(root: Path) -> Path:
    asset_dir = root / "continual_sd_lora_adapter"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "adapter_meta.json").write_text(
        """
        {
          "schema_version": "v6",
          "engine": "continual_sd_lora",
          "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
          "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
          "class_to_idx": {"healthy": 0, "blight": 1},
          "ood_calibration": {"version": 3},
          "target_modules_resolved": ["encoder.layer.0.attention.q_proj"],
          "adapter_runtime": {"adapter_wrapped": true}
        }
        """.strip(),
        encoding="utf-8",
    )
    return asset_dir


def _write_adapter_export_with_crop_info(root: Path, crop_name: str = "tomato") -> Path:
    asset_dir = _write_adapter_export(root)
    (asset_dir / "crop_info.json").write_text(
        f'{{"crop": "{crop_name}", "run_id": "run_local", "export_source": "notebook_2"}}',
        encoding="utf-8",
    )
    return asset_dir


def _write_drive_adapter_export(root: Path, crop_name: str = "tomato") -> Path:
    asset_dir = root / "artifacts" / "adapter"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "crop_info.json").write_text(
        f'{{"crop": "{crop_name}", "run_id": "{root.name}"}}',
        encoding="utf-8",
    )
    (asset_dir / "adapter_meta.json").write_text(
        """
        {
          "schema_version": "v6",
          "engine": "continual_sd_lora",
          "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
          "fusion": {"layers": [2, 5, 8, 11], "output_dim": 768, "dropout": 0.1, "gating": "softmax"},
          "class_to_idx": {"healthy": 0, "blight": 1},
          "ood_calibration": {"version": 3},
          "target_modules_resolved": ["encoder.layer.0.attention.q_proj"],
          "adapter_runtime": {"adapter_wrapped": true}
        }
        """.strip(),
        encoding="utf-8",
    )
    return asset_dir


def test_load_adapter_summary_accepts_parent_export_dir(monkeypatch, tmp_path: Path):
    export_root = tmp_path / "adapter_export"
    asset_dir = _write_adapter_export(export_root)
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_dir=export_root, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["backbone_model_name"] == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert summary["class_names"] == ["healthy", "blight"]
    assert summary["target_modules_resolved"] == ["encoder.layer.0.attention.q_proj"]


def test_load_adapter_summary_accepts_asset_dir(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_dir=asset_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["class_count"] == 2


def test_load_adapter_summary_accepts_adapter_meta_file(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    meta_path = asset_dir / "adapter_meta.json"
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_dir=meta_path, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)


def test_load_adapter_summary_accepts_drive_run_dir_and_infers_crop(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "telemetry" / "run_123"
    asset_dir = _write_drive_adapter_export(run_dir, crop_name="tomato")
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=run_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"


def test_load_adapter_summary_infers_crop_from_local_export_crop_info(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export_with_crop_info(
        tmp_path / "outputs" / "colab_notebook_training",
        crop_name="tomato",
    )
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_dir=asset_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"


def test_predict_single_image_returns_notebook_payload(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (8, 8), color="green").save(image_path)

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 224)

    result = smoke.predict_single_image(image_path, "tomato", adapter_dir=asset_dir, device="cpu")

    assert result["image_name"] == "leaf.png"
    assert result["predicted_class"] == "blight"
    assert result["confidence"] == 0.91
    assert result["is_ood"] is False
    assert result["raw_payload"]["status"] == "success"


def test_predict_single_image_infers_crop_from_drive_export(monkeypatch, tmp_path: Path):
    asset_dir = _write_drive_adapter_export(tmp_path / "telemetry" / "run_123", crop_name="tomato")
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (8, 8), color="green").save(image_path)

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 224)

    result = smoke.predict_single_image(image_path, None, adapter_dir=asset_dir, device="cpu")

    assert result["predicted_class"] == "blight"
    assert result["adapter_dir"] == str(asset_dir)


def test_predict_image_folder_skips_non_images_and_records_errors(monkeypatch, tmp_path: Path):
    asset_dir = _write_adapter_export(tmp_path / "adapter_export")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), color="green").save(image_dir / "ok.png")
    (image_dir / "broken.png").write_text("not an image", encoding="utf-8")
    (image_dir / "notes.txt").write_text("skip me", encoding="utf-8")

    def zero_preprocess(_image, target_size=224):
        return torch.zeros(3, target_size, target_size)

    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))
    monkeypatch.setattr(smoke, "preprocess_image", zero_preprocess)
    monkeypatch.setattr(smoke, "_target_size", lambda _env: 224)

    rows = smoke.predict_image_folder(image_dir, "tomato", adapter_dir=asset_dir, device="cpu")

    assert len(rows) == 2
    assert {row["image_name"] for row in rows} == {"ok.png", "broken.png"}
    ok_row = next(row for row in rows if row["image_name"] == "ok.png")
    broken_row = next(row for row in rows if row["image_name"] == "broken.png")
    assert ok_row["predicted_class"] == "blight"
    assert ok_row["error"] == ""
    assert broken_row["status"] == "error"
    assert broken_row["error"]


def test_discover_adapter_candidates_reads_drive_exports(tmp_path: Path):
    drive_root = tmp_path / "drive_root"
    asset_dir = _write_drive_adapter_export(drive_root / "telemetry" / "run_456", crop_name="tomato")

    candidates = smoke.discover_adapter_candidates([drive_root], crop_name="tomato")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["adapter_dir"] == str(asset_dir)
    assert candidate["crop_name"] == "tomato"
    assert candidate["run_id"] == "run_456"
    assert "run=run_456" in candidate["display_name"]


def test_discover_adapter_candidates_scans_project_root_and_skips_cache_dirs(tmp_path: Path):
    project_root = tmp_path / "project"
    asset_dir = _write_adapter_export_with_crop_info(
        project_root / "outputs" / "colab_notebook_training",
        crop_name="tomato",
    )
    _write_adapter_export(project_root / ".venv" / "ignored_export")

    candidates = smoke.discover_adapter_candidates([project_root], crop_name=None)

    assert len(candidates) == 1
    assert candidates[0]["adapter_dir"] == str(asset_dir)
    assert candidates[0]["crop_name"] == "tomato"


def test_load_adapter_summary_accepts_crop_dir_as_adapter_root(monkeypatch, tmp_path: Path):
    crop_dir = tmp_path / "models" / "adapters" / "tomato"
    asset_dir = _write_adapter_export(crop_dir)
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary("tomato", adapter_root=crop_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)


def test_load_adapter_summary_accepts_crop_dir_as_adapter_root_without_crop_name(
    monkeypatch, tmp_path: Path
):
    crop_dir = tmp_path / "models" / "adapters" / "tomato"
    asset_dir = _write_adapter_export(crop_dir)
    monkeypatch.setattr(smoke, "_build_adapter", lambda crop_name, device: _FakeAdapter(crop_name, device))

    summary = smoke.load_adapter_summary(None, adapter_root=crop_dir, device="cpu")

    assert summary["resolved_adapter_dir"] == str(asset_dir)
    assert summary["crop_name"] == "tomato"
