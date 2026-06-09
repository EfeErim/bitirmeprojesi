from pathlib import Path

from PIL import Image

from scripts import colab_roi_ablation as roi_ablation


class FakeWorkflow:
    calls = []

    def __init__(self, *, environment=None, device="cuda", adapter_root=None, status_callback=None):
        self.environment = environment
        self.device = device
        self.adapter_root = adapter_root
        self.status_callback = status_callback

    def predict(self, image, *, crop_hint=None, part_hint=None, return_ood=True, trust_crop_hint=False):
        self.calls.append(
            {
                "image": image,
                "crop_hint": crop_hint,
                "part_hint": part_hint,
                "return_ood": return_ood,
                "trust_crop_hint": trust_crop_hint,
                "environment": self.environment,
                "device": self.device,
                "adapter_root": self.adapter_root,
            }
        )
        return {
            "status": "success",
            "crop": crop_hint or "tomato",
            "part": part_hint or "leaf",
            "diagnosis": "healthy",
            "confidence": 0.88,
            "router_confidence": 0.93,
            "router": {
                "status": "trusted_hint_skipped",
                "primary_detection": {
                    "crop": crop_hint or "tomato",
                    "part": part_hint or "leaf",
                    "crop_confidence": 1.0,
                },
            },
            "ood_analysis": {
                "is_ood": False,
                "primary_score": 0.12,
            },
        }


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color=(20, 120, 40)).save(path)
    return path


def _router_result(*, status="ok", crop="tomato", part="leaf", bbox=None):
    return {
        "status": status,
        "crop": crop,
        "part": part,
        "router_confidence": 0.93,
        "router": {
            "status": status,
            "primary_detection": {
                "crop": crop,
                "part": part,
                "crop_confidence": 0.93,
                "part_confidence": 0.85,
                "bbox": bbox,
            },
        },
    }


def test_prepare_primary_roi_sanitizes_valid_and_invalid_bbox():
    image = Image.new("RGB", (100, 80), color="green")

    roi, bbox, area_ratio = roi_ablation.prepare_primary_roi(image, [-5, 10, 50, 90])
    assert roi is not None
    assert bbox == [0.0, 10.0, 50.0, 80.0]
    assert area_ratio > 0.0

    roi, bbox, area_ratio = roi_ablation.prepare_primary_roi(image, [10, 10, 10, 30])
    assert roi is None
    assert bbox is None
    assert area_ratio == 0.0


def test_tokenized_git_remote_url_uses_github_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "secret-token")

    push_url = roi_ablation._tokenized_git_remote_url("https://github.com/EfeErim/bitirmeprojesi.git")

    assert push_url == "https://x-access-token:secret-token@github.com/EfeErim/bitirmeprojesi.git"


def test_primary_roi_ablation_skips_without_fallback_when_bbox_missing(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(bbox=None),
    )

    assert rows[0]["status"] == "roi_missing"
    assert rows[0]["input_view"] == "router_primary_roi"
    assert FakeWorkflow.calls == []


def test_hybrid_ablation_falls_back_to_full_image_when_bbox_missing(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="hybrid_roi_fallback",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(bbox=None),
    )

    assert rows[0]["status"] == "fallback_full_image"
    assert rows[0]["input_view"] == "fallback_full_image"
    assert FakeWorkflow.calls[0]["image"] == image_path
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "leaf"
    assert FakeWorkflow.calls[0]["trust_crop_hint"] is True


def test_full_image_baseline_uses_fixed_adapter_without_router(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "fruit.jpg")

    def fail_router(*args, **kwargs):
        raise AssertionError("full-image fixed-adapter baseline should not call the router")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="full_image_baseline",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=fail_router,
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["crop"] == "tomato"
    assert rows[0]["part"] == "fruit"
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "fruit"
    assert FakeWorkflow.calls[0]["trust_crop_hint"] is True


def test_fixed_adapter_roi_ignores_wrong_router_crop_for_adapter_selection(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "fruit.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        adapter_crop="tomato",
        adapter_part="fruit",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(crop="eggplant", part="unknown", bbox=[10, 10, 50, 40]),
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["crop"] == "tomato"
    assert rows[0]["part"] == "fruit"
    assert rows[0]["router_status"] == "ok"
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "fruit"


def test_part_unknown_skips_adapter_load(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="hybrid_roi_fallback",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(part="unknown", bbox=[5, 5, 40, 40]),
    )

    assert rows[0]["status"] == "adapter_skipped"
    assert rows[0]["part"] == "unknown"
    assert FakeWorkflow.calls == []


def test_roi_prediction_uses_cropped_pil_image_and_same_adapter_metadata(tmp_path: Path):
    FakeWorkflow.calls.clear()
    image_path = _write_image(tmp_path / "leaf.jpg")

    rows = roi_ablation.run_ablation_image(
        image_path,
        ablation_name="primary_roi_inference",
        workflow_factory=FakeWorkflow,
        router_runner=lambda *args, **kwargs: _router_result(bbox=[10, 10, 50, 40]),
        adapter_root=Path("models/adapters"),
        device="cpu",
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["input_view"] == "router_primary_roi"
    assert rows[0]["bbox"] == [10.0, 10.0, 50.0, 40.0]
    assert FakeWorkflow.calls[0]["image"].size[0] < 80
    assert FakeWorkflow.calls[0]["crop_hint"] == "tomato"
    assert FakeWorkflow.calls[0]["part_hint"] == "leaf"
    assert FakeWorkflow.calls[0]["adapter_root"] == Path("models/adapters")


def test_build_mixed_full_roi_dataset_writes_full_and_roi_training_view(tmp_path: Path):
    source_root = tmp_path / "source"
    dataset = source_root / "tomato__fruit"
    _write_image(dataset / "continual" / "healthy" / "train_a.jpg")
    _write_image(dataset / "val" / "healthy" / "val_a.jpg")
    _write_image(dataset / "test" / "healthy" / "test_a.jpg")
    _write_image(dataset / "ood" / "field" / "ood_a.jpg")

    def strict_router_runner(
        image_path,
        *,
        config_env="colab",
        device="cuda",
        include_adapter_target=True,
        status_printer=None,
    ):
        assert image_path.name == "train_a.jpg"
        assert config_env == "colab"
        assert device == "cuda"
        assert include_adapter_target is False
        assert status_printer is None
        return _router_result(crop="tomato", part="fruit", bbox=[10, 10, 50, 40])

    manifest = roi_ablation.build_mixed_full_roi_dataset(
        source_root,
        dataset_key="tomato__fruit",
        output_root=tmp_path / "mixed",
        router_runner=strict_router_runner,
        status_printer=None,
    )

    target = tmp_path / "mixed" / "tomato__fruit"
    assert len(list((target / "continual" / "healthy").glob("full__*.jpg"))) == 1
    assert len(list((target / "continual" / "healthy").glob("roi__*.jpg"))) == 1
    assert len(list((target / "val" / "healthy").glob("full__*.jpg"))) == 1
    assert not list((target / "val" / "healthy").glob("roi__*.jpg"))
    assert (target / "ood" / "field" / "ood_a.jpg").exists()
    assert manifest["splits"]["continual"]["full_images"] == 1
    assert manifest["splits"]["continual"]["roi_images"] == 1
