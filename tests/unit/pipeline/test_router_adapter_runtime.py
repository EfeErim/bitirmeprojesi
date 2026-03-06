from pathlib import Path

from PIL import Image

from src.pipeline.router_adapter_runtime import RouterAdapterRuntime


class FakeRouter:
    def __init__(self):
        self.loaded = False

    def load_models(self):
        self.loaded = True

    def analyze_image(self, image):
        return {
            "detections": [
                {
                    "crop": "tomato",
                    "part": "leaf",
                    "crop_confidence": 0.94,
                }
            ]
        }


class FakeAdapter:
    def __init__(self, crop_name, device="cpu"):
        self.crop_name = crop_name
        self.device = device
        self.loaded_path = None

    def load_adapter(self, adapter_dir):
        self.loaded_path = adapter_dir

    def predict_with_ood(self, image):
        return {
            "status": "success",
            "disease": {"class_index": 0, "name": "healthy", "confidence": 0.91},
            "ood_analysis": {
                "ensemble_score": 0.12,
                "class_threshold": 0.8,
                "is_ood": False,
                "calibration_version": 3,
            },
        }


def _write_adapter_meta(root: Path, crop: str) -> Path:
    adapter_dir = root / crop / "continual_sd_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_meta.json").write_text("{}", encoding="utf-8")
    return adapter_dir


def test_predict_routes_and_loads_adapter(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "success"
    assert result["crop"] == "tomato"
    assert result["part"] == "leaf"
    assert result["diagnosis"] == "healthy"
    assert {"ensemble_score", "class_threshold", "is_ood", "calibration_version"} <= set(result["ood_analysis"].keys())


def test_predict_returns_adapter_unavailable_when_assets_missing(monkeypatch, tmp_path):
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
    )
    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "adapter_unavailable"
    assert result["crop"] == "tomato"


def test_crop_hint_bypasses_router(monkeypatch, tmp_path):
    adapter_root = tmp_path / "models"
    _write_adapter_meta(adapter_root, "tomato")

    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(adapter_root), "target_size": 224},
        },
        device="cpu",
        adapter_root=adapter_root,
    )

    def _fail_router():
        raise AssertionError("router should not be built when crop_hint is provided")

    monkeypatch.setattr(runtime, "_build_router", _fail_router)
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"), crop_hint="tomato", part_hint="leaf")

    assert result["status"] == "success"
    assert result["router_confidence"] == 1.0


def test_unknown_crop_payload_when_router_returns_nothing(monkeypatch, tmp_path):
    runtime = RouterAdapterRuntime(
        config={
            "router": {"crop_mapping": {"tomato": {"parts": ["leaf"]}}, "vlm": {"enabled": True}},
            "training": {"continual": {"ood": {"threshold_factor": 2.0}}},
            "ood": {"threshold_factor": 2.0},
            "inference": {"adapter_root": str(tmp_path / "models"), "target_size": 224},
        },
        device="cpu",
        adapter_root=tmp_path / "models",
    )

    class EmptyRouter(FakeRouter):
        def analyze_image(self, image):
            return {"detections": []}

    monkeypatch.setattr(runtime, "_build_router", lambda: EmptyRouter())

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "unknown_crop"
    assert result["crop"] is None
