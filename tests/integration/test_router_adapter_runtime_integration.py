from PIL import Image

from src.core.config_manager import ConfigurationManager
from src.pipeline.router_adapter_runtime import RouterAdapterRuntime


class FakeRouter:
    def load_models(self):
        return None

    def analyze_image(self, image):
        return {
            "detections": [
                {"crop": "tomato", "part": "leaf", "crop_confidence": 0.97}
            ]
        }


class FakeAdapter:
    def __init__(self, crop_name, device="cpu"):
        self.crop_name = crop_name

    def load_adapter(self, adapter_dir):
        return None

    def predict_with_ood(self, image):
        return {
            "status": "success",
            "disease": {"class_index": 0, "name": "healthy", "confidence": 0.93},
            "ood_analysis": {
                "ensemble_score": 0.1,
                "class_threshold": 0.7,
                "is_ood": False,
                "calibration_version": 1,
            },
        }


def test_runtime_integration_with_config_manager_and_fake_router(monkeypatch, tmp_path):
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    adapter_root = tmp_path / "models" / "adapters"
    adapter_dir = adapter_root / "tomato" / "continual_sd_lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_meta.json").write_text("{}", encoding="utf-8")

    runtime = RouterAdapterRuntime(config=cfg, device="cpu", adapter_root=adapter_root)
    monkeypatch.setattr(runtime, "_build_router", lambda: FakeRouter())
    monkeypatch.setattr(runtime, "_build_adapter", lambda crop_name: FakeAdapter(crop_name, device="cpu"))

    result = runtime.predict(Image.new("RGB", (32, 32), color="green"))

    assert result["status"] == "success"
    assert result["crop"] == "tomato"
    assert result["diagnosis"] == "healthy"
    assert result["router"]["status"] == "ok"
    assert result["router"]["primary_detection"]["crop"] == "tomato"
