"""Integration coverage for the v6 image -> router -> adapter -> OOD pipeline flow."""

import torch

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline


class _RouterStub:
    def __init__(self):
        self.calls = 0

    def route(self, image_tensor):
        self.calls += 1
        return "tomato", 0.93


class _AdapterStub:
    def __init__(self):
        self.calls = 0

    def predict_with_ood(self, image_tensor):
        self.calls += 1
        return {
            "status": "success",
            "disease": {
                "class_index": 1,
                "name": "late_blight",
                "confidence": 0.88,
            },
            "ood_analysis": {
                "ensemble_score": 0.91,
                "class_threshold": 0.80,
                "is_ood": True,
                "calibration_version": 4,
            },
        }


def test_full_pipeline_image_to_result_with_ood_payload():
    config = {
        "crops": ["tomato"],
        "router": {"target_size": 224},
        "cache_enabled": False,
    }
    pipeline = IndependentMultiCropPipeline(config=config, device="cpu")
    router = _RouterStub()
    adapter = _AdapterStub()
    pipeline.router = router
    pipeline.adapters["tomato"] = adapter

    image = torch.randn(3, 224, 224)
    result = pipeline.process_image(image)

    assert router.calls == 1
    assert adapter.calls == 1
    assert result["status"] == "success"
    assert result["crop"] == "tomato"
    assert result["diagnosis"]["name"] == "late_blight"
    assert result["ood_analysis"]["is_ood"] is True
    assert result["ood_analysis"]["ensemble_score"] == 0.91
    assert result["recommendations"]["expert_consultation"] is True
