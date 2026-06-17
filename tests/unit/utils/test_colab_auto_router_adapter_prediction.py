from pathlib import Path

import scripts.colab_auto_router_adapter_prediction as auto_prediction
from scripts.colab_auto_router_adapter_prediction import (
    clear_auto_prediction_workflow_cache,
    run_auto_router_adapter_prediction,
)


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
            "crop": crop_hint,
            "part": part_hint,
            "router_confidence": 1.0,
            "diagnosis": "healthy",
            "confidence": 0.91,
            "router": {"status": "trusted_hint_skipped"},
            "ood_analysis": {
                "score_method": "ensemble",
                "primary_score": 0.1,
                "decision_threshold": 0.8,
                "is_ood": False,
                "calibration_version": 1,
            },
        }


class CacheableFakeWorkflow:
    instances = []

    def __init__(self, *, environment=None, device="cuda", adapter_root=None, status_callback=None):
        self.environment = environment
        self.device = device
        self.adapter_root = adapter_root
        self.runtime = type("Runtime", (), {"status_callback": status_callback, "router": None})()
        self.instances.append(self)


def _router_result(status="ok", crop="tomato", part="leaf"):
    return {
        "status": status,
        "crop": crop,
        "part": part,
        "router_confidence": 0.94,
        "router": {
            "status": status,
            "message": "",
            "detections_count": 1,
            "primary_detection": {
                "crop": crop,
                "part": part,
                "crop_confidence": 0.94,
                "part_confidence": 0.88,
            },
        },
    }


def test_auto_prediction_uses_trusted_router_handoff_for_adapter():
    FakeWorkflow.calls.clear()

    result = run_auto_router_adapter_prediction(
        Path("leaf.jpg"),
        router_result=_router_result(),
        config_env="colab",
        device="cpu",
        adapter_root=Path("models/adapters"),
        return_ood=True,
        workflow_factory=FakeWorkflow,
    )

    assert result["status"] == "success"
    assert result["diagnosis"] == "healthy"
    assert result["router_handoff"]["adapter_ran"] is True
    assert result["router_source"]["primary_detection"]["crop"] == "tomato"
    assert FakeWorkflow.calls == [
        {
            "image": Path("leaf.jpg"),
            "crop_hint": "tomato",
            "part_hint": "leaf",
            "return_ood": True,
            "trust_crop_hint": True,
            "environment": "colab",
            "device": "cpu",
            "adapter_root": Path("models/adapters"),
        }
    ]


def test_auto_prediction_skips_adapter_when_router_is_uncertain():
    FakeWorkflow.calls.clear()

    result = run_auto_router_adapter_prediction(
        "leaf.jpg",
        router_result=_router_result(status="router_uncertain"),
        workflow_factory=FakeWorkflow,
    )

    assert result["status"] == "router_uncertain"
    assert result["diagnosis"] is None
    assert result["router_handoff"]["adapter_ran"] is False
    assert FakeWorkflow.calls == []


def test_auto_prediction_skips_adapter_when_crop_is_unknown():
    FakeWorkflow.calls.clear()

    result = run_auto_router_adapter_prediction(
        "leaf.jpg",
        router_result=_router_result(status="ok", crop="unknown", part="unknown"),
        workflow_factory=FakeWorkflow,
    )

    assert result["status"] == "unknown_crop"
    assert result["crop"] == "unknown"
    assert result["router_handoff"]["adapter_ran"] is False
    assert FakeWorkflow.calls == []


def test_auto_prediction_skips_adapter_when_part_is_unknown():
    FakeWorkflow.calls.clear()

    result = run_auto_router_adapter_prediction(
        "leaf.jpg",
        router_result=_router_result(status="ok", crop="tomato", part="unknown"),
        workflow_factory=FakeWorkflow,
    )

    assert result["status"] == "router_uncertain"
    assert result["crop"] == "tomato"
    assert result["part"] == "unknown"
    assert result["message"] == "Router could not resolve a supported plant part."
    assert result["router_handoff"]["adapter_ran"] is False
    assert FakeWorkflow.calls == []


def test_auto_prediction_skips_adapter_for_unsupported_final_demo_crop():
    FakeWorkflow.calls.clear()

    result = run_auto_router_adapter_prediction(
        "leaf.jpg",
        router_result=_router_result(status="ok", crop="potato", part="leaf"),
        workflow_factory=FakeWorkflow,
    )

    assert result["status"] == "unknown_crop"
    assert result["crop"] == "potato"
    assert "outside the final demo supported crop set" in result["message"]
    assert result["router_handoff"]["adapter_ran"] is False
    assert FakeWorkflow.calls == []


def test_auto_prediction_skips_adapter_for_unsupported_final_demo_part():
    FakeWorkflow.calls.clear()

    result = run_auto_router_adapter_prediction(
        "leaf.jpg",
        router_result=_router_result(status="ok", crop="tomato", part="bud"),
        workflow_factory=FakeWorkflow,
    )

    assert result["status"] == "router_uncertain"
    assert result["part"] == "bud"
    assert "outside the final demo supported part set" in result["message"]
    assert result["router_handoff"]["adapter_ran"] is False
    assert FakeWorkflow.calls == []


def test_auto_prediction_reuses_default_workflow_and_ready_router(monkeypatch):
    CacheableFakeWorkflow.instances.clear()
    clear_auto_prediction_workflow_cache()
    ready_router = object()
    monkeypatch.setattr(auto_prediction, "InferenceWorkflow", CacheableFakeWorkflow)
    monkeypatch.setattr(auto_prediction, "ensure_router_ready", lambda **_kwargs: ready_router)

    first = auto_prediction._resolve_workflow(
        config_env="colab",
        device="cpu",
        adapter_root=Path("models/adapters"),
        status_printer=None,
        workflow_factory=CacheableFakeWorkflow,
    )
    second = auto_prediction._resolve_workflow(
        config_env="colab",
        device="cpu",
        adapter_root=Path("models/adapters"),
        status_printer=print,
        workflow_factory=CacheableFakeWorkflow,
    )

    assert first is second
    assert len(CacheableFakeWorkflow.instances) == 1
    assert second.runtime.router is ready_router
    assert second.runtime.status_callback is print
