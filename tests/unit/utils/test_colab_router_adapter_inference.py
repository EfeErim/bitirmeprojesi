import json
import sys
from pathlib import Path

from scripts import colab_router_adapter_inference as router_script
from src.shared.contracts import RouterAnalysisResult, RouterDetection


class _FakeImage:
    def __init__(self) -> None:
        self.convert_mode = ""

    def convert(self, mode: str):
        self.convert_mode = str(mode)
        return self


def test_run_inference_runs_router_only_pipeline(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    fake_image = _FakeImage()
    status_messages: list[str] = []

    class FakeRouter:
        def __init__(self, *, config=None, device="cuda"):
            calls["init"] = {
                "config": config,
                "device": device,
            }

        def load_models(self):
            calls["load_models"] = True

        def is_ready(self):
            return True

        def analyze_image_result(self, image):
            calls["analyze_image_result"] = image
            return RouterAnalysisResult(
                status="ok",
                message="",
                detections=[
                    RouterDetection(
                        crop="tomato",
                        part="fruit",
                        crop_confidence=0.95,
                        part_confidence=0.72,
                    )
                ],
            )

    def _open_image(path: Path):
        calls["opened_image"] = Path(path)
        return fake_image

    monkeypatch.setattr(router_script, "get_config", lambda environment=None: {"environment": environment})
    monkeypatch.setattr(router_script, "VLMPipeline", FakeRouter)
    monkeypatch.setattr(router_script.Image, "open", _open_image)

    result = router_script.run_inference(
        tmp_path / "leaf.png",
        config_env="colab",
        adapter_root=tmp_path / "models" / "adapters",
        device="cpu",
        status_printer=status_messages.append,
    )

    assert result == {
        "status": "ok",
        "crop": "tomato",
        "part": "fruit",
        "router_confidence": 0.95,
        "message": "",
        "router": {
            "status": "ok",
            "message": "",
            "detections_count": 1,
            "primary_detection": {
                "crop": "tomato",
                "part": "fruit",
                "crop_confidence": 0.95,
                "part_confidence": 0.72,
            },
        },
    }
    assert status_messages == [
        "[INFER] image=leaf.png device=cpu",
        "[ROUTER] Loading models on cpu...",
        "[ROUTER] Ready.",
        "[ROUTER] crop=tomato part=fruit confidence=0.950",
        "[RESULT] status=ok crop=tomato part=fruit router_confidence=0.950",
    ]
    assert calls["opened_image"] == tmp_path / "leaf.png"
    assert fake_image.convert_mode == "RGB"
    assert calls["init"] == {
        "config": {"environment": "colab"},
        "device": "cpu",
    }
    assert calls["load_models"] is True
    assert calls["analyze_image_result"] is fake_image


def test_run_inference_crop_hint_skips_router(monkeypatch, tmp_path: Path):
    fake_image = _FakeImage()
    status_messages: list[str] = []

    def _open_image(_path: Path):
        return fake_image

    monkeypatch.setattr(router_script.Image, "open", _open_image)
    monkeypatch.setattr(
        router_script,
        "VLMPipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("router should not be built")),
    )

    result = router_script.run_inference(
        tmp_path / "leaf.png",
        config_env="colab",
        crop_hint="tomato",
        part_hint="leaf",
        device="cpu",
        status_printer=status_messages.append,
    )

    assert result == {
        "status": "skipped",
        "crop": "tomato",
        "part": "leaf",
        "router_confidence": 1.0,
        "message": "Router skipped because crop_hint was provided.",
        "router": {
            "status": "skipped",
            "message": "Router skipped because crop_hint was provided.",
            "detections_count": 1,
            "primary_detection": {
                "crop": "tomato",
                "part": "leaf",
                "crop_confidence": 1.0,
                "part_confidence": 1.0,
            },
        },
    }
    assert status_messages == [
        "[INFER] image=leaf.png device=cpu",
        "[ROUTER] Skipped; using crop hint crop=tomato part=leaf",
        "[RESULT] status=skipped crop=tomato part=leaf router_confidence=1.000",
    ]


def test_main_prints_json_payload(monkeypatch, capsys, tmp_path: Path):
    calls: dict[str, object] = {}

    def _run_inference(
        image_path,
        *,
        config_env="colab",
        crop_hint=None,
        part_hint=None,
        adapter_root=None,
        device="cuda",
        status_printer=None,
    ):
        calls["run_inference"] = {
            "image_path": image_path,
            "config_env": config_env,
            "crop_hint": crop_hint,
            "part_hint": part_hint,
            "adapter_root": adapter_root,
            "device": device,
            "status_printer": status_printer,
        }
        return {
            "status": "success",
            "crop": "tomato",
            "router": {
                "status": "ok",
                "message": "",
                "detections_count": 1,
                "primary_detection": {"crop": "tomato", "part": "leaf", "crop_confidence": 0.95},
            },
        }

    monkeypatch.setattr(router_script, "run_inference", _run_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "colab_router_adapter_inference.py",
            str(tmp_path / "leaf.png"),
            "--config-env",
            "colab",
            "--crop",
            "tomato",
            "--part",
            "leaf",
            "--adapter-root",
            str(tmp_path / "models" / "adapters"),
            "--device",
            "cpu",
        ],
    )

    exit_code = router_script.main()

    assert exit_code == 0
    assert calls["run_inference"] == {
        "image_path": tmp_path / "leaf.png",
        "config_env": "colab",
        "crop_hint": "tomato",
        "part_hint": "leaf",
        "adapter_root": tmp_path / "models" / "adapters",
        "device": "cpu",
        "status_printer": None,
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "success",
        "crop": "tomato",
        "router": {
            "status": "ok",
            "message": "",
            "detections_count": 1,
            "primary_detection": {"crop": "tomato", "part": "leaf", "crop_confidence": 0.95},
        },
    }
