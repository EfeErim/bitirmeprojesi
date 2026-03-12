import json
import sys
from pathlib import Path

from scripts import colab_router_adapter_inference as router_script


class _FakeImage:
    def __init__(self) -> None:
        self.convert_mode = ""

    def convert(self, mode: str):
        self.convert_mode = str(mode)
        return self


def test_run_inference_dispatches_workflow(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    fake_image = _FakeImage()
    status_messages: list[str] = []

    class FakeWorkflow:
        def __init__(self, *, environment=None, adapter_root=None, device="cuda", status_callback=None):
            calls["init"] = {
                "environment": environment,
                "adapter_root": adapter_root,
                "device": device,
                "status_callback": status_callback,
            }

        def predict(self, image, *, crop_hint=None, part_hint=None, return_ood=True):
            calls["predict"] = {
                "image": image,
                "crop_hint": crop_hint,
                "part_hint": part_hint,
                "return_ood": return_ood,
            }
            return {
                "status": "success",
                "crop": "tomato",
                "diagnosis": "healthy",
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

    def _open_image(path: Path):
        calls["opened_image"] = Path(path)
        return fake_image

    monkeypatch.setattr(router_script, "InferenceWorkflow", FakeWorkflow)
    monkeypatch.setattr(router_script.Image, "open", _open_image)

    result = router_script.run_inference(
        tmp_path / "leaf.png",
        config_env="colab",
        crop_hint="tomato",
        part_hint="leaf",
        adapter_root=tmp_path / "models" / "adapters",
        device="cpu",
        status_printer=status_messages.append,
    )

    assert result["status"] == "success"
    assert result["router"]["primary_detection"]["crop"] == "tomato"
    assert status_messages == ["[INFER] image=leaf.png device=cpu"]
    assert calls["opened_image"] == tmp_path / "leaf.png"
    assert fake_image.convert_mode == "RGB"
    assert calls["init"] == {
        "environment": "colab",
        "adapter_root": tmp_path / "models" / "adapters",
        "device": "cpu",
        "status_callback": status_messages.append,
    }
    assert calls["predict"] == {
        "image": fake_image,
        "crop_hint": "tomato",
        "part_hint": "leaf",
        "return_ood": True,
    }


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
