import sys
import types
from pathlib import Path

import pytest

from scripts import colab_simple_adapter_smoke_ui as ui


def test_ensure_colab_widget_manager_returns_false_outside_colab(monkeypatch):
    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    assert ui._ensure_colab_widget_manager() is False


def test_ensure_colab_widget_manager_enables_custom_manager(monkeypatch):
    calls: list[str] = []

    output_module = types.ModuleType("google.colab.output")
    output_module.enable_custom_widget_manager = lambda: calls.append("enabled")

    colab_module = types.ModuleType("google.colab")
    colab_module.output = output_module

    google_module = types.ModuleType("google")
    google_module.colab = colab_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    monkeypatch.setitem(sys.modules, "google.colab.output", output_module)

    assert ui._ensure_colab_widget_manager() is True
    assert calls == ["enabled"]


def test_launch_simple_adapter_smoke_ui_raises_when_widget_manager_bootstrap_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "widgets", object())
    monkeypatch.setattr(ui, "_ensure_colab_widget_manager", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="Colab custom widgets could not be initialized"):
        ui.launch_simple_adapter_smoke_ui(tmp_path)


def test_build_result_html_sets_explicit_contrast_colors():
    html = ui._build_result_html(
        {
            "resolved_adapter_dir": "/tmp/adapter",
            "crop_name": "tomato",
        },
        {
            "predicted_class": "septoria",
            "confidence": 0.9712,
            "is_ood": False,
            "primary_score": 0.8673,
            "decision_threshold": 1.8515,
            "view_consistency": {"stable": True, "warning_codes": []},
            "uncertainty_diagnostics": {"warning_codes": []},
        },
        image_path=ui.Path("/tmp/leaf.png"),
    )

    assert "background:#ffffff;color:#111827" in html
    assert "summary style=\"cursor:pointer;color:#111827;font-weight:600;\"" in html
    assert "Stable across derived views" in html


def test_persist_upload_value_writes_uploaded_bytes(tmp_path: Path):
    persisted = ui._persist_upload_value(
        (
            {
                "name": "leaf.png",
                "content": memoryview(b"abc123"),
            },
        ),
        tmp_path,
    )

    assert persisted == tmp_path / "leaf.png"
    assert persisted.read_bytes() == b"abc123"


def test_persist_upload_value_returns_none_for_empty_upload(tmp_path: Path):
    assert ui._persist_upload_value((), tmp_path) is None
