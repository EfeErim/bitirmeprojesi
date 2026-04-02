#!/usr/bin/env python3
"""Minimal Notebook 4 UI for direct adapter smoke testing."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Optional

from IPython.display import HTML, clear_output, display
from PIL import Image

from scripts.colab_adapter_smoke_test import (
    discover_adapter_candidates,
    load_adapter_summary,
    predict_single_image,
)

try:
    import ipywidgets as widgets
except Exception:  # pragma: no cover - notebook runtime fallback
    widgets = None


def _ensure_colab_widget_manager() -> bool:
    """Enable the Colab widget bridge before rendering FileUpload widgets."""
    try:
        from google.colab import output as colab_output
    except Exception:
        return False

    enable_manager = getattr(colab_output, "enable_custom_widget_manager", None)
    if enable_manager is None:
        return False

    enable_manager()
    return True


def _extract_upload_record(upload_value: Any) -> tuple[Optional[str], Optional[bytes]]:
    if isinstance(upload_value, dict):
        records = list(upload_value.values())
    elif isinstance(upload_value, (list, tuple)):
        records = list(upload_value)
    else:
        records = []
    if not records:
        return None, None

    record = records[0]
    if isinstance(record, dict):
        name = record.get("name", "uploaded_image")
        content = record.get("content", b"")
    else:
        name = getattr(record, "name", "uploaded_image")
        content = getattr(record, "content", b"")

    if isinstance(content, memoryview):
        content = content.tobytes()
    elif hasattr(content, "tobytes"):
        content = content.tobytes()
    return str(name), bytes(content)


def _build_result_html(summary: dict[str, Any], result: dict[str, Any], image_path: Path) -> str:
    confidence = float(result.get("confidence", 0.0)) * 100.0
    ood_label = "OOD" if bool(result.get("is_ood")) else "In-distribution"
    threshold = result.get("decision_threshold")
    score = result.get("primary_score")
    score_text = "-" if score is None else f"{float(score):.4f}"
    threshold_text = "-" if threshold is None else f"{float(threshold):.4f}"
    view_consistency = dict(result.get("view_consistency", {}))
    uncertainty = dict(result.get("uncertainty_diagnostics", {}))
    robustness_warning_codes = [str(code) for code in list(view_consistency.get("warning_codes", []))]
    uncertainty_warning_codes = [str(code) for code in list(uncertainty.get("warning_codes", []))]
    if bool(view_consistency.get("stable")):
        robustness_status = "Stable across derived views"
        robustness_accent = "#0f766e"
    else:
        headline_warnings = robustness_warning_codes[:2] or uncertainty_warning_codes[:2]
        if headline_warnings:
            robustness_status = "Review recommended: " + ", ".join(headline_warnings)
        else:
            robustness_status = "Review recommended"
        robustness_accent = "#b45309"
    robustness_warning_text = ", ".join(robustness_warning_codes) if robustness_warning_codes else "-"
    uncertainty_warning_text = ", ".join(uncertainty_warning_codes) if uncertainty_warning_codes else "-"
    return f"""
    <div style="border:1px solid #d0d7de;border-radius:10px;padding:16px;margin-top:12px;background:#ffffff;color:#111827;box-shadow:0 1px 3px rgba(15,23,42,0.08);">
      <div style="font-size:18px;font-weight:700;margin-bottom:8px;color:#111827;">Tahmin Sonucu</div>
      <div style="color:#374151;"><b style="color:#111827;">Adapter:</b> {escape(str(summary['resolved_adapter_dir']))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Crop:</b> {escape(str(summary['crop_name']))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Sinif:</b> {escape(str(result.get('predicted_class') or '-'))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Confidence:</b> {confidence:.2f}%</div>
      <div style="color:#374151;"><b style="color:#111827;">OOD Karari:</b> {ood_label}</div>
      <div style="color:#374151;"><b style="color:#111827;">OOD Score:</b> {score_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">Karar Esigi:</b> {threshold_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">Robustluk:</b> <span style="color:{robustness_accent};font-weight:600;">{escape(robustness_status)}</span></div>
      <div style="color:#374151;word-break:break-word;"><b style="color:#111827;">Goruntu:</b> {escape(str(image_path))}</div>
      <details style="margin-top:10px;color:#374151;">
        <summary style="cursor:pointer;color:#111827;font-weight:600;">Kompakt Teshis</summary>
        <div style="margin-top:8px;"><b style="color:#111827;">View warnings:</b> {escape(robustness_warning_text)}</div>
        <div><b style="color:#111827;">Uncertainty warnings:</b> {escape(uncertainty_warning_text)}</div>
      </details>
    </div>
    """


def launch_simple_adapter_smoke_ui(
    root: str | Path,
    *,
    search_roots: Optional[list[str | Path]] = None,
    config_env: str = "colab",
    device: str = "cuda",
    upload_dir_name: str = "notebook4_uploads",
) -> None:
    """Render the minimal direct-adapter smoke-test UI used by Notebook 4."""
    if widgets is None:
        raise RuntimeError(
            "This notebook UI requires ipywidgets. Re-run the bootstrap cell after dependency installation."
        )
    try:
        _ensure_colab_widget_manager()
    except Exception as exc:
        raise RuntimeError(
            "Colab custom widgets could not be initialized. Re-run the bootstrap cell before using Notebook 4 uploads."
        ) from exc

    root_path = Path(root)
    resolved_search_roots = [
        Path(candidate)
        for candidate in (
            search_roots
            or [
                root_path,
                root_path / "outputs",
                root_path / "models",
                root_path / "models" / "adapters",
            ]
        )
    ]
    upload_dir = root_path / ".runtime_tmp" / upload_dir_name
    upload_dir.mkdir(parents=True, exist_ok=True)

    adapter_candidates = discover_adapter_candidates(resolved_search_roots, crop_name=None)
    dropdown_options = [
        (candidate["display_name"], index)
        for index, candidate in enumerate(adapter_candidates)
    ] or [("Adapter bulunamadi, asagidan yol girin", -1)]

    title = widgets.HTML("<h3 style=\"margin:0 0 8px 0;\">Basit Adapter Testi</h3>")
    help_text = widgets.HTML(
        "<p style=\"margin:0 0 12px 0;\">Adapter secin veya yol girin, bir resim yukleyin, sonra <b>Tahmin Et</b> butonuna basin.</p>"
    )
    adapter_dropdown = widgets.Dropdown(
        options=dropdown_options,
        value=dropdown_options[0][1],
        description="Adapter:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    adapter_path_text = widgets.Text(
        value="",
        placeholder="Isterseniz ADAPTER_DIR veya adapter_meta.json yolu girin",
        description="Yol:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    image_upload = widgets.FileUpload(
        accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp",
        multiple=False,
        description="Resim Yukle",
    )
    image_path_text = widgets.Text(
        value="",
        placeholder="Upload yerine mevcut bir dosya yolu da verebilirsiniz",
        description="Resim:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    refresh_button = widgets.Button(description="Adapterleri Yenile", button_style="info")
    run_button = widgets.Button(description="Tahmin Et", button_style="success")
    status_output = widgets.Output()
    result_output = widgets.Output()

    def selected_candidate() -> dict[str, Any]:
        manual_path = adapter_path_text.value.strip()
        if manual_path:
            return {"adapter_dir": manual_path, "crop_name": None, "display_name": manual_path}

        selected_index = int(adapter_dropdown.value)
        if selected_index < 0 or selected_index >= len(adapter_candidates):
            raise FileNotFoundError(
                "Adapter bulunamadi. Ya yol girin ya da search_roots altinda adapter bulundurun."
            )
        return adapter_candidates[selected_index]

    def resolve_image_path() -> Path:
        upload_name, upload_bytes = _extract_upload_record(image_upload.value)
        if upload_name and upload_bytes:
            target_path = upload_dir / Path(upload_name).name
            target_path.write_bytes(upload_bytes)
            return target_path

        raw_path = image_path_text.value.strip()
        if raw_path:
            return Path(raw_path).expanduser()
        raise ValueError("Bir resim yukleyin veya mevcut bir dosya yolu girin.")

    def render_result(summary: dict[str, Any], result: dict[str, Any], image_path: Path) -> None:
        display(HTML(_build_result_html(summary, result, image_path)))

    def refresh(_button: Any = None) -> None:
        nonlocal adapter_candidates
        with status_output:
            clear_output(wait=True)
            print("Adapter listesi yenileniyor...")
        adapter_candidates = discover_adapter_candidates(resolved_search_roots, crop_name=None)
        options = [
            (candidate["display_name"], index)
            for index, candidate in enumerate(adapter_candidates)
        ] or [("Adapter bulunamadi, asagidan yol girin", -1)]
        adapter_dropdown.options = options
        adapter_dropdown.value = options[0][1]
        with status_output:
            clear_output(wait=True)
            print(f"Bulunan adapter sayisi: {len(adapter_candidates)}")
            for candidate_root in resolved_search_roots:
                print(f"- taranan kok: {candidate_root}")

    def run_prediction(_button: Any = None) -> None:
        with result_output:
            clear_output(wait=True)
            candidate = selected_candidate()
            summary = load_adapter_summary(
                candidate.get("crop_name"),
                adapter_dir=candidate.get("adapter_dir"),
                config_env=config_env,
                device=device,
            )
            image_path = resolve_image_path()
            if not image_path.exists():
                raise FileNotFoundError(f"Resim bulunamadi: {image_path}")
            with Image.open(image_path) as preview:
                display(preview.copy())
            result = predict_single_image(
                image_path,
                summary["crop_name"],
                adapter_dir=summary["resolved_adapter_dir"],
                config_env=config_env,
                device=device,
                enable_robust_smoke=True,
            )
            render_result(summary, result, image_path)
            display(
                HTML(
                    "<details><summary>Ham JSON</summary><pre>"
                    + json.dumps(result, indent=2)
                    + "</pre></details>"
                )
            )

    refresh_button.on_click(refresh)
    run_button.on_click(run_prediction)
    display(
        widgets.VBox(
            [
                title,
                help_text,
                adapter_dropdown,
                adapter_path_text,
                image_upload,
                image_path_text,
                widgets.HBox([refresh_button, run_button]),
                status_output,
                result_output,
            ]
        )
    )
    refresh()

__all__ = ["launch_simple_adapter_smoke_ui", "_ensure_colab_widget_manager", "_build_result_html"]
