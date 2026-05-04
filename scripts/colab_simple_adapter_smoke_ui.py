#!/usr/bin/env python3
"""Minimal Notebook 4 UI for direct adapter smoke testing."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Optional

from IPython.display import HTML, clear_output, display
from PIL import Image

# Defer heavy imports until actually needed (when UI is launched and user interacts)
try:
    import ipywidgets as widgets
except Exception:  # pragma: no cover - notebook runtime fallback
    widgets = None

# Lazy import placeholders - will be populated on-demand
_build_prediction_visualization_images = None
_discover_adapter_candidates = None
_load_adapter_summary = None
_predict_single_image = None

def _ensure_adapter_smoke_imports():
    """Lazy import adapter smoke functions when needed."""
    global _build_prediction_visualization_images, _discover_adapter_candidates, _load_adapter_summary, _predict_single_image
    if _build_prediction_visualization_images is None:
        from src.pipeline.adapter_smoke import (
            build_prediction_visualization_images as _bpvi,
            discover_adapter_candidates as _dac,
            load_adapter_summary as _las,
            predict_single_image as _psi,
        )
        _build_prediction_visualization_images = _bpvi
        _discover_adapter_candidates = _dac
        _load_adapter_summary = _las
        _predict_single_image = _psi


def _running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_colab_widget_manager() -> bool:
    """Backward-compatible helper kept for existing tests and callers."""
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


def _persist_upload_value(upload_value: Any, upload_dir: Path) -> Optional[Path]:
    upload_name, upload_bytes = _extract_upload_record(upload_value)
    if not upload_name or not upload_bytes:
        return None
    target_path = upload_dir / Path(upload_name).name
    target_path.write_bytes(upload_bytes)
    return target_path


def _format_optional_float(value: Any, *, scale: float = 1.0, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * scale:.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def _format_mapping_items(mapping: Any, *, value_formatter: Optional[Any] = None) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return "-"
    parts: list[str] = []
    for key, value in mapping.items():
        rendered_value = value_formatter(value) if value_formatter else str(value)
        parts.append(f"{key}: {rendered_value}")
    return "; ".join(parts)


def _bool_vote_label(value: Any) -> str:
    if value is True:
        return "OOD"
    if value is False:
        return "In-distribution"
    return "-"


def _warning_detail_items(
    view_consistency: dict[str, Any],
    uncertainty: dict[str, Any],
) -> list[str]:
    view_warning_codes = [str(code) for code in list(view_consistency.get("warning_codes", []))]
    uncertainty_warning_codes = [str(code) for code in list(uncertainty.get("warning_codes", []))]
    items: list[str] = [
        "Robust smoke, ayni gorseli farkli on-isleme gorunumleriyle tekrar tahmin eder; sonuc degisiyorsa tek tahmine guvenmeden goruntuyu ve adapteri manuel incelemek gerekir."
    ]

    if bool(view_consistency.get("stable")):
        items.append(
            "Stabil: uretilen gorunumler ayni sinifi ve ayni OOD kararini verdi; belirgin confidence farki veya gorunum hatasi yok."
        )
    elif not view_warning_codes and not uncertainty_warning_codes:
        items.append(
            "Inceleme onerisi var ama ayrintili uyari kodu gelmedi; ham JSON altindan views alanini kontrol edin."
        )

    view_warning_explanations = {
        "view_class_disagreement": (
            "view_class_disagreement: farkli gorunumler farkli siniflar tahmin etti. "
            "Gorunum bazli siniflar: "
            + _format_mapping_items(view_consistency.get("predicted_classes"))
        ),
        "view_ood_disagreement": (
            "view_ood_disagreement: gorunumler OOD kararinda ayrildi. "
            "Gorunum bazli kararlar: "
            + _format_mapping_items(view_consistency.get("ood_votes"), value_formatter=_bool_vote_label)
        ),
        "view_confidence_spread_high": (
            "view_confidence_spread_high: gorunumler arasindaki confidence farki yuksek. "
            "Min "
            + _format_optional_float(view_consistency.get("confidence_min"), scale=100.0, suffix="%")
            + ", max "
            + _format_optional_float(view_consistency.get("confidence_max"), scale=100.0, suffix="%")
            + ", fark "
            + _format_optional_float(view_consistency.get("confidence_spread"), scale=100.0, suffix=" puan")
            + "."
        ),
        "view_error_present": (
            "view_error_present: en az bir gorunum tahmini hata verdi. Hata veren gorunumler: "
            + (", ".join(str(name) for name in view_consistency.get("failed_views", [])) or "-")
        ),
    }
    uncertainty_warning_explanations = {
        "prediction_error": "prediction_error: ana gorunum tahmini hata verdi; hata satirini ve ham JSON'u kontrol edin.",
        "confidence_not_calibrated": (
            "confidence_not_calibrated: confidence top-1 softmax degeridir; kalibre edilmis olasilik gibi yorumlanmamalidir."
        ),
        "ood_flagged": "ood_flagged: ana gorunum adapter esigine gore OOD olarak isaretlendi.",
        "sure_confidence_reject": "sure_confidence_reject: daha siki confidence kontrolu tahmini reddetti.",
        "conformal_set_wide": (
            "conformal_set_wide: birden fazla makul sinif var; conformal set: "
            + (", ".join(str(item) for item in uncertainty.get("conformal_set", [])) or "-")
        ),
        "view_instability": "view_instability: robust gorunumler stabil degil; yukaridaki view uyarilari karar nedenini gosterir.",
    }

    for code in view_warning_codes:
        items.append(view_warning_explanations.get(code, f"{code}: tanimli olmayan view uyarisi; ham JSON'u kontrol edin."))
    for code in uncertainty_warning_codes:
        items.append(
            uncertainty_warning_explanations.get(
                code,
                f"{code}: tanimli olmayan belirsizlik uyarisi; ham JSON'u kontrol edin.",
            )
        )

    return items


def _warning_details_html(view_consistency: dict[str, Any], uncertainty: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<li style=\"margin:4px 0;\">{escape(item)}</li>"
        for item in _warning_detail_items(view_consistency, uncertainty)
    )
    return f"<ul style=\"margin:6px 0 0 18px;padding:0;color:#374151;\">{rows}</ul>"


def _is_huggingface_gated_access_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    return (
        "gated repo" in lowered
        or "401 unauthorized" in lowered
        or "access to model" in lowered and "restricted" in lowered
        or "you must have access to it and be authenticated" in lowered
    )


def _hf_access_error_html(exc: BaseException) -> str:
    return f"""
    <div style="border:1px solid #fecaca;border-radius:10px;padding:14px;margin-top:12px;background:#fff7f7;color:#111827;">
      <div style="font-size:17px;font-weight:700;color:#991b1b;margin-bottom:8px;">Hugging Face model erisimi gerekli</div>
      <div>Secilen adapter gated backbone kullanıyor. Tahmin icin Colab secret olarak <b>HF_TOKEN</b> ekleyin.</div>
      <ol style="margin:8px 0 0 20px;padding:0;">
        <li>Hugging Face hesabinizda model erisimini onaylayin.</li>
        <li>Colab sol panel Secrets bolumune <b>HF_TOKEN</b> ekleyin.</li>
        <li>Runtime'i yeniden baslatin ve notebook bootstrap hucresini tekrar calistirin.</li>
      </ol>
      <details style="margin-top:10px;">
        <summary style="cursor:pointer;color:#111827;font-weight:600;">Teknik hata</summary>
        <pre style="white-space:pre-wrap;word-break:break-word;color:#7f1d1d;">{escape(str(exc))}</pre>
      </details>
    </div>
    """


def _upload_via_colab_files(upload_dir: Path) -> Optional[Path]:
    try:
        from google.colab import files
    except Exception:
        return None

    uploaded = files.upload()
    if not uploaded:
        return None
    upload_name, upload_bytes = next(iter(uploaded.items()))
    target_path = upload_dir / Path(str(upload_name)).name
    target_path.write_bytes(bytes(upload_bytes))
    return target_path


def _build_result_html(summary: dict[str, Any], result: dict[str, Any], image_path: Path) -> str:
    status = str(result.get("status", "")).strip().lower() or "unknown"
    confidence_value = result.get("confidence")
    confidence_text = "-"
    if confidence_value is not None and status != "error":
        confidence_text = f"{float(confidence_value) * 100.0:.2f}%"
    is_ood = result.get("is_ood")
    if status == "error":
        ood_label = "-"
    elif is_ood is True:
        ood_label = "OOD"
    elif is_ood is False:
        ood_label = "In-distribution"
    else:
        ood_label = "-"
    threshold = result.get("decision_threshold")
    score = result.get("primary_score")
    score_text = "-" if score is None else f"{float(score):.4f}"
    threshold_text = "-" if threshold is None else f"{float(threshold):.4f}"
    error_text = str(result.get("error") or "").strip()
    view_consistency = dict(result.get("view_consistency", {}))
    uncertainty = dict(result.get("uncertainty_diagnostics", {}))
    robustness_warning_codes = [str(code) for code in list(view_consistency.get("warning_codes", []))]
    uncertainty_warning_codes = [str(code) for code in list(uncertainty.get("warning_codes", []))]
    if status == "error":
        robustness_status = "Prediction failed"
        robustness_accent = "#b91c1c"
    elif bool(view_consistency.get("stable")):
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
    warning_details = _warning_details_html(view_consistency, uncertainty)
    visualization = dict(result.get("visualization", {}))
    visualization_line = ""
    if visualization.get("status") == "unavailable":
        visualization_line = (
            '<div style="color:#b45309;"><b style="color:#92400e;">Gorsel Aciklama:</b> '
            f"{escape(str(visualization.get('method') or '-'))} hazirlanamadi: "
            f"{escape(str(visualization.get('error') or '-'))}</div>"
        )
    elif visualization.get("method") == "occlusion_sensitivity":
        visualization_line = (
            '<div style="color:#374151;"><b style="color:#111827;">Gorsel Aciklama:</b> '
            f"{escape(str(visualization.get('view_name') or '-'))} gorunumu icin occlusion sensitivity haritasi hazir.</div>"
        )
    elif visualization.get("method") == "attention_map":
        visualization_line = (
            '<div style="color:#374151;"><b style="color:#111827;">Gorsel Aciklama:</b> '
            f"{escape(str(visualization.get('view_name') or '-'))} gorunumu icin attention map hazir.</div>"
        )
    return f"""
    <div style="border:1px solid #d0d7de;border-radius:10px;padding:16px;margin-top:12px;background:#ffffff;color:#111827;box-shadow:0 1px 3px rgba(15,23,42,0.08);">
      <div style="font-size:18px;font-weight:700;margin-bottom:8px;color:#111827;">{'Tahmin Basarisiz' if status == 'error' else 'Tahmin Sonucu'}</div>
      <div style="color:#374151;"><b style="color:#111827;">Adapter:</b> {escape(str(summary['resolved_adapter_dir']))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Crop:</b> {escape(str(summary['crop_name']))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Status:</b> {escape(status)}</div>
      <div style="color:#374151;"><b style="color:#111827;">Sinif:</b> {escape(str(result.get('predicted_class') or '-'))}</div>
      <div style="color:#374151;"><b style="color:#111827;">Confidence:</b> {confidence_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">OOD Karari:</b> {ood_label}</div>
      <div style="color:#374151;"><b style="color:#111827;">OOD Score:</b> {score_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">Karar Esigi:</b> {threshold_text}</div>
      <div style="color:#374151;"><b style="color:#111827;">Robustluk:</b> <span style="color:{robustness_accent};font-weight:600;">{escape(robustness_status)}</span></div>
      {visualization_line}
      <div style="margin-top:6px;color:#374151;"><b style="color:#111827;">Robustluk Aciklamasi:</b>{warning_details}</div>
      <div style="color:#374151;word-break:break-word;"><b style="color:#111827;">Goruntu:</b> {escape(str(image_path))}</div>
      {'<div style="color:#b91c1c;word-break:break-word;"><b style="color:#991b1b;">Hata:</b> ' + escape(error_text) + '</div>' if error_text else ''}
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
    show_all_adapters: bool = False,
    show_mirror_adapters: bool = False,
    config_env: str = "colab",
    device: str = "cuda",
    upload_dir_name: str = "notebook4_uploads",
    enable_prediction_visualization: bool = True,
    explanation_method: str = "attention_map",
    explanation_grid_size: int = 7,
) -> None:
    """Render the minimal direct-adapter smoke-test UI used by Notebook 4.

    ``show_all_adapters`` is kept for older notebook copies, but mirror exports
    are hidden unless ``show_mirror_adapters`` is explicitly enabled.
    """
    if widgets is None:
        raise RuntimeError(
            "This notebook UI requires ipywidgets. Re-run the bootstrap cell after dependency installation."
        )

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

    _ensure_adapter_smoke_imports()
    adapter_candidates = _discover_adapter_candidates(
        resolved_search_roots,
        crop_name=None,
        collapse_run_mirrors=not show_mirror_adapters,
    )
    dropdown_options = [
        (candidate["display_name"], index)
        for index, candidate in enumerate(adapter_candidates)
    ] or [("Adapter bulunamadi, asagidan yol girin", -1)]

    title = widgets.HTML("<h3 style=\"margin:0 0 8px 0;\">Basit Adapter Testi</h3>")
    help_parts = ["Adapter secin veya yol girin."]
    if _running_in_colab():
        help_parts.append("Yeni tahmin icin hucreyi tekrar calistirmadan resim yukleyin veya mevcut resim yolunu degistirin.")
    else:
        help_parts.append("Colab disinda <b>Resim</b> alanina mevcut dosya yolunu girin.")
    help_text = widgets.HTML("<p style=\"margin:0 0 12px 0;\">" + " ".join(help_parts) + "</p>")
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
    image_path_text = widgets.Text(
        value="",
        placeholder="Mevcut dosya yolunu girin veya asagidaki yukleme dugmesini kullanin",
        description="Resim:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    upload_widget = widgets.FileUpload(
        accept="image/*",
        multiple=False,
        description="Resim Yukle",
        layout=widgets.Layout(width="180px"),
    )
    visualization_checkbox = widgets.Checkbox(
        value=bool(enable_prediction_visualization),
        description="Gorsel aciklama",
        indent=False,
        layout=widgets.Layout(width="95%"),
    )
    explanation_method_dropdown = widgets.Dropdown(
        options=[
            ("Attention map", "attention_map"),
            ("Occlusion sensitivity", "occlusion_sensitivity"),
        ],
        value=str(explanation_method),
        description="Yontem:",
        layout=widgets.Layout(width="95%"),
        style={"description_width": "80px"},
    )
    refresh_button = widgets.Button(description="Adapterleri Yenile", button_style="info")
    clear_image_button = widgets.Button(description="Yeni Resim", button_style="warning")
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
        raw_path = image_path_text.value.strip()
        if raw_path:
            return Path(raw_path).expanduser()
        if _running_in_colab():
            uploaded_path = _upload_via_colab_files(upload_dir)
            if uploaded_path is not None:
                image_path_text.value = str(uploaded_path)
                with status_output:
                    clear_output(wait=True)
                    print(f"Yuklenen resim hazir: {uploaded_path.name}")
                return uploaded_path
            raise ValueError("Colab upload iptal edildi veya dosya secilmedi.")
        raise ValueError("Bir resim yolu girin. Colab'da bos birakirsaniz upload penceresi acilir.")

    def handle_widget_upload(change: Any = None) -> None:
        uploaded_path = _persist_upload_value(upload_widget.value, upload_dir)
        if uploaded_path is None:
            return
        image_path_text.value = str(uploaded_path)
        with status_output:
            clear_output(wait=True)
            print(f"Yuklenen resim hazir: {uploaded_path.name}")

    def clear_image(_button: Any = None) -> None:
        image_path_text.value = ""
        with status_output:
            clear_output(wait=True)
            print("Yeni resim yukleyin veya Resim alanina dosya yolu girin.")

    def render_result(summary: dict[str, Any], result: dict[str, Any], image_path: Path) -> None:
        display(HTML(_build_result_html(summary, result, image_path)))

    def refresh(_button: Any = None) -> None:
        nonlocal adapter_candidates
        with status_output:
            clear_output(wait=True)
            print("Adapter listesi yenileniyor...")
        _ensure_adapter_smoke_imports()
        adapter_candidates = _discover_adapter_candidates(
            resolved_search_roots,
            crop_name=None,
            collapse_run_mirrors=not show_mirror_adapters,
        )
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
            try:
                image_path = resolve_image_path()
                if not image_path.exists():
                    raise FileNotFoundError(f"Resim bulunamadi: {image_path}")
                candidate = selected_candidate()
                _ensure_adapter_smoke_imports()
                summary = _load_adapter_summary(
                    candidate.get("crop_name"),
                    adapter_dir=candidate.get("adapter_dir"),
                    config_env=config_env,
                    device=device,
                )
                with Image.open(image_path) as preview:
                    display(preview.copy())
                _ensure_adapter_smoke_imports()
                result = _predict_single_image(
                    image_path,
                    summary["crop_name"],
                    adapter_dir=summary["resolved_adapter_dir"],
                    config_env=config_env,
                    device=device,
                    enable_robust_smoke=True,
                    explain_prediction=bool(visualization_checkbox.value),
                    explanation_grid_size=int(explanation_grid_size),
                    explanation_method=str(explanation_method_dropdown.value),
                )
                _ensure_adapter_smoke_imports()
                visualization_images = _build_prediction_visualization_images(image_path, result)
                if visualization_images:
                    display(HTML("<div style=\"margin-top:12px;font-weight:700;color:#111827;\">Model gorunumu ve aciklama haritasi</div>"))
                    display(visualization_images["model_view"])
                    display(visualization_images["heatmap_overlay"])
                render_result(summary, result, image_path)
                display(
                    HTML(
                        "<details><summary>Ham JSON</summary><pre>"
                        + json.dumps(result, indent=2)
                        + "</pre></details>"
                    )
                )
            except Exception as exc:
                if _is_huggingface_gated_access_error(exc):
                    display(HTML(_hf_access_error_html(exc)))
                    return
                display(
                    HTML(
                        "<div style=\"border:1px solid #fecaca;border-radius:10px;padding:14px;margin-top:12px;background:#fff7f7;color:#991b1b;\">"
                        "<b>Tahmin hatasi:</b><pre style=\"white-space:pre-wrap;word-break:break-word;\">"
                        + escape(str(exc))
                        + "</pre></div>"
                    )
                )

    refresh_button.on_click(refresh)
    clear_image_button.on_click(clear_image)
    run_button.on_click(run_prediction)
    upload_widget.observe(handle_widget_upload, names="value")
    display(
        widgets.VBox(
            [
                title,
                help_text,
                adapter_dropdown,
                adapter_path_text,
                image_path_text,
                widgets.HBox([upload_widget, clear_image_button]),
                visualization_checkbox,
                explanation_method_dropdown,
                widgets.HBox([refresh_button, run_button]),
                status_output,
                result_output,
            ]
        )
    )
    refresh()


__all__ = [
    "launch_simple_adapter_smoke_ui",
    "_running_in_colab",
    "_ensure_colab_widget_manager",
    "_build_result_html",
    "_persist_upload_value",
    "_upload_via_colab_files",
    "_is_huggingface_gated_access_error",
    "_hf_access_error_html",
]
