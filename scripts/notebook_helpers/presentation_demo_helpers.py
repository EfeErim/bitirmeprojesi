from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Dict

PRESENTATION_STYLES = """
<style>
  .aads-demo {font-family: Arial, sans-serif; color: #102c40;}
  .aads-demo h2 {background: #071b2c; color: white; padding: 12px 16px; border-radius: 10px;}
  .aads-demo h3 {color: #12364d; margin-top: 22px;}
  .aads-demo .flow {font-size: 17px; font-weight: 700; color: #12364d; padding: 12px; background: #e8f7f5;}
  .aads-demo .flow-grid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0 18px;}
  .aads-demo .stage-card {padding: 12px; border: 2px solid #58a99f; border-radius: 12px; background: #f5fbfb;}
  .aads-demo .stage-card.success {border-color: #3a9d68; background: #f2fbf6;}
  .aads-demo .stage-card.warning {border-color: #e59b43; background: #fff9ef;}
  .aads-demo .stage-number {font-size: 20px; color: #58a99f; font-weight: 700;}
  .aads-demo .stage-title {font-size: 17px; color: #12364d; font-weight: 700; margin: 4px 0 7px;}
  .aads-demo .stage-purpose {font-size: 13px; min-height: 83px; margin: 9px 0;}
  .aads-demo .stage-output {font-size: 13px; border-top: 1px solid #c6d7df; padding-top: 8px;}
  .aads-demo .badge {display: inline-block; color: white; background: #2b7c75; padding: 3px 7px; border-radius: 10px; font-size: 11px; font-weight: 700;}
  .aads-demo .badge.success {background: #3a9d68;}
  .aads-demo .badge.warning {background: #d2852f;}
  .aads-demo .result-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 18px 0;}
  .aads-demo .result-card {padding: 14px; border-radius: 12px; background: #edf5f7; border-left: 5px solid #2b7c75;}
  .aads-demo .result-card strong {display: block; color: #12364d; margin-bottom: 5px;}
  .aads-demo table {border-collapse: collapse; width: 100%; margin: 8px 0 16px;}
  .aads-demo th, .aads-demo td {border: 1px solid #c6d7df; padding: 7px 9px; text-align: left;}
  .aads-demo th {background: #edf5f7; width: 32%;}
  .aads-demo .note {background: #fff4df; padding: 10px 12px; border-left: 5px solid #f4a261;}
</style>
"""


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _text(value: Any, default: str = "-") -> str:
    normalized = str(value or "").strip()
    return normalized or default


def _yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _badge(label: str, *, tone: str = "info") -> str:
    return f"<span class='badge {escape(tone)}'>{escape(label)}</span>"


def _flow_card(
    number: int,
    title: str,
    purpose: str,
    output: str,
    *,
    badge: str,
    tone: str = "info",
) -> str:
    return (
        f"<div class='stage-card {escape(tone)}'>"
        f"<div class='stage-number'>{number:02d}</div>"
        f"<div class='stage-title'>{escape(title)}</div>"
        f"{_badge(badge, tone=tone)}"
        f"<div class='stage-purpose'>{escape(purpose)}</div>"
        f"<div class='stage-output'><strong>Result for this image:</strong><br>{escape(output)}</div>"
        "</div>"
    )


MAX_PRESENTATION_BOXES = 3


def _select_presentation_detections(detections: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return detections[:MAX_PRESENTATION_BOXES]


def build_presentation_summary(
    image_path: str | Path,
    *,
    router_result: Dict[str, Any],
    auto_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract recording-friendly facts from the maintained inference payloads."""
    router_details = _as_dict(router_result.get("router_details"))
    diagnostics = _as_dict(router_result.get("diagnostics"))
    adapter_target = _as_dict(router_result.get("adapter_target"))
    handoff = _as_dict(auto_result.get("router_handoff"))
    ood = _as_dict(auto_result.get("ood_analysis"))
    input_guard = _as_dict(auto_result.get("input_guard"))
    detections = [_as_dict(item) for item in _as_list(router_details.get("detections"))]
    timings = _as_dict(router_details.get("stage_timings_ms"))
    roi_stats = _as_dict(router_details.get("roi_stats"))
    candidates = [_as_dict(item) for item in _as_list(diagnostics.get("top_crop_candidates"))]

    notebook_gate = _as_dict(router_result.get("notebook_gate"))
    if notebook_gate:
        gate_status = "Accepted" if notebook_gate.get("accepted") else "Rejected"
        gate_reason = "; ".join(str(item) for item in _as_list(notebook_gate.get("reasons"))) or "Threshold checks passed."
    else:
        gate_status = "Not applied"
        gate_reason = "Notebook-level crop confidence gate was not applied."

    adapter_ran = bool(handoff.get("adapter_ran"))
    diagnosis = auto_result.get("diagnosis")
    final_status = _text(auto_result.get("status"), "unknown")
    final_message = _text(auto_result.get("message"), "")
    if diagnosis:
        final_decision = f"Model prediction: {diagnosis} ({_float(auto_result.get('confidence')):.3f})"
    elif final_message:
        final_decision = f"No disease prediction: {final_message}"
    else:
        final_decision = f"No disease prediction: status={final_status}"

    return {
        "image_path": str(image_path),
        "pipeline_type": _text(router_details.get("pipeline_type"), "sam3_bioclip25"),
        "processing_time_ms": _float(router_details.get("processing_time_ms")),
        "sam3_instances_raw": int(router_details.get("sam3_instances_raw", router_details.get("sam3_instances", 0)) or 0),
        "sam3_instances_retained": int(router_details.get("sam3_instances_retained", len(detections)) or 0),
        "roi_seen": int(roi_stats.get("seen", 0) or 0),
        "roi_retained": int(roi_stats.get("retained", 0) or 0),
        "roi_classification_calls": int(roi_stats.get("classification_calls", 0) or 0),
        "stage_timings_ms": timings,
        "detections": detections,
        "top_crop_candidates": candidates,
        "crop": _text(router_result.get("crop"), "unknown"),
        "part": _text(router_result.get("part"), "unknown"),
        "router_confidence": _float(router_result.get("router_confidence")),
        "crop_margin": _float(diagnostics.get("crop_confidence_margin")),
        "raw_part_label": _text(diagnostics.get("raw_part_label"), "unknown"),
        "raw_part_confidence": _float(diagnostics.get("raw_part_confidence")),
        "part_unknown_confidence": _float(diagnostics.get("part_unknown_confidence")),
        "part_rejection_reason": _text(diagnostics.get("part_rejection_reason"), ""),
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "adapter_dir": _text(adapter_target.get("adapter_dir")),
        "adapter_exists": bool(adapter_target.get("exists")),
        "adapter_ran": adapter_ran,
        "handoff_status": _text(handoff.get("source_status"), "unknown"),
        "final_status": final_status,
        "final_decision": final_decision,
        "ood_available": bool(ood),
        "ood_method": _text(ood.get("score_method")),
        "ood_score": _float(ood.get("primary_score")),
        "ood_threshold": _float(ood.get("decision_threshold")),
        "is_ood": bool(ood.get("is_ood")),
        "input_guard_available": bool(input_guard),
        "input_guard_decision": _text(input_guard.get("decision")),
        "input_guard_plant_like": bool(input_guard.get("is_plant_like")),
    }


def build_presentation_flow_html(summary: Dict[str, Any]) -> str:
    """Build the top-level visual explanation for a non-technical audience."""
    adapter_tone = "success" if summary["adapter_ran"] else "warning"
    final_tone = "success" if summary["adapter_ran"] and not summary["is_ood"] else "warning"
    ood_result = (
        "Input is outside the calibrated support."
        if summary["is_ood"]
        else "Input is within calibrated support. This does not confirm that the predicted class is correct."
        if summary["ood_available"]
        else "No OOD assessment was produced because adapter inference did not run."
    )
    cards = (
        _flow_card(
            1,
            "Input Image",
            "The system receives the single plant image uploaded by the user.",
            Path(summary["image_path"]).name,
            badge="INPUT",
        ),
        _flow_card(
            2,
            "SAM3 Region Proposal",
            "SAM3 proposes plant regions to inspect. The boxes are not disease predictions.",
            (
                f"{summary['sam3_instances_raw']} raw regions found; "
                f"{summary['sam3_instances_retained']} regions retained after filtering."
            ),
            badge="ROI PROPOSALS",
        ),
        _flow_card(
            3,
            "BioCLIP-2.5 Router",
            "BioCLIP-2.5 scores crop and plant-part evidence from the full image and proposed regions.",
            (
                f"Selected route: crop={summary['crop']}, part={summary['part']}, "
                f"crop confidence={summary['router_confidence']:.3f}."
            ),
            badge="CROP + PART",
        ),
        _flow_card(
            4,
            "Safety Gate",
            "If evidence is weak or conflicting, the system abstains instead of loading an adapter.",
        summary["gate_status"],
            badge=summary["gate_status"].upper(),
            tone="success" if summary["gate_status"] == "Accepted" else "warning",
        ),
        _flow_card(
            5,
            "SD-LoRA Adapter",
            "After an accepted route, the system loads the specialist disease adapter trained for the selected crop and part.",
            f"Specialist adapter loaded={_yes_no(summary['adapter_ran'])}.",
            badge="LOADED" if summary["adapter_ran"] else "SKIPPED",
            tone=adapter_tone,
        ),
        _flow_card(
            6,
            "Adapter Output + OOD Check",
            "The specialist adapter returns a model prediction. OOD checks calibrated support; it does not verify class correctness.",
            f"{summary['final_decision']} {ood_result}",
            badge="RESULT",
            tone=final_tone,
        ),
    )
    return (
        "<div class='flow-grid'>"
        + "".join(cards)
        + "</div>"
    )


def _render_router_figure(summary: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    image = Image.open(summary["image_path"]).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    visible_detections = _select_presentation_detections(summary["detections"])
    for detection in visible_detections:
        bbox = _as_list(detection.get("bbox"))
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [_float(value) for value in bbox[:4]]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        x1, x2 = max(0.0, x1), min(width - 1.0, x2)
        y1, y2 = max(0.0, y1), min(height - 1.0, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        color = (66, 133, 244)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(image)
    axes[0].set_title("1. Uploaded input image")
    axes[1].imshow(annotated)
    axes[1].set_title(f"2. SAM3 candidate regions for inspection (showing {len(visible_detections)})")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


def render_presentation_demo(
    image_path: str | Path,
    *,
    router_result: Dict[str, Any],
    auto_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Render one detailed, recording-friendly notebook output from real payloads."""
    from IPython.display import HTML, display

    summary = build_presentation_summary(image_path, router_result=router_result, auto_result=auto_result)
    header = (
        f"{PRESENTATION_STYLES}<div class='aads-demo'><h2>AADS v6 - Detailed Router-to-Adapter Demo</h2>"
        "<div class='flow'>Input image &rarr; SAM3 region proposals &rarr; BioCLIP-2.5 routing "
        "&rarr; safety gate &rarr; specialist SD-LoRA adapter &rarr; disease + OOD result</div>"
        + build_presentation_flow_html(summary)
        + "<p class='note'><strong>Important:</strong> SAM3 boxes are not disease predictions. "
        "They mark candidate plant regions for BioCLIP-2.5 inspection.</p>"
        + "<p class='note'><strong>Model limitation:</strong> The adapter output is a model prediction, not a verified diagnosis. "
        "An in-distribution OOD result does not prove that the predicted class is correct.</p>"
        + "</div>"
    )
    display(HTML(header))
    _render_router_figure(summary)

    remaining = (
        "<div class='aads-demo'><h3>Audience Summary</h3>"
        "<div class='result-grid'>"
        f"<div class='result-card'><strong>BioCLIP-2.5 route</strong>Crop: {escape(summary['crop'])}<br>"
        f"Plant part: {escape(summary['part'])}</div>"
        f"<div class='result-card'><strong>Safety decision</strong>{escape(summary['gate_status'])}<br>"
        f"Specialist adapter loaded: {_yes_no(summary['adapter_ran'])}</div>"
        f"<div class='result-card'><strong>Adapter model prediction (not a verified diagnosis)</strong>"
        f"{escape(summary['final_decision'])}</div>"
        f"<div class='result-card'><strong>OOD assessment</strong>"
        f"{'Out-of-distribution input' if summary['is_ood'] else 'Within calibrated support; class correctness is not guaranteed'}</div>"
        "</div>"
        + "</div>"
    )
    display(HTML(remaining))
    return summary


__all__ = [
    "PRESENTATION_STYLES",
    "build_presentation_flow_html",
    "build_presentation_summary",
    "render_presentation_demo",
]
