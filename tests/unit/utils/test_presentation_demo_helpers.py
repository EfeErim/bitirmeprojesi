from scripts.notebook_helpers.presentation_demo_helpers import (
    _proposal_label,
    build_presentation_flow_html,
    build_presentation_summary,
)


def test_build_presentation_summary_exposes_real_router_and_adapter_details() -> None:
    summary = build_presentation_summary(
        "sample.jpg",
        router_result={
            "crop": "tomato",
            "part": "leaf",
            "router_confidence": 0.92,
            "router_details": {
                "pipeline_type": "sam3_bioclip25",
                "processing_time_ms": 123.4,
                "sam3_instances_raw": 4,
                "sam3_instances_retained": 2,
                "roi_stats": {"seen": 4, "retained": 2, "classification_calls": 4},
                "stage_timings_ms": {"sam3_inference": 50.0, "roi_classification": 60.0},
                "detections": [{"crop": "tomato", "part": "leaf", "bbox": [1, 2, 3, 4]}],
            },
            "diagnostics": {
                "crop_confidence_margin": 0.31,
                "raw_part_label": "leaf",
                "raw_part_confidence": 0.81,
                "top_crop_candidates": [{"crop": "tomato", "part": "leaf", "crop_confidence": 0.92}],
            },
            "notebook_gate": {"accepted": True, "reasons": []},
            "adapter_target": {"adapter_dir": "models/adapters/tomato/leaf", "exists": True},
        },
        auto_result={
            "status": "ok",
            "diagnosis": "late_blight",
            "confidence": 0.87,
            "router_handoff": {"adapter_ran": True, "source_status": "ok"},
            "ood_analysis": {"score_method": "ensemble", "primary_score": 0.12, "decision_threshold": 0.5},
        },
    )

    assert summary["pipeline_type"] == "sam3_bioclip25"
    assert summary["sam3_instances_raw"] == 4
    assert summary["adapter_ran"] is True
    assert summary["final_decision"] == "late_blight (0.870)"
    assert summary["ood_available"] is True


def test_build_presentation_summary_keeps_abstention_visible() -> None:
    summary = build_presentation_summary(
        "sample.jpg",
        router_result={
            "crop": "tomato",
            "part": "unknown",
            "diagnostics": {"part_rejection_reason": "part confidence below threshold"},
            "notebook_gate": {"accepted": True, "reasons": []},
        },
        auto_result={
            "status": "router_uncertain",
            "message": "Router could not resolve a supported plant part.",
            "router_handoff": {"adapter_ran": False, "source_status": "ok"},
        },
    )

    assert summary["part"] == "unknown"
    assert summary["adapter_ran"] is False
    assert summary["ood_available"] is False
    assert summary["final_decision"].startswith("No disease prediction:")


def test_build_presentation_flow_html_explains_each_model_role() -> None:
    html = build_presentation_flow_html(
        {
            "image_path": "sample.jpg",
            "sam3_instances_raw": 4,
            "sam3_instances_retained": 2,
            "crop": "tomato",
            "part": "leaf",
            "router_confidence": 0.92,
            "gate_status": "Accepted",
            "gate_reason": "Threshold checks passed.",
            "adapter_ran": True,
            "adapter_dir": "models/adapters/tomato/leaf",
            "final_decision": "late_blight (0.870)",
            "ood_available": True,
            "ood_score": 0.12,
            "ood_threshold": 0.5,
            "is_ood": False,
        }
    )

    assert "SAM3 Region Proposal" in html
    assert "The boxes are not disease predictions" in html
    assert "BioCLIP-2.5 Router" in html
    assert "SD-LoRA Adapter" in html
    assert "Disease + OOD Result" in html


def test_presentation_figure_labels_sam3_boxes_as_candidate_regions() -> None:
    assert _proposal_label(3) == "Candidate region 3"
