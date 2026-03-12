from src.shared.contracts import InferenceResult, RouterAnalysisResult, RouterDetection


def test_router_analysis_result_preserves_router_order_without_quality_scores():
    result = RouterAnalysisResult(
        detections=[
            RouterDetection(crop="tomato", part="leaf", crop_confidence=0.61),
            RouterDetection(crop="potato", part="leaf", crop_confidence=0.95),
        ]
    )

    assert result.primary_detection is not None
    assert result.primary_detection.crop == "tomato"
    assert [detection.crop for detection in result.detections] == ["tomato", "potato"]


def test_router_analysis_result_prefers_quality_score_when_present():
    result = RouterAnalysisResult(
        detections=[
            RouterDetection(crop="tomato", part="leaf", crop_confidence=0.95, quality_score=0.40),
            RouterDetection(crop="potato", part="leaf", crop_confidence=0.61, quality_score=0.75),
        ]
    )

    assert result.primary_detection is not None
    assert result.primary_detection.crop == "potato"
    assert [detection.crop for detection in result.detections] == ["potato", "tomato"]


def test_inference_result_serializes_router_summary_block():
    payload = InferenceResult(
        status="success",
        crop="tomato",
        part="leaf",
        router_confidence=0.91,
        diagnosis="healthy",
        confidence=0.88,
        router=RouterAnalysisResult(
            status="ok",
            message="",
            detections=[
                RouterDetection(crop="tomato", part="leaf", crop_confidence=0.91, quality_score=0.80),
                RouterDetection(crop="potato", part="leaf", crop_confidence=0.92, quality_score=0.20),
            ],
        ),
    ).to_dict(include_ood=False)

    assert payload["router"] == {
        "status": "ok",
        "message": "",
        "detections_count": 2,
        "primary_detection": {
            "crop": "tomato",
            "part": "leaf",
            "crop_confidence": 0.91,
            "part_confidence": 0.0,
            "quality_score": 0.8,
        },
    }
