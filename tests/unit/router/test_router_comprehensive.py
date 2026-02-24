import torch

from src.router.vlm_pipeline import DiagnosticScoutingAnalyzer, VLMPipeline


def test_router_comprehensive_pipeline_init_and_process_disabled_path():
    config = {
        'vlm_enabled': False,
        'vlm_confidence_threshold': 0.8,
        'vlm_max_detections': 10,
    }
    pipeline = VLMPipeline(config=config, device='cpu')

    dummy_input = torch.randn(1, 3, 224, 224)
    result = pipeline.process_image(dummy_input)

    assert isinstance(result, dict)
    assert 'status' in result
    assert 'analysis' in result
    assert isinstance(result['analysis'], dict)
    assert 'detections' in result['analysis']


def test_router_comprehensive_diagnostic_quick_assessment_contract():
    config = {
        'vlm_enabled': False,
        'vlm_confidence_threshold': 0.8,
        'vlm_max_detections': 10,
    }
    analyzer = DiagnosticScoutingAnalyzer(config, device='cpu')

    dummy_input = torch.randn(1, 3, 224, 224)
    quick_result = analyzer.quick_assessment(dummy_input)

    assert isinstance(quick_result, dict)
    assert 'status' in quick_result
    assert 'explanation' in quick_result