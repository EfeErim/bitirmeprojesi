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


def test_router_comprehensive_process_enabled_reports_diagnostic_scouting():
    config = {
        'vlm_enabled': True,
        'vlm_confidence_threshold': 0.8,
        'vlm_max_detections': 10,
    }
    pipeline = VLMPipeline(config=config, device='cpu')

    dummy_input = torch.randn(1, 3, 224, 224)
    result = pipeline.process_image(dummy_input)

    assert result['status'] == 'ok'
    assert result['scenario'] == 'diagnostic_scouting'


def test_router_comprehensive_is_ready_requires_runtime_surfaces():
    pipeline = VLMPipeline(config={'vlm_enabled': True}, device='cpu')
    assert pipeline.is_ready() is False

    pipeline.models_loaded = True
    pipeline.actual_pipeline = 'sam3'
    pipeline.sam_model = object()
    pipeline.sam_processor = object()
    pipeline.bioclip = object()
    pipeline.bioclip_processor = object()
    pipeline.sam_backend = 'sam3'

    assert pipeline.is_ready() is True


def test_router_comprehensive_profile_overlay_updates_stage_order():
    config = {
        'router': {
            'vlm': {
                'enabled': False,
                'policy_graph': {
                    'execution': {
                        'sam3_stage_order': ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'],
                    }
                },
                'profiles': {
                    'fast': {
                        'policy_graph': {
                            'execution': {
                                'sam3_stage_order': ['roi_filter', 'open_set_gate'],
                            }
                        }
                    }
                },
            }
        }
    }

    pipeline = VLMPipeline(config=config, device='cpu')
    applied = pipeline.set_runtime_profile('fast')

    assert applied is True
    assert pipeline.active_profile == 'fast'
    assert pipeline._sam3_stage_order() == ['roi_filter', 'open_set_gate']


def test_router_comprehensive_route_batch_maps_detection_and_unknown_paths():
    pipeline = VLMPipeline(config={'vlm_enabled': False}, device='cpu')
    responses = [
        {
            'detections': [
                {'crop': 'tomato', 'part': 'leaf', 'bbox': [1, 2, 3, 4], 'crop_confidence': 0.93}
            ]
        },
        {'detections': []},
    ]
    pipeline.analyze_image = lambda _image: responses.pop(0)  # type: ignore[assignment]

    batch = torch.randn(2, 3, 32, 32)
    crops_out, confs = pipeline.route_batch(batch)

    assert crops_out[0]['crop'] == 'tomato'
    assert crops_out[0]['part'] == 'leaf'
    assert confs[0] == 0.93
    assert crops_out[1]['crop'] == 'unknown'
    assert confs[1] == 0.0


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
