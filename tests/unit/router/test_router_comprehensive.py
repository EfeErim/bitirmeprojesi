import json

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


def test_router_comprehensive_resolve_effective_confidence_threshold_policy_multiplier_and_clamp():
    config = {
        'router': {
            'vlm': {
                'enabled': False,
                'policy_graph': {
                    'execution': {
                        'confidence_threshold_multiplier': 1.2,
                        'confidence_threshold_min': 0.25,
                        'confidence_threshold_max': 0.6,
                    }
                },
            }
        }
    }
    pipeline = VLMPipeline(config=config, device='cpu')

    assert pipeline._resolve_effective_confidence_threshold(0.2) == 0.25
    assert pipeline._resolve_effective_confidence_threshold(0.5) == 0.6


def test_router_comprehensive_open_set_gate_rejects_unknown_and_low_confidence():
    assert VLMPipeline._passes_open_set_gate('unknown', 0.99, 0.4) is False
    assert VLMPipeline._passes_open_set_gate('tomato', 0.39, 0.4) is False
    assert VLMPipeline._passes_open_set_gate('tomato', 0.4, 0.4) is True


def test_router_comprehensive_postprocess_sorts_strips_quality_and_caps():
    pipeline = VLMPipeline(config={'vlm_enabled': False}, device='cpu')
    detections = [
        {'crop': 'tomato', 'bbox': [0, 0, 10, 10], '_quality_score': 0.5},
        {'crop': 'potato', 'bbox': [1, 1, 9, 9], '_quality_score': 0.9},
    ]
    settings = {
        'detection_nms_iou_threshold': 0.75,
        'detection_nms_same_crop_only': True,
    }

    out = pipeline._postprocess_sam3_detections(
        detections=detections,
        settings=settings,
        effective_max_detections=1,
        stage_order=['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'],
    )

    assert len(out) == 1
    assert out[0]['crop'] == 'potato'
    assert '_quality_score' not in out[0]


def test_router_comprehensive_build_runtime_settings_contains_required_keys():
    pipeline = VLMPipeline(config={'vlm_enabled': False}, device='cpu')
    settings = pipeline._build_sam3_runtime_settings(effective_threshold=0.4)

    required = {
        'sam3_threshold',
        'classification_min_confidence',
        'detection_nms_iou_threshold',
        'generic_part_labels',
        'crop_num_prompts',
        'part_num_prompts',
        'focus_part_mode_enabled',
    }
    assert required <= set(settings.keys())
    assert settings['classification_min_confidence'] >= 0.4


def test_router_comprehensive_load_taxonomy_and_compatibility_helpers(tmp_path):
    taxonomy = {
        'crops': ['tomato'],
        'parts': {'core': ['leaf']},
        'crop_part_compatibility': {'Tomato': ['Leaf']},
    }
    path = tmp_path / 'taxonomy.json'
    path.write_text(json.dumps(taxonomy), encoding='utf-8')

    crops, parts = VLMPipeline._load_taxonomy(str(path))
    compatibility = VLMPipeline._load_crop_part_compatibility(str(path))

    assert crops == ['tomato']
    assert parts == ['leaf']
    assert compatibility == {'tomato': ['leaf']}
