import torch
import pytest
from unittest.mock import patch

from src.router.vlm_pipeline import VLMPipeline


def _build_pipeline(stage_order):
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'crop_labels': ['tomato', 'potato'],
                'part_labels': ['leaf', 'whole plant'],
                'policy_graph': {
                    'execution': {
                        'sam3_stage_order': stage_order
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline.models_loaded = True
    pipeline.actual_pipeline = 'sam3'

    pipeline._run_sam3 = lambda image, prompt, threshold=0.7: {
        'masks': torch.ones((1, 4, 4), dtype=torch.float32),
        'boxes': [[0.0, 0.0, 100.0, 100.0]],
        'scores': [0.95],
    }

    def _fake_ensemble(image, labels, label_type='generic', num_prompts=None):
        if label_type == 'part':
            return 'leaf', 0.75, {'leaf': 0.75, 'whole plant': 0.25}
        return 'unknown', 0.30, {'unknown': 0.30, 'tomato': 0.20}

    pipeline._clip_score_labels_ensemble = _fake_ensemble
    pipeline._select_best_crop_with_fallback = (
        lambda roi_image, crop_scores, part_label, part_scores, min_confidence=0.2: ('unknown', 0.30)
    )
    return pipeline


def _build_pipeline_for_dedupe(stage_order):
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'crop_labels': ['tomato', 'potato'],
                'part_labels': ['leaf', 'whole plant'],
                'policy_graph': {
                    'execution': {
                        'sam3_stage_order': stage_order
                    },
                    'dedupe': {
                        'enabled': True,
                        'detection_nms_iou_threshold': 0.75,
                        'detection_nms_same_crop_only': True,
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline.models_loaded = True
    pipeline.actual_pipeline = 'sam3'

    pipeline._run_sam3 = lambda image, prompt, threshold=0.7: {
        'masks': torch.ones((2, 4, 4), dtype=torch.float32),
        'boxes': [[0.0, 0.0, 100.0, 100.0], [5.0, 5.0, 105.0, 105.0]],
        'scores': [0.95, 0.94],
    }

    def _fake_ensemble(image, labels, label_type='generic', num_prompts=None):
        if label_type == 'part':
            return 'leaf', 0.80, {'leaf': 0.80, 'whole plant': 0.20}
        return 'tomato', 0.96, {'tomato': 0.96, 'potato': 0.04}

    pipeline._clip_score_labels_ensemble = _fake_ensemble
    pipeline._select_best_crop_with_fallback = (
        lambda roi_image, crop_scores, part_label, part_scores, min_confidence=0.2: ('tomato', 0.96)
    )
    return pipeline


def test_open_set_gate_enabled_filters_unknown_detection():
    pipeline = _build_pipeline(['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'])

    image = torch.rand(3, 224, 224)
    result = pipeline.analyze_image(image)

    assert result['detections'] == []


def test_open_set_gate_removed_from_stage_order_keeps_unknown_detection():
    pipeline = _build_pipeline(['roi_filter', 'roi_classification', 'postprocess'])

    image = torch.rand(3, 224, 224)
    result = pipeline.analyze_image(image)

    assert len(result['detections']) == 1
    assert result['detections'][0]['crop'] == 'unknown'


def test_postprocess_stage_enabled_dedupes_overlapping_detections():
    pipeline = _build_pipeline_for_dedupe(['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'])

    image = torch.rand(3, 224, 224)
    result = pipeline.analyze_image(image)

    assert len(result['detections']) == 1


def test_postprocess_stage_removed_keeps_overlapping_detections():
    pipeline = _build_pipeline_for_dedupe(['roi_filter', 'roi_classification', 'open_set_gate'])

    image = torch.rand(3, 224, 224)
    result = pipeline.analyze_image(image)

    assert len(result['detections']) == 2


def test_threshold_multiplier_raises_effective_threshold():
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'policy_graph': {
                    'execution': {
                        'confidence_threshold_multiplier': 1.5,
                        'confidence_threshold_min': 0.0,
                        'confidence_threshold_max': 1.0,
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    effective = pipeline._resolve_effective_confidence_threshold(0.4)
    assert effective == pytest.approx(0.6)


def test_threshold_multiplier_respects_min_max_clamps():
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'policy_graph': {
                    'execution': {
                        'confidence_threshold_multiplier': 2.0,
                        'confidence_threshold_min': 0.3,
                        'confidence_threshold_max': 0.7,
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    low_effective = pipeline._resolve_effective_confidence_threshold(0.1)
    high_effective = pipeline._resolve_effective_confidence_threshold(0.5)

    assert low_effective == pytest.approx(0.3)
    assert high_effective == pytest.approx(0.7)


def test_runtime_profile_switch_changes_effective_threshold():
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'profile': 'balanced',
                'policy_graph': {
                    'execution': {
                        'confidence_threshold_multiplier': 1.0,
                        'confidence_threshold_min': 0.0,
                        'confidence_threshold_max': 1.0,
                    }
                },
                'profiles': {
                    'fast': {
                        'policy_graph': {
                            'execution': {
                                'confidence_threshold_multiplier': 1.15
                            }
                        }
                    },
                    'calibrated': {
                        'policy_graph': {
                            'execution': {
                                'confidence_threshold_multiplier': 0.90
                            }
                        }
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline.set_runtime_profile('fast')
    fast_threshold = pipeline._resolve_effective_confidence_threshold(0.4)

    pipeline.set_runtime_profile('calibrated')
    calibrated_threshold = pipeline._resolve_effective_confidence_threshold(0.4)

    assert fast_threshold == pytest.approx(0.46)
    assert calibrated_threshold == pytest.approx(0.36)
    assert fast_threshold > calibrated_threshold


def test_stage_order_invalid_list_falls_back_to_default():
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'policy_graph': {
                    'execution': {
                        'sam3_stage_order': ['invalid_stage', 'another_invalid']
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    assert pipeline._sam3_stage_order() == ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']


def test_stage_order_normalizes_and_deduplicates_values():
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'policy_graph': {
                    'execution': {
                        'sam3_stage_order': ['roi_filter', 'postprocess', 'roi_filter', 'open_set_gate', 'invalid']
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    assert pipeline._sam3_stage_order() == ['roi_filter', 'postprocess', 'open_set_gate']


def test_leaf_fruit_production_profile_loads():
    """Test that leaf_fruit_production profile loads without errors."""
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'profile': 'leaf_fruit_production',
                'focus_part_mode_enabled': False,  # global default
                'focus_parts': ['leaf'],
                'focus_min_confidence_fallback': 0.50,
                'focus_fallback_enabled': True,
                'profiles': {
                    'leaf_fruit_production': {
                        'focus_part_mode_enabled': True,
                        'focus_parts': ['leaf', 'fruit'],
                        'focus_min_confidence_fallback': 0.55,
                        'focus_fallback_enabled': True,
                        'policy_graph': {
                            'execution': {
                                'sam3_stage_order': ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']
                            }
                        }
                    }
                },
                'policy_graph': {
                    'execution': {
                        'sam3_stage_order': ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']
                    }
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    # Verify profile is active
    assert pipeline.vlm_config.get('profile') == 'leaf_fruit_production'


def test_focus_mode_keys_in_runtime_settings():
    """Test that focus mode keys are properly extracted into runtime settings."""
    config = _build_pipeline({}).config
    
    # Enable focus mode in config
    config['router']['vlm']['focus_part_mode_enabled'] = True
    config['router']['vlm']['focus_parts'] = ['leaf', 'fruit']
    config['router']['vlm']['focus_min_confidence_fallback'] = 0.55
    config['router']['vlm']['focus_fallback_enabled'] = True
    
    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')
    
    # Build runtime settings
    settings = pipeline._build_sam3_runtime_settings(0.25)
    
    # Check that focus keys are present and correct
    assert settings.get('focus_part_mode_enabled') is True
    assert settings.get('focus_parts') == ['leaf', 'fruit']
    assert settings.get('focus_min_confidence_fallback') == 0.55
    assert settings.get('focus_fallback_enabled') is True


def test_focus_mode_confidence_fallback_logic():
    """Test post-classification focus confidence-based fallback."""
    pipeline = _build_pipeline(['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'])
    
    # Prepare config with focus mode
    pipeline.vlm_config['focus_part_mode_enabled'] = True
    pipeline.vlm_config['focus_parts'] = ['leaf']
    pipeline.vlm_config['focus_min_confidence_fallback'] = 0.60
    pipeline.vlm_config['focus_fallback_enabled'] = True
    
    # Build settings
    settings = pipeline._build_sam3_runtime_settings(0.25)
    
    # Create mock candidates
    candidates = [
        {'bbox': [0.0, 0.0, 100.0, 100.0], 'sam3_score': 0.95}
    ]
    
    # Create mock pil_image
    pil_image = torch.ones((3, 256, 256))
    
    # Mock _classify_sam3_roi_candidate to return different scenarios
    def mock_classify_high_confidence(pil_image, candidate, image_width, image_height, settings):
        """Mock: high confidence leaf detection."""
        return {
            'bbox': candidate['bbox'],
            'part': 'leaf',
            'part_confidence': 0.85,  # High confidence, should pass
            'crop': 'tomato',
            'crop_confidence': 0.75
        }, 1
    
    with patch.object(pipeline, '_classify_sam3_roi_candidate', side_effect=mock_classify_high_confidence):
        detections, roi_kept, calls, ms = pipeline._run_sam3_roi_classification_stage(
            pil_image, candidates, 256, 256, settings, 
            ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']
        )
        # Should keep the detection because confidence (0.85) >= threshold (0.60)
        assert len(detections) > 0, "Should keep leaf with high confidence"
    
    # Test low confidence scenario
    def mock_classify_low_confidence(pil_image, candidate, image_width, image_height, settings):
        """Mock: low confidence leaf detection - should trigger fallback."""
        return {
            'bbox': candidate['bbox'],
            'part': 'leaf',
            'part_confidence': 0.40,  # Low confidence, should trigger fallback
            'crop': 'tomato',
            'crop_confidence': 0.75
        }, 1
    
    with patch.object(pipeline, '_classify_sam3_roi_candidate', side_effect=mock_classify_low_confidence):
        # With fallback enabled, should still keep the detection
        detections, roi_kept, calls, ms = pipeline._run_sam3_roi_classification_stage(
            pil_image, candidates, 256, 256, settings,
            ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']
        )
        # With fallback, should revert to all detections (1 in this case)
        assert len(detections) > 0, "Should fall back to all detections when confidence too low"


def test_focus_mode_no_detections_fallback():
    """Test that focus mode falls back when no focused ROIs are found."""
    pipeline = _build_pipeline(['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'])
    
    # Prepare config with focus mode looking for fruit
    pipeline.vlm_config['focus_part_mode_enabled'] = True
    pipeline.vlm_config['focus_parts'] = ['fruit']  # Looking for fruit
    pipeline.vlm_config['focus_min_confidence_fallback'] = 0.50
    pipeline.vlm_config['focus_fallback_enabled'] = True
    
    settings = pipeline._build_sam3_runtime_settings(0.25)
    candidates = [
        {'bbox': [0.0, 0.0, 100.0, 100.0], 'sam3_score': 0.95}
    ]
    pil_image = torch.ones((3, 256, 256))
    
    # Mock classifier to return leaf (not fruit)
    def mock_classify_non_focus(pil_image, candidate, image_width, image_height, settings):
        return {
            'bbox': candidate['bbox'],
            'part': 'leaf',  # Not in focus_parts ['fruit']
            'part_confidence': 0.75,
            'crop': 'tomato',
            'crop_confidence': 0.75
        }, 1
    
    with patch.object(pipeline, '_classify_sam3_roi_candidate', side_effect=mock_classify_non_focus):
        detections, roi_kept, calls, ms = pipeline._run_sam3_roi_classification_stage(
            pil_image, candidates, 256, 256, settings,
            ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']
        )
        # With fallback enabled, should revert to all detections (1 leaf in this case)
        assert len(detections) > 0, "Should fall back when no focused parts found"

