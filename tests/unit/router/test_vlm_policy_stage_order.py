import torch
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
