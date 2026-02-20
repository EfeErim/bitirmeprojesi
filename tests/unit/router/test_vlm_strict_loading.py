import pytest
import torch
from unittest.mock import patch

from src.router.vlm_pipeline import VLMPipeline


class _FakeProcessor:
    def __call__(self, images, return_tensors='pt'):
        return {'pixel_values': torch.randn(1, 3, 224, 224)}


class _FakeModelOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _FakeModel:
    def __init__(self, logits: torch.Tensor, id2label=None):
        self._logits = logits
        self.config = type('Config', (), {'id2label': id2label or {}})()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return _FakeModelOutput(self._logits)


def test_strict_loading_requires_model_ids():
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': True,
        'router': {
            'vlm': {
                'enabled': True,
                'strict_model_loading': True,
                'model_source': 'huggingface',
                'model_ids': {
                    'grounding_dino': 'invalid/grounding-dino',
                    'sam': 'invalid/sam',
                    'bioclip': 'invalid/bioclip'
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline._load_grounding_dino = lambda model_id: (_ for _ in ()).throw(ValueError('bad grounding dino id'))

    with pytest.raises(RuntimeError, match='Strict VLM model loading failed'):
        pipeline.load_models()


def test_non_strict_loading_falls_back_to_placeholder():
    """Test that non-strict mode leaves models as None instead of using placeholder strings."""
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': False,
        'router': {
            'vlm': {
                'enabled': True,
                'strict_model_loading': False,
                'model_source': 'huggingface',
                'model_ids': {
                    'grounding_dino': 'invalid/grounding-dino',
                    'sam': 'invalid/sam',
                    'bioclip': 'invalid/bioclip'
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline._load_grounding_dino = lambda model_id: (_ for _ in ()).throw(ValueError('bad grounding dino id'))

    pipeline.load_models()

    # Models should remain None (not placeholder strings) - inspired by reference implementation
    assert pipeline.models_loaded is False
    assert pipeline.grounding_dino is None
    assert pipeline.sam2 is None
    assert pipeline.bioclip is None


def test_strict_loading_with_models_runs_inference():
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': True,
        'router': {
            'crop_mapping': {
                'tomato': {'parts': ['leaf', 'whole']},
                'potato': {'parts': ['leaf', 'tuber']}
            },
            'vlm': {
                'enabled': True,
                'strict_model_loading': True,
                'model_source': 'huggingface',
                'model_ids': {
                    'grounding_dino': 'fake-grounding-dino',
                    'sam': 'fake-sam',
                    'bioclip': 'fake-bioclip'
                },
                'crop_labels': ['tomato', 'potato'],
                'part_labels': ['leaf', 'whole']
            }
        }
    }

    fake_model = _FakeModel(torch.tensor([[5.0, 1.0]]), id2label={0: 'tomato', 1: 'potato'})
    fake_processor = _FakeProcessor()

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline._load_grounding_dino = lambda model_id: (fake_processor, fake_model)
    pipeline._load_sam = lambda model_id: (fake_processor, fake_model)
    pipeline._load_clip_like_model = lambda model_id: (fake_processor, fake_model)
    pipeline.load_models()

    pipeline._run_grounding_dino = lambda image: {
        'detections': [
            {'label': 'tomato leaf', 'score': 0.97, 'bbox': [0.0, 0.0, 100.0, 100.0], 'crop_guess': 'tomato', 'part_guess': 'leaf'}
        ]
    }
    pipeline._clip_score_labels = lambda image, labels, label_type='generic': (labels[0], 0.95)
    pipeline._run_sam_mask = lambda image, bbox: None

    result = pipeline.analyze_image(torch.rand(3, 224, 224))
    detection = result['detections'][0]

    assert pipeline.is_ready() is True
    assert detection['crop'] == 'tomato'
    assert detection['part'] == 'leaf'
    assert detection['crop_confidence'] > 0.9
