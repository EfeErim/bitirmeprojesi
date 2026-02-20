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
                'model_ids': {}
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    with pytest.raises(RuntimeError, match='Strict VLM model loading failed'):
        pipeline.load_models()


def test_non_strict_loading_falls_back_to_placeholder():
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': False,
        'router': {
            'vlm': {
                'enabled': True,
                'strict_model_loading': False,
                'model_source': 'huggingface',
                'model_ids': {}
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline.load_models()

    assert pipeline.models_loaded is False
    assert pipeline.grounding_dino == 'GroundingDINO model'


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
                    'crop': 'fake-crop-model',
                    'part': 'fake-part-model'
                },
                'crop_labels': ['tomato', 'potato'],
                'part_labels': ['leaf', 'whole']
            }
        }
    }

    crop_model = _FakeModel(torch.tensor([[5.0, 1.0]]), id2label={0: 'tomato', 1: 'potato'})
    part_model = _FakeModel(torch.tensor([[3.0, 0.5]]), id2label={0: 'leaf', 1: 'whole'})

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    def _mock_load(model_id):
        if model_id == 'fake-crop-model':
            return _FakeProcessor(), crop_model
        if model_id == 'fake-part-model':
            return _FakeProcessor(), part_model
        raise ValueError('unexpected model id')

    pipeline._load_hf_classifier = _mock_load
    pipeline.load_models()

    result = pipeline.analyze_image(torch.rand(3, 224, 224))
    detection = result['detections'][0]

    assert pipeline.is_ready() is True
    assert detection['crop'] == 'tomato'
    assert detection['part'] == 'leaf'
    assert detection['crop_confidence'] > 0.9
