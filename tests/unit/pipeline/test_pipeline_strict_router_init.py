import pytest
from unittest.mock import patch

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline


def test_initialize_router_raises_in_strict_mode_when_models_missing():
    config = {
        'crops': ['tomato'],
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
        pipeline = IndependentMultiCropPipeline(config, device='cpu')

    with pytest.raises(RuntimeError):
        pipeline.initialize_router()
