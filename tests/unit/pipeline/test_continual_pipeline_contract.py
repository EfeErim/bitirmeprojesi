import torch

from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline


class DummyAdapter:
    def __init__(self):
        self.is_trained = True
        self.engine = 'continual_sd_lora'
        self.class_to_idx = {'healthy': 0}

    def predict_with_ood(self, image):
        return {
            'status': 'success',
            'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.9},
            'ood_analysis': {
                'ensemble_score': 0.2,
                'class_threshold': 0.8,
                'is_ood': False,
                'calibration_version': 1,
            },
        }


def test_get_crop_status_uses_engine_field():
    cfg = {'router': {'crop_mapping': {'tomato': {}}}}
    pipeline = IndependentMultiCropPipeline(cfg, device='cpu')
    pipeline.adapters['tomato'] = DummyAdapter()

    status = pipeline.get_crop_status()

    assert status['tomato']['is_trained'] is True
    assert status['tomato']['engine'] == 'continual_sd_lora'
    assert status['tomato']['num_classes'] == 1


def test_process_with_adapter_emits_v6_ood_payload():
    cfg = {'router': {'crop_mapping': {'tomato': {}}}}
    pipeline = IndependentMultiCropPipeline(cfg, device='cpu')
    pipeline.adapters['tomato'] = DummyAdapter()

    result = pipeline._process_with_adapter(torch.zeros(1, 3, 224, 224), 'tomato', None, True)

    assert 'ood_analysis' in result
    assert {'ensemble_score', 'class_threshold', 'is_ood', 'calibration_version'} <= set(result['ood_analysis'].keys())
