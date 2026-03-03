"""Integration tests for v6 continual configuration and adapter flow."""

from pathlib import Path

import torch

from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.core.config_manager import ConfigurationManager
from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer


def test_colab_config_contains_continual_contract():
    manager = ConfigurationManager(config_dir='config')
    cfg = manager.load_all_configs()

    continual = cfg['training']['continual']
    assert continual['backbone']['model_name'] == 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    assert continual['quantization']['mode'] == 'int8_hybrid'


def test_continual_config_rejects_low_bit_payload():
    try:
        ContinualSDLoRAConfig.from_training_config(
            {
                'backbone': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'},
                'quantization': {'mode': 'int8_hybrid'},
                'adapter': {'target_modules_strategy': 'all_linear_transformer', 'lora_r': 4, 'lora_alpha': 8},
                'fusion': {'layers': [2, 5, 8, 11]},
                'load_in_4bit': True,
            }
        )
        assert False, 'low-bit payload should be rejected'
    except ValueError:
        assert True


def test_adapter_metadata_roundtrip_without_model_download(monkeypatch, tmp_path):
    from src.training import continual_sd_lora as continual_module
    from src.adapter import independent_crop_adapter as adapter_module

    class FakeTrainer:
        def __init__(self, config):
            self.config = type('Cfg', (), {
                'backbone_model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
                'fusion_layers': [2, 5, 8, 11],
                'fusion_output_dim': 768,
                'fusion_dropout': 0.1,
                'fusion_gating': 'softmax',
            })()
            self.class_to_idx = {}
            self.target_modules_resolved = ['transformer.block.0.linear']
            self.ood_detector = type('OOD', (), {'calibration_version': 1})()

        @property
        def quantization_metadata(self):
            return {'mode': 'int8_hybrid'}

        def initialize_engine(self, class_to_idx=None):
            self.class_to_idx = dict(class_to_idx or {})

        def add_classes(self, names):
            for n in names:
                if n not in self.class_to_idx:
                    self.class_to_idx[n] = len(self.class_to_idx)
            return dict(self.class_to_idx)

        def train_increment(self, train_loader, num_epochs=None):
            return {'train_loss': [0.1]}

        def calibrate_ood(self, loader):
            return {'num_classes': float(len(self.class_to_idx))}

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

        def save_adapter(self, output_dir):
            root = Path(output_dir) / 'continual_sd_lora_adapter'
            root.mkdir(parents=True, exist_ok=True)
            return root

        def load_adapter(self, adapter_dir):
            return {}

    monkeypatch.setattr(continual_module, 'ContinualSDLoRATrainer', FakeTrainer)
    monkeypatch.setattr(adapter_module, 'ContinualSDLoRATrainer', FakeTrainer)

    adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
    adapter.initialize_engine(class_names=['healthy'])
    adapter.add_classes(['disease_a'])
    adapter.train_increment(train_loader=[{'images': torch.zeros(1, 3, 224, 224), 'labels': torch.zeros(1, dtype=torch.long)}])

    save_dir = tmp_path / 'model'
    adapter.save_adapter(str(save_dir))

    reloaded = IndependentCropAdapter(crop_name='tomato', device='cpu')
    reloaded.load_adapter(str(save_dir / 'continual_sd_lora_adapter'))
    assert reloaded.is_trained is True

