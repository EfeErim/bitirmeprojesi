import json
from pathlib import Path

import torch

from src.adapter import independent_crop_adapter as adapter_module
from src.adapter.independent_crop_adapter import IndependentCropAdapter


class FakeOOD:
    def __init__(self):
        self.calibration_version = 2


class FakeConfig:
    backbone_model_name = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    fusion_layers = [2, 5, 8, 11]
    fusion_output_dim = 768
    fusion_dropout = 0.1
    fusion_gating = 'softmax'


class FakeTrainer:
    def __init__(self, config):
        self.config = FakeConfig()
        self.class_to_idx = {}
        self.target_modules_resolved = ['transformer.block.0.linear']
        self.ood_detector = FakeOOD()

    def initialize_engine(self, class_to_idx=None):
        self.class_to_idx = dict(class_to_idx or {})

    def add_classes(self, new_classes):
        for name in new_classes:
            if name not in self.class_to_idx:
                self.class_to_idx[name] = len(self.class_to_idx)
        return dict(self.class_to_idx)

    def train_increment(self, train_loader, num_epochs=None, progress_callback=None):
        if progress_callback is not None:
            progress_callback({'epoch': 1, 'batch': 1, 'total_batches': 1, 'batch_loss': 0.1, 'epoch_progress': 1.0})
            progress_callback({'epoch_done': 1, 'epoch_loss': 0.1})
        return {'train_loss': [0.1]}

    def calibrate_ood(self, loader):
        self.ood_detector.calibration_version += 1
        return {'num_classes': float(len(self.class_to_idx))}

    def predict_with_ood(self, image):
        return {
            'status': 'success',
            'disease': {'class_index': 0, 'name': 'healthy', 'confidence': 0.9},
            'ood_analysis': {
                'ensemble_score': 0.2,
                'class_threshold': 0.8,
                'is_ood': False,
                'calibration_version': self.ood_detector.calibration_version,
            },
        }

    def save_adapter(self, output_dir):
        root = Path(output_dir) / 'continual_sd_lora_adapter'
        root.mkdir(parents=True, exist_ok=True)
        (root / 'classifier.pth').write_bytes(b'')
        (root / 'fusion.pth').write_bytes(b'')
        return root

    def load_adapter(self, adapter_dir):
        meta = json.loads((Path(adapter_dir) / 'adapter_meta.json').read_text(encoding='utf-8'))
        self.class_to_idx = dict(meta.get('class_to_idx', {}))
        return meta


def test_adapter_lifecycle_surface(monkeypatch):
    monkeypatch.setattr(adapter_module, 'ContinualSDLoRATrainer', FakeTrainer)

    adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
    initialized = adapter.initialize_engine(class_names=['healthy'])
    assert initialized['status'] == 'initialized'

    added = adapter.add_classes(['disease_a'])
    assert added['num_classes'] == 2

    trained = adapter.train_increment(train_loader=[{'images': torch.zeros(1, 3, 224, 224), 'labels': torch.zeros(1, dtype=torch.long)}])
    assert trained['status'] == 'trained'

    pred = adapter.predict_with_ood(torch.zeros(3, 224, 224))
    assert {'ensemble_score', 'class_threshold', 'is_ood', 'calibration_version'} <= set(pred['ood_analysis'].keys())


def test_adapter_save_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, 'ContinualSDLoRATrainer', FakeTrainer)

    adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
    adapter.initialize_engine(class_names=['healthy', 'disease_a'])

    save_dir = tmp_path / 'model_dir'
    adapter.save_adapter(str(save_dir))

    loaded = IndependentCropAdapter(crop_name='tomato', device='cpu')
    monkeypatch.setattr(adapter_module, 'ContinualSDLoRATrainer', FakeTrainer)
    loaded.load_adapter(str(save_dir / 'continual_sd_lora_adapter'))

    assert loaded.class_to_idx
    assert loaded.is_trained is True


def test_adapter_metadata_contains_required_contract_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, 'ContinualSDLoRATrainer', FakeTrainer)

    adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
    adapter.initialize_engine(class_names=['healthy', 'disease_a'])

    save_dir = tmp_path / 'model_dir'
    adapter.save_adapter(str(save_dir))

    meta_path = save_dir / 'continual_sd_lora_adapter' / 'adapter_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    required = {
        'schema_version',
        'engine',
        'backbone',
        'fusion',
        'class_to_idx',
        'ood_calibration',
        'target_modules_resolved',
    }
    assert required <= set(meta.keys())


def test_adapter_train_increment_forwards_progress_callback(monkeypatch):
    monkeypatch.setattr(adapter_module, 'ContinualSDLoRATrainer', FakeTrainer)

    adapter = IndependentCropAdapter(crop_name='tomato', device='cpu')
    adapter.initialize_engine(class_names=['healthy'])

    events = []
    result = adapter.train_increment(
        train_loader=[{'images': torch.zeros(1, 3, 224, 224), 'labels': torch.zeros(1, dtype=torch.long)}],
        progress_callback=events.append,
    )

    assert result['status'] == 'trained'
    assert any('batch' in event for event in events)
    assert any('epoch_done' in event for event in events)

