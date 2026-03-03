import pytest
import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.block = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        ])
        self.router_head = nn.Linear(8, 2)
        self.classifier = nn.Linear(8, 2)
        self.config = type('Cfg', (), {'hidden_size': 8})()

    def forward(self, images, output_hidden_states=False):
        batch = images.shape[0]
        hidden = torch.randn(batch, 4, 8, device=images.device)
        if output_hidden_states:
            return type('Output', (), {'hidden_states': [hidden] * 12})()
        return type('Output', (), {'last_hidden_state': hidden})()



def test_config_from_training_config_accepts_v6_contract():
    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            'backbone': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'},
            'quantization': {'mode': 'int8_hybrid', 'strict_backend': False, 'allow_cpu_fallback': True},
            'adapter': {'target_modules_strategy': 'all_linear_transformer', 'lora_r': 4, 'lora_alpha': 8},
            'fusion': {'layers': [2, 5, 8, 11]},
            'device': 'cpu',
        }
    )
    assert cfg.backbone_model_name == 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    assert cfg.quantization_mode == 'int8_hybrid'
    assert cfg.target_modules_strategy == 'all_linear_transformer'


def test_target_resolver_excludes_classifier_and_router_heads():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
        quantization_mode='int8_hybrid',
        target_modules_strategy='all_linear_transformer',
        fusion_layers=[2, 5, 8, 11],
        device='cpu',
    )
    trainer = ContinualSDLoRATrainer(cfg)
    names = trainer.resolve_target_modules(DummyBackbone())
    assert names
    assert all('classifier' not in n.lower() for n in names)
    assert all('router' not in n.lower() for n in names)


def test_add_classes_expands_classifier_shape():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
        quantization_mode='int8_hybrid',
        target_modules_strategy='all_linear_transformer',
        fusion_layers=[2],
        fusion_output_dim=8,
        device='cpu',
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.classifier = nn.Linear(8, 1)
    trainer.class_to_idx = {'healthy': 0}

    updated = trainer.add_classes(['disease_a', 'disease_b'])

    assert set(updated.keys()) == {'healthy', 'disease_a', 'disease_b'}
    assert trainer.classifier.out_features == 3


def test_predict_payload_contains_v6_ood_keys():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
        quantization_mode='int8_hybrid',
        target_modules_strategy='all_linear_transformer',
        fusion_layers=[2],
        fusion_output_dim=4,
        device='cpu',
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {'healthy': 0}

    class DummyModule(nn.Module):
        def forward(self, *args, **kwargs):
            return args[0]

    trainer.adapter_model = DummyModule()
    trainer.fusion = DummyModule()
    trainer.classifier = nn.Linear(4, 1)

    def fake_encode(images):
        return torch.zeros(images.shape[0], 4)

    trainer.encode = fake_encode  # type: ignore[assignment]
    trainer.ood_detector.score = lambda features, logits, predicted_labels=None: {
        'mahalanobis_z': torch.tensor([0.1]),
        'energy_z': torch.tensor([0.2]),
        'ensemble_score': torch.tensor([0.15]),
        'class_threshold': torch.tensor([0.8]),
        'is_ood': torch.tensor([False]),
        'calibration_version': torch.tensor([3]),
    }

    result = trainer.predict_with_ood(torch.zeros(1, 3, 224, 224))

    assert 'ood_analysis' in result
    assert {'ensemble_score', 'class_threshold', 'is_ood', 'calibration_version'} <= set(result['ood_analysis'].keys())


def test_warns_when_peft_is_missing(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    cfg = ContinualSDLoRAConfig(
        backbone_model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
        quantization_mode='int8_hybrid',
        target_modules_strategy='all_linear_transformer',
        fusion_layers=[2],
        fusion_output_dim=8,
        device='cpu',
    )
    trainer = ContinualSDLoRATrainer(cfg)
    backbone = DummyBackbone()

    monkeypatch.setattr(continual_module, 'LoraConfig', None)

    with pytest.warns(RuntimeWarning, match='peft is not installed'):
        wrapped = trainer._apply_lora(backbone, ['transformer.block.0.0'])

    assert wrapped is backbone
    assert trainer._adapter_wrapped is False


def test_train_increment_emits_progress_callback_events():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name='facebook/dinov3-vitl16-pretrain-lvd1689m',
        quantization_mode='int8_hybrid',
        target_modules_strategy='all_linear_transformer',
        fusion_layers=[2],
        fusion_output_dim=4,
        device='cpu',
        num_epochs=1,
    )
    trainer = ContinualSDLoRATrainer(cfg)

    class DummyModule(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    trainer.adapter_model = DummyModule()
    trainer.classifier = DummyModule()
    trainer.fusion = DummyModule()

    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]

    train_loader = [
        {'images': torch.zeros(1, 3, 8, 8), 'labels': torch.zeros(1, dtype=torch.long)},
        {'images': torch.zeros(1, 3, 8, 8), 'labels': torch.zeros(1, dtype=torch.long)},
    ]

    events = []
    history = trainer.train_increment(train_loader, num_epochs=1, progress_callback=events.append)

    assert len(history['train_loss']) == 1
    batch_events = [event for event in events if 'batch' in event]
    epoch_events = [event for event in events if 'epoch_done' in event]
    assert len(batch_events) == 2
    assert len(epoch_events) == 1
    assert batch_events[0]['epoch'] == 1
    assert batch_events[0]['total_batches'] == 2
    assert 'batch_loss' in batch_events[0]
    assert 'epoch_progress' in batch_events[0]
    assert 'epoch_loss' in epoch_events[0]


def test_trainer_end_to_end_surface_with_dummy_backbone(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    def fake_load(*_args, **_kwargs):
        return DummyBackbone()

    monkeypatch.setattr(continual_module, 'load_hybrid_int8_backbone', fake_load)
    monkeypatch.setattr(continual_module, 'AutoModel', object())
    monkeypatch.setattr(ContinualSDLoRATrainer, '_apply_lora', lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            'backbone': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'},
            'quantization': {'mode': 'int8_hybrid', 'strict_backend': True, 'allow_cpu_fallback': True},
            'adapter': {'target_modules_strategy': 'all_linear_transformer', 'lora_r': 4, 'lora_alpha': 8},
            'fusion': {'layers': [2, 5, 8, 11], 'output_dim': 8},
            'ood': {'threshold_factor': 2.0},
            'device': 'cpu',
            'num_epochs': 1,
        }
    )

    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={'healthy': 0})
    trainer.add_classes(['disease_a'])

    train_loader = [
        {'images': torch.zeros(2, 3, 8, 8), 'labels': torch.tensor([0, 1], dtype=torch.long)},
        {'images': torch.zeros(2, 3, 8, 8), 'labels': torch.tensor([1, 0], dtype=torch.long)},
    ]
    history = trainer.train_increment(train_loader, num_epochs=1)
    assert len(history['train_loss']) == 1

    cal = trainer.calibrate_ood(train_loader)
    assert int(cal['num_classes']) >= 1

    pred = trainer.predict_with_ood(torch.zeros(1, 3, 8, 8))
    assert pred['status'] == 'success'
    assert {'ensemble_score', 'class_threshold', 'is_ood', 'calibration_version'} <= set(pred['ood_analysis'].keys())

