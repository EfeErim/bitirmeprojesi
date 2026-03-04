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
        self.config = type("Cfg", (), {"hidden_size": 8})()

    def forward(self, images, output_hidden_states=False):
        batch = images.shape[0]
        hidden = torch.randn(batch, 4, 8, device=images.device)
        if output_hidden_states:
            return type("Output", (), {"hidden_states": [hidden] * 12})()
        return type("Output", (), {"last_hidden_state": hidden})()



def test_config_from_training_config_accepts_v6_contract():
    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {"target_modules_strategy": "all_linear_transformer", "lora_r": 4, "lora_alpha": 8},
            "fusion": {"layers": [2, 5, 8, 11]},
            "device": "cpu",
        }
    )
    assert cfg.backbone_model_name == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert cfg.target_modules_strategy == "all_linear_transformer"



def test_target_resolver_excludes_classifier_and_router_heads():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2, 5, 8, 11],
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    names = trainer.resolve_target_modules(DummyBackbone())
    assert names
    assert all("classifier" not in n.lower() for n in names)
    assert all("router" not in n.lower() for n in names)



def test_add_classes_expands_classifier_shape():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.classifier = nn.Linear(8, 1)
    trainer.class_to_idx = {"healthy": 0}

    updated = trainer.add_classes(["disease_a", "disease_b"])

    assert set(updated.keys()) == {"healthy", "disease_a", "disease_b"}
    assert trainer.classifier.out_features == 3



def test_predict_payload_contains_v6_ood_keys():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}

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
        "mahalanobis_z": torch.tensor([0.1]),
        "energy_z": torch.tensor([0.2]),
        "ensemble_score": torch.tensor([0.15]),
        "class_threshold": torch.tensor([0.8]),
        "is_ood": torch.tensor([False]),
        "calibration_version": torch.tensor([3]),
    }

    result = trainer.predict_with_ood(torch.zeros(1, 3, 224, 224))

    assert "ood_analysis" in result
    assert {"ensemble_score", "class_threshold", "is_ood", "calibration_version"} <= set(result["ood_analysis"].keys())



def test_raises_when_peft_is_missing(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    backbone = DummyBackbone()

    monkeypatch.setattr(continual_module, "LoraConfig", None)

    with pytest.raises(RuntimeError, match="peft is required for SD-LoRA adapter wrapping"):
        trainer._apply_lora(backbone, ["transformer.block.0.0"])



def test_apply_lora_avoids_low_cpu_mem_usage_meta_tensors(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    call_kwargs = {}

    def fake_get_peft_model(model, _cfg, **kwargs):
        call_kwargs.update(kwargs)
        return model

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "LoraConfig", FakeLoraConfig)
    monkeypatch.setattr(continual_module, "get_peft_model", fake_get_peft_model)

    model = DummyBackbone()
    wrapped = trainer._apply_lora(model, ["transformer.block.0.0"])

    assert wrapped is model
    assert call_kwargs.get("low_cpu_mem_usage") is False


def test_apply_lora_without_low_cpu_mem_usage_kwarg_support(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    called = {"count": 0}

    def fake_get_peft_model(model, _cfg):
        called["count"] += 1
        return model

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "LoraConfig", FakeLoraConfig)
    monkeypatch.setattr(continual_module, "get_peft_model", fake_get_peft_model)

    model = DummyBackbone()
    wrapped = trainer._apply_lora(model, ["transformer.block.0.0"])

    assert wrapped is model
    assert called["count"] == 1


def test_train_increment_emits_progress_callback_events():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
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
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)},
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)},
    ]

    events = []
    history = trainer.train_increment(train_loader, num_epochs=1, progress_callback=events.append)

    assert len(history["train_loss"]) == 1
    assert history["stopped_early"] is False
    batch_events = [event for event in events if "batch" in event]
    epoch_events = [event for event in events if "epoch_done" in event]
    assert len(batch_events) == 2
    assert len(epoch_events) == 1
    assert {"global_step", "lr", "grad_norm", "step_time_sec", "samples_per_sec", "eta_sec"} <= set(batch_events[0].keys())


def test_train_increment_reports_validation_metrics():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        num_epochs=1,
    )
    trainer = ContinualSDLoRATrainer(cfg)

    class DummyModule(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    trainer.adapter_model = DummyModule()
    trainer.classifier = nn.Linear(4, 2)
    trainer.fusion = DummyModule()

    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.05)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]
    trainer.forward_logits = lambda images: torch.zeros(images.shape[0], 2)  # type: ignore[assignment]

    train_loader = [
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)},
    ]
    val_loader = [
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.zeros(2, dtype=torch.long)},
    ]

    events = []
    history = trainer.train_increment(train_loader, num_epochs=1, val_loader=val_loader, progress_callback=events.append)

    assert len(history["val_loss"]) == 1
    assert len(history["val_accuracy"]) == 1
    assert len(history["macro_f1"]) == 1
    assert len(history["weighted_f1"]) == 1
    assert len(history["balanced_accuracy"]) == 1
    assert len(history["generalization_gap"]) == 1
    assert len(history["per_class_accuracy"]) == 1
    assert len(history["worst_classes"]) == 1
    epoch_events = [event for event in events if "epoch_done" in event]
    assert len(epoch_events) == 1
    assert "val_loss" in epoch_events[0]
    assert "val_accuracy" in epoch_events[0]
    assert "macro_f1" in epoch_events[0]
    assert "weighted_f1" in epoch_events[0]
    assert "balanced_accuracy" in epoch_events[0]
    assert "per_class_accuracy" in epoch_events[0]
    assert "worst_classes" in epoch_events[0]
    assert "generalization_gap" in epoch_events[0]


def test_train_increment_honors_should_stop_signal():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        num_epochs=2,
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
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)},
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)},
    ]

    stop_flag = {"value": False}
    events = []

    def callback(event):
        events.append(event)
        if "batch" in event:
            stop_flag["value"] = True

    history = trainer.train_increment(
        train_loader,
        num_epochs=2,
        progress_callback=callback,
        should_stop=lambda: stop_flag["value"],
    )

    assert history["stopped_early"] is True
    batch_events = [event for event in events if "batch" in event and not event.get("stop_requested")]
    stop_events = [event for event in events if event.get("stop_requested")]
    assert len(batch_events) == 1
    assert len(stop_events) == 1



def test_initialize_engine_with_non_quantized_backbone(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {"target_modules_strategy": "all_linear_transformer", "lora_r": 4, "lora_alpha": 8},
            "fusion": {"layers": [2, 5, 8, 11], "output_dim": 8},
            "device": "cpu",
        }
    )

    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0})

    assert trainer.backbone is not None
    assert trainer.classifier is not None
    assert trainer.classifier.out_features == 1


def test_initialize_engine_raises_when_backbone_contains_meta_tensors(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class MetaBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 8, device="meta")
            self.config = type("Cfg", (), {"hidden_size": 8})()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return MetaBackbone()

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)

    with pytest.raises(RuntimeError, match="backbone contains meta tensors before device move"):
        trainer.initialize_engine(class_to_idx={"healthy": 0})


def test_initialize_engine_raises_when_adapter_contains_meta_tensors(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    class MetaAdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Linear(8, 8, device="meta")

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, _model, _targets: MetaAdapterModel())

    with pytest.raises(RuntimeError, match="adapter_model contains meta tensors before device move"):
        trainer.initialize_engine(class_to_idx={"healthy": 0})


def test_initialize_engine_allows_dispatch_managed_meta_adapter(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    class DispatchManagedMetaAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Linear(8, 8, device="meta")
            self.hf_device_map = {"": "cpu"}

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(
        ContinualSDLoRATrainer,
        "_apply_lora",
        lambda self, _model, _targets: DispatchManagedMetaAdapter(),
    )

    trainer.initialize_engine(class_to_idx={"healthy": 0})

    assert trainer.adapter_model is not None
    assert trainer.classifier is not None
    assert trainer.classifier.out_features == 1


def test_prepare_module_for_device_skips_dispatch_managed_not_implemented(monkeypatch):
    class NotImplementedOnTo(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.hf_device_map = {"": "cpu"}

        def to(self, *args, **kwargs):
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)

    module = NotImplementedOnTo()
    prepared = trainer._prepare_module_for_device(module, module_name="adapter_model")

    assert prepared is module


def test_prepare_module_for_device_re_raises_not_implemented_without_dispatch():
    class NotImplementedOnTo(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def to(self, *args, **kwargs):
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=8,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    module = NotImplementedOnTo()

    with pytest.raises(NotImplementedError, match="Cannot copy out of meta tensor; no data!"):
        trainer._prepare_module_for_device(module, module_name="adapter_model")



def test_trainer_end_to_end_surface_with_dummy_backbone(monkeypatch):
    from src.training import continual_sd_lora as continual_module

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return DummyBackbone()

    monkeypatch.setattr(continual_module, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ContinualSDLoRATrainer, "_apply_lora", lambda self, model, _targets: model)

    cfg = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
            "adapter": {"target_modules_strategy": "all_linear_transformer", "lora_r": 4, "lora_alpha": 8},
            "fusion": {"layers": [2, 5, 8, 11], "output_dim": 8},
            "ood": {"threshold_factor": 2.0},
            "device": "cpu",
            "num_epochs": 1,
        }
    )

    trainer = ContinualSDLoRATrainer(cfg)
    trainer.initialize_engine(class_to_idx={"healthy": 0})
    trainer.add_classes(["disease_a"])

    train_loader = [
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)},
        {"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([1, 0], dtype=torch.long)},
    ]
    history = trainer.train_increment(train_loader, num_epochs=1)
    assert len(history["train_loss"]) == 1

    cal = trainer.calibrate_ood(train_loader)
    assert int(cal["num_classes"]) >= 1

    pred = trainer.predict_with_ood(torch.zeros(1, 3, 8, 8))
    assert pred["status"] == "success"
    assert {"ensemble_score", "class_threshold", "is_ood", "calibration_version"} <= set(pred["ood_analysis"].keys())
