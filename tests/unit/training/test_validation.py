import pytest
import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
from src.training.validation import evaluate_model, evaluate_model_with_predictions


class IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def test_evaluate_model_reports_expected_metrics():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = nn.Linear(4, 2)
    trainer.fusion = IdentityModule()
    trainer.forward_logits = lambda images: torch.tensor([[2.0, 0.0], [1.0, 0.0]])  # type: ignore[assignment]

    report = evaluate_model(
        trainer,
        [{"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}],
    )

    assert report is not None
    assert report.val_loss >= 0.0
    assert report.val_accuracy == 0.5
    assert report.macro_precision == pytest.approx(0.25)
    assert report.macro_recall == pytest.approx(0.5)
    assert "healthy" in report.per_class_accuracy
    assert len(report.worst_classes) >= 1


def test_evaluate_model_with_predictions_returns_labels_and_predictions():
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0, "disease_a": 1}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = nn.Linear(4, 2)
    trainer.fusion = IdentityModule()
    trainer.forward_logits = lambda images: torch.tensor([[2.0, 0.0], [1.0, 0.0]])  # type: ignore[assignment]

    result = evaluate_model_with_predictions(
        trainer,
        [{"images": torch.zeros(2, 3, 8, 8), "labels": torch.tensor([0, 1], dtype=torch.long)}],
    )

    assert result is not None
    report, y_true, y_pred = result
    assert report.val_accuracy == 0.5
    assert y_true == [0, 1]
    assert y_pred == [0, 0]
