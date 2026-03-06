import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
from src.training.validation import evaluate_model


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
    assert "healthy" in report.per_class_accuracy
    assert len(report.worst_classes) >= 1
