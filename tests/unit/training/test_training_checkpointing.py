import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer


class IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def _build_minimal_trainer() -> tuple[ContinualSDLoRATrainer, nn.Parameter]:
    cfg = ContinualSDLoRAConfig(
        backbone_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        target_modules_strategy="all_linear_transformer",
        fusion_layers=[2],
        fusion_output_dim=4,
        device="cpu",
        num_epochs=2,
    )
    trainer = ContinualSDLoRATrainer(cfg)
    trainer.class_to_idx = {"healthy": 0}
    trainer.adapter_model = IdentityModule()
    trainer.classifier = IdentityModule()
    trainer.fusion = IdentityModule()
    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]
    return trainer, trainable


def _make_loader(batch_count: int) -> list[dict[str, torch.Tensor]]:
    return [
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)}
        for _ in range(batch_count)
    ]


def test_training_checkpoint_roundtrip_preserves_progress_and_history(tmp_path):
    trainer, _ = _build_minimal_trainer()
    train_loader = _make_loader(2)
    history = trainer.train_increment(train_loader, num_epochs=1)

    ckpt_dir = trainer.save_training_checkpoint(
        str(tmp_path),
        progress_state={"epoch": 1, "batch": 2, "global_step": int(history["global_step"]), "elapsed_sec": 3.0},
        history=history,
        run_id="run_ckpt",
    )
    assert (ckpt_dir / "training_checkpoint.pt").exists()
    assert (ckpt_dir / "checkpoint_meta.json").exists()

    resumed, _ = _build_minimal_trainer()
    payload = resumed.load_training_checkpoint(str(ckpt_dir))
    assert payload["run_id"] == "run_ckpt"
    assert payload["progress_state"]["epoch"] == 1
    assert resumed.current_epoch == 1


def test_resume_state_continues_global_step_and_marks_resume_start_epoch():
    trainer, _ = _build_minimal_trainer()
    train_loader = _make_loader(2)
    resume_state = {
        "progress_state": {"epoch": 1, "global_step": 7, "elapsed_sec": 2.5},
        "history_snapshot": {
            "train_loss": [0.3],
            "val_loss": [],
            "val_accuracy": [],
            "macro_f1": [],
            "weighted_f1": [],
            "balanced_accuracy": [],
            "generalization_gap": [],
            "per_class_accuracy": [],
            "worst_classes": [],
        },
    }
    history = trainer.train_increment(train_loader, num_epochs=2, resume_state=resume_state)
    assert history["global_step"] >= 9
    assert history["resume_start_epoch"] == 1
    assert len(history["train_loss"]) >= 2

