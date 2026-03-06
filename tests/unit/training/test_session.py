import torch
import torch.nn as nn

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer
from src.training.session import ContinualTrainingSession


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
    trainer.classifier = nn.Linear(4, 2)
    trainer.fusion = IdentityModule()
    trainable = nn.Parameter(torch.tensor([1.0], requires_grad=True))
    trainer.optimizer = torch.optim.SGD([trainable], lr=0.1)
    trainer.training_step = lambda _batch: (trainable ** 2).sum()  # type: ignore[assignment]
    trainer.forward_logits = lambda images: torch.zeros(images.shape[0], 2)  # type: ignore[assignment]
    return trainer, trainable


def _make_loader(batch_count: int) -> list[dict[str, torch.Tensor]]:
    return [
        {"images": torch.zeros(1, 3, 8, 8), "labels": torch.zeros(1, dtype=torch.long)}
        for _ in range(batch_count)
    ]


def test_session_run_emits_observer_events_and_history():
    trainer, _ = _build_minimal_trainer()
    events = []
    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        1,
        val_loader=_make_loader(1),
        observers=[events.append],
        run_id="run_1",
    )

    history = session.run()

    assert len(history.train_loss) == 1
    assert len(history.val_loss) == 1
    assert history.global_step == 2
    event_types = [event["event_type"] for event in events]
    assert event_types.count("batch_end") == 2
    assert event_types.count("epoch_end") == 1


def test_session_resume_continues_global_step_and_resume_epoch():
    trainer, _ = _build_minimal_trainer()
    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        2,
        resume_state={
            "progress_state": {"epoch": 1, "global_step": 7, "elapsed_sec": 2.5},
            "history": {
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
        },
    )

    history = session.run()

    assert history.global_step >= 9
    assert history.resume_start_epoch == 1
    assert len(history.train_loss) >= 2


def test_session_honors_stop_policy():
    trainer, _ = _build_minimal_trainer()
    stop_flag = {"value": False}
    events = []

    def observer(event):
        events.append(event)
        if event["event_type"] == "batch_end":
            stop_flag["value"] = True

    session = ContinualTrainingSession(
        trainer,
        _make_loader(2),
        2,
        observers=[observer],
        stop_policy=lambda: stop_flag["value"],
    )

    history = session.run()

    assert history.stopped_early is True
    event_types = [event["event_type"] for event in events]
    assert event_types.count("batch_end") == 1
    assert "stop_requested" in event_types
