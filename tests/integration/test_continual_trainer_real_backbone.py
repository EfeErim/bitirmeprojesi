import os

import pytest
import torch

from src.training.continual_sd_lora import ContinualSDLoRAConfig, ContinualSDLoRATrainer


def _resolve_image_size(backbone) -> int:
    image_size = getattr(getattr(backbone, "config", None), "image_size", 224)
    if isinstance(image_size, (list, tuple)):
        image_size = image_size[0] if image_size else 224
    try:
        resolved = int(image_size)
    except Exception:
        resolved = 224
    return max(1, resolved)


@pytest.mark.integration
@pytest.mark.heavy_model
def test_continual_trainer_smoke_with_real_hf_backbone():
    model_name = os.getenv("AADS_HEAVY_BACKBONE_MODEL", "hf-internal-testing/tiny-random-ViTModel")

    config = ContinualSDLoRAConfig.from_training_config(
        {
            "backbone": {"model_name": model_name},
            "quantization": {
                "mode": "int8_hybrid",
                "strict_backend": False,
                "allow_cpu_fallback": True,
            },
            "adapter": {
                "target_modules_strategy": "all_linear_transformer",
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
            },
            "fusion": {"layers": [2, 5, 8, 11], "output_dim": 32, "dropout": 0.0},
            "ood": {"threshold_factor": 2.0},
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "num_epochs": 1,
            "batch_size": 2,
            "device": "cpu",
        }
    )
    trainer = ContinualSDLoRATrainer(config)

    try:
        trainer.initialize_engine(class_to_idx={"healthy": 0})
    except Exception as exc:
        pytest.skip(f"Unable to initialize real backbone '{model_name}': {exc}")

    trainer.add_classes(["disease_a"])
    side = _resolve_image_size(trainer.backbone)

    train_loader = [
        {"images": torch.randn(2, 3, side, side), "labels": torch.tensor([0, 1], dtype=torch.long)},
        {"images": torch.randn(2, 3, side, side), "labels": torch.tensor([1, 0], dtype=torch.long)},
    ]

    history = trainer.train_increment(train_loader, num_epochs=1)
    calibration = trainer.calibrate_ood(train_loader)
    prediction = trainer.predict_with_ood(torch.randn(1, 3, side, side))

    assert len(history["train_loss"]) == 1
    assert int(calibration["num_classes"]) >= 1
    assert prediction["status"] == "success"
    assert {"ensemble_score", "class_threshold", "is_ood", "calibration_version"} <= set(prediction["ood_analysis"].keys())
