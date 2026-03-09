from src.core.config_manager import ConfigurationManager
from src.training.services.config_surface import extract_continual_training_config


def test_colab_training_surface_normalizes_runtime_aliases():
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    colab_training = cfg["colab"]["training"]
    assert colab_training["checkpoint_every_n_steps"] == 200
    assert colab_training["checkpoint_interval"] == 200
    assert colab_training["checkpoint_on_exception"] is True
    assert colab_training["stdout_progress_batch_interval"] == 50
    assert colab_training["stdout_progress_min_interval_sec"] == 15.0


def test_training_continual_surface_exposes_reliability_defaults():
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    continual = cfg["training"]["continual"]
    assert continual["ood"]["ber_enabled"] is False
    assert continual["ood"]["ber_lambda_old"] == 0.1
    assert continual["ood"]["ber_lambda_new"] == 0.1
    assert continual["seed"] == 42
    assert continual["deterministic"] is False
    assert continual["optimization"]["grad_accumulation_steps"] == 1
    assert continual["optimization"]["scheduler"]["name"] == "none"
    assert continual["evaluation"]["best_metric"] == "val_loss"
    assert continual["evaluation"]["require_ood_for_gate"] is True
    assert continual["evaluation"]["ood_fallback_strategy"] == "held_out_benchmark"
    assert continual["evaluation"]["ood_benchmark_auto_run"] is True
    assert continual["evaluation"]["ood_benchmark_min_classes"] == 3
    assert continual["data"]["loader_error_policy"] == "tolerant"
    assert continual["data"]["validate_images_on_init"] is False


def test_extract_continual_training_config_normalizes_root_and_legacy_shapes():
    root_payload = {
        "training": {
            "continual": {
                "backbone": {"model_name": "demo-model"},
                "evaluation": {"best_metric": "macro_f1"},
            }
        }
    }
    legacy_payload = {
        "model_name": "legacy-model",
        "lora_r": 4,
        "fusion_output_dim": 256,
        "device": "cpu",
    }

    root_normalized = extract_continual_training_config(root_payload, model_name="ignored", device="cuda")
    legacy_normalized = extract_continual_training_config(legacy_payload, model_name="ignored", device="cuda")

    assert root_normalized["backbone"]["model_name"] == "demo-model"
    assert root_normalized["evaluation"]["best_metric"] == "macro_f1"
    assert root_normalized["optimization"]["scheduler"]["name"] == "none"
    assert legacy_normalized["backbone"]["model_name"] == "legacy-model"
    assert legacy_normalized["adapter"]["lora_r"] == 4
    assert legacy_normalized["fusion"]["output_dim"] == 256
    assert legacy_normalized["device"] == "cpu"


def test_extract_continual_training_config_normalizes_ber_fields():
    payload = {
        "training": {
            "continual": {
                "ood": {
                    "ber_enabled": True,
                    "ber_lambda_old": "0.05",
                    "ber_lambda_new": 0.2,
                }
            }
        }
    }

    normalized = extract_continual_training_config(payload)

    assert normalized["ood"]["ber_enabled"] is True
    assert normalized["ood"]["ber_lambda_old"] == 0.05
    assert normalized["ood"]["ber_lambda_new"] == 0.2
