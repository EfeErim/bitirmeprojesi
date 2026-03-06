from src.core.config_manager import ConfigurationManager


def test_colab_training_surface_normalizes_runtime_aliases():
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    colab_training = cfg["colab"]["training"]
    assert colab_training["checkpoint_every_n_steps"] == 200
    assert colab_training["checkpoint_interval"] == 200
    assert colab_training["checkpoint_on_exception"] is True
    assert colab_training["stdout_progress_batch_interval"] == 50


def test_training_continual_surface_exposes_reliability_defaults():
    cfg = ConfigurationManager(config_dir="config", environment="colab").load_all_configs()

    continual = cfg["training"]["continual"]
    assert continual["seed"] == 42
    assert continual["deterministic"] is False
    assert continual["optimization"]["grad_accumulation_steps"] == 1
    assert continual["optimization"]["scheduler"]["name"] == "none"
    assert continual["evaluation"]["best_metric"] == "val_loss"
    assert continual["data"]["loader_error_policy"] == "tolerant"
