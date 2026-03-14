import json

import pytest

from src.core.config_manager import ConfigurationManager
from src.core.config_migrations import CURRENT_CONFIG_SCHEMA_VERSION, is_versioned_config_surface_payload


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manager_migrates_unversioned_top_level_ood_aliases(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_dir / "base.json",
        {
            "router": {"enabled": True},
            "training": {
                "continual": {
                    "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                    "adapter": {"target_modules_strategy": "all_linear_transformer"},
                    "fusion": {"layers": [2, 5, 8, 11]},
                }
            },
            "ood": {
                "enabled": True,
                "threshold_factor": "2.5",
                "primary_score_method": "KNN",
            },
        },
    )

    merged = ConfigurationManager(config_dir=str(config_dir)).load_all_configs()

    assert merged["config_schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION
    assert merged["training"]["continual"]["ood"]["threshold_factor"] == 2.5
    assert merged["training"]["continual"]["ood"]["primary_score_method"] == "knn"
    assert merged["ood"]["threshold_factor"] == 2.5
    assert merged["ood"]["primary_score_method"] == "knn"


def test_manager_migrates_environment_legacy_aliases_before_merge(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_dir / "base.json",
        {
            "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
            "router": {"enabled": True},
            "training": {
                "continual": {
                    "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                    "adapter": {"target_modules_strategy": "all_linear_transformer"},
                    "fusion": {"layers": [2, 5, 8, 11]},
                    "ood": {"threshold_factor": 2.0, "primary_score_method": "auto"},
                }
            },
            "ood": {"enabled": True, "threshold_factor": 2.0, "primary_score_method": "auto"},
        },
    )
    _write_json(
        config_dir / "legacy.json",
        {
            "ood": {
                "threshold_factor": 3.0,
                "primary_score_method": "energy",
            },
            "colab": {"training": {"checkpoint_interval": 321}},
        },
    )

    merged = ConfigurationManager(config_dir=str(config_dir), environment="legacy").load_all_configs()

    assert merged["config_schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION
    assert merged["training"]["continual"]["ood"]["threshold_factor"] == 3.0
    assert merged["training"]["continual"]["ood"]["primary_score_method"] == "energy"
    assert merged["ood"]["threshold_factor"] == 3.0
    assert merged["ood"]["primary_score_method"] == "energy"
    assert merged["colab"]["training"]["checkpoint_every_n_steps"] == 321
    assert merged["colab"]["training"]["checkpoint_interval"] == 321


def test_validate_merged_config_requires_current_schema_version(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_dir / "base.json",
        {
            "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION,
            "router": {"enabled": True},
            "training": {
                "continual": {
                    "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                    "adapter": {"target_modules_strategy": "all_linear_transformer"},
                    "fusion": {"layers": [2, 5, 8, 11]},
                }
            },
            "ood": {"enabled": True},
        },
    )

    manager = ConfigurationManager(config_dir=str(config_dir))
    assert manager.validate_merged_config() is True


def test_manager_rejects_future_config_schema_version(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_dir / "base.json",
        {
            "config_schema_version": CURRENT_CONFIG_SCHEMA_VERSION + 1,
            "router": {"enabled": True},
            "training": {
                "continual": {
                    "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                    "adapter": {"target_modules_strategy": "all_linear_transformer"},
                    "fusion": {"layers": [2, 5, 8, 11]},
                }
            },
            "ood": {"enabled": True},
        },
    )

    with pytest.raises(ValueError, match="Unsupported config_schema_version"):
        ConfigurationManager(config_dir=str(config_dir)).load_all_configs()


def test_load_config_file_migrates_versioned_surface(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config_dir / "legacy.json",
        {
            "router": {"enabled": True},
            "training": {
                "continual": {
                    "backbone": {"model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
                    "adapter": {"target_modules_strategy": "all_linear_transformer"},
                    "fusion": {"layers": [2, 5, 8, 11]},
                }
            },
            "ood": {"enabled": True, "threshold_factor": 2.75},
        },
    )

    payload = ConfigurationManager(config_dir=str(config_dir)).load_config_file("legacy.json")

    assert payload["config_schema_version"] == CURRENT_CONFIG_SCHEMA_VERSION
    assert payload["training"]["continual"]["ood"]["threshold_factor"] == 2.75


def test_load_config_file_leaves_non_surface_payload_untouched(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_dir / "plant_taxonomy.json", {"crops": {"tomato": ["leaf"]}})

    payload = ConfigurationManager(config_dir=str(config_dir)).load_config_file("plant_taxonomy.json")

    assert payload == {"crops": {"tomato": ["leaf"]}}


def test_is_versioned_config_surface_payload_matches_supported_sections():
    assert is_versioned_config_surface_payload({"training": {}}) is True
    assert is_versioned_config_surface_payload({"colab": {}}) is True
    assert is_versioned_config_surface_payload({"crops": {"tomato": ["leaf"]}}) is False
