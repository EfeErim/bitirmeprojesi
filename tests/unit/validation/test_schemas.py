#!/usr/bin/env python3
"""
Tests for configuration schemas.
"""

import pytest
from src.core.schemas import (
    router_schema,
    ood_schema,
    monitoring_schema,
    security_schema
)
from src.core.configuration_validator import config_validator

class TestSchemas:
    """Test suite for configuration schemas."""

    def test_router_schema_structure(self):
        """Test router schema has required structure."""
        schema = router_schema()
        
        assert "title" in schema
        assert "type" in schema
        assert "properties" in schema
        assert "router" in schema["properties"]

    def test_ood_schema_structure(self):
        """Test OOD schema has required structure."""
        schema = ood_schema()
        
        assert "title" in schema
        assert "type" in schema
        assert "properties" in schema
        assert "ood" in schema["properties"]

    def test_monitoring_schema_structure(self):
        """Test monitoring schema has required structure."""
        schema = monitoring_schema()
        
        assert "title" in schema
        assert "type" in schema
        assert "properties" in schema
        assert "monitoring" in schema["properties"]

    def test_security_schema_structure(self):
        """Test security schema has required structure."""
        schema = security_schema()
        
        assert "title" in schema
        assert "type" in schema
        assert "properties" in schema
        assert "security" in schema["properties"]

    def test_schema_registration(self):
        """Test that schemas can be registered with validator."""
        # Clear existing schemas
        config_validator._schemas.clear()
        
        # Register each schema
        config_validator.register_schema("router", router_schema())
        config_validator.register_schema("ood", ood_schema())
        config_validator.register_schema("monitoring", monitoring_schema())
        config_validator.register_schema("security", security_schema())
        
        # Check they're registered
        assert "router" in config_validator._schemas
        assert "ood" in config_validator._schemas
        assert "monitoring" in config_validator._schemas
        assert "security" in config_validator._schemas

    def test_valid_router_config(self):
        """Test validation of valid router configuration."""
        config_validator._schemas.clear()
        config_validator.register_schema("router", router_schema())
        
        valid_config = {
            "router": {
                "enabled": True,
                "type": "enhanced",
                "strategy": "vlm_based",
                "crop_mapping": {
                    "tomato": {
                        "parts": ["leaf", "fruit"],
                        "model_path": "models/tomato"
                    }
                },
                "fallback_strategy": "best_available",
                "confidence_threshold": 0.7
            }
        }
        
        result = config_validator.validate("router", valid_config)
        assert result["router"]["enabled"] is True

    def test_invalid_router_config(self):
        """Test validation rejects invalid router configuration."""
        config_validator._schemas.clear()
        config_validator.register_schema("router", router_schema())
        
        invalid_config = {
            "router": {
                "type": "invalid_type",  # Should be one of: basic, enhanced, vlm
                "confidence_threshold": 1.5  # Should be between 0 and 1
            }
        }
        
        with pytest.raises(Exception):  # Should raise ConfigurationError
            config_validator.validate("router", invalid_config)

    def test_valid_router_policy_execution_config(self):
        """Test router schema accepts valid policy_graph.execution configuration."""
        config_validator._schemas.clear()
        config_validator.register_schema("router", router_schema())

        valid_config = {
            "router": {
                "enabled": True,
                "type": "vlm",
                "vlm": {
                    "enabled": True,
                    "policy_graph": {
                        "execution": {
                            "sam3_stage_order": [
                                "roi_filter",
                                "roi_classification",
                                "open_set_gate",
                                "postprocess"
                            ],
                            "confidence_threshold_multiplier": 1.15,
                            "confidence_threshold_min": 0.0,
                            "confidence_threshold_max": 1.0
                        }
                    }
                }
            }
        }

        result = config_validator.validate("router", valid_config)
        assert result["router"]["vlm"]["policy_graph"]["execution"]["confidence_threshold_multiplier"] == 1.15

    def test_invalid_router_policy_execution_stage_order(self):
        """Test router schema rejects invalid sam3 stage names in execution config."""
        config_validator._schemas.clear()
        config_validator.register_schema("router", router_schema())

        invalid_config = {
            "router": {
                "enabled": True,
                "type": "vlm",
                "vlm": {
                    "enabled": True,
                    "policy_graph": {
                        "execution": {
                            "sam3_stage_order": ["roi_filter", "invalid_stage"]
                        }
                    }
                }
            }
        }

        with pytest.raises(Exception):
            config_validator.validate("router", invalid_config)

    def test_invalid_router_policy_execution_threshold_bounds(self):
        """Test router schema rejects out-of-range threshold clamp values."""
        config_validator._schemas.clear()
        config_validator.register_schema("router", router_schema())

        invalid_config = {
            "router": {
                "enabled": True,
                "type": "vlm",
                "vlm": {
                    "enabled": True,
                    "policy_graph": {
                        "execution": {
                            "confidence_threshold_min": -0.1,
                            "confidence_threshold_max": 1.2
                        }
                    }
                }
            }
        }

        with pytest.raises(Exception):
            config_validator.validate("router", invalid_config)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])