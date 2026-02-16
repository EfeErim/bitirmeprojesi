#!/usr/bin/env python3
"""
Tests for the configuration management system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.core.config_manager import ConfigurationManager, ConfigurationError
from src.core.configuration_validator import ConfigurationError as ValidatorError

class TestConfigurationManager:
    """Test suite for ConfigurationManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()
        
        # Create base config
        base_config = {
            "version": "5.5.0",
            "description": "Test base configuration",
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "database": {
                "type": "postgresql",
                "host": "localhost"
            }
        }
        
        with open(self.config_dir / "base.json", 'w') as f:
            json.dump(base_config, f)
        
        # Create router config
        router_config = {
            "router": {
                "enabled": True,
                "type": "enhanced",
                "crop_mapping": {
                    "tomato": {
                        "parts": ["leaf", "fruit"],
                        "model_path": "models/tomato"
                    }
                }
            }
        }
        
        with open(self.config_dir / "router-config.json", 'w') as f:
            json.dump(router_config, f)
        
        # Create OOD config
        ood_config = {
            "ood": {
                "enabled": True,
                "method": "mahalanobis",
                "threshold": 0.95
            }
        }
        
        with open(self.config_dir / "ood-config.json", 'w') as f:
            json.dump(ood_config, f)
        
        # Create monitoring config
        monitoring_config = {
            "monitoring": {
                "enabled": True,
                "prometheus": {
                    "enabled": True,
                    "port": 9090
                }
            }
        }
        
        with open(self.config_dir / "monitoring-config.json", 'w') as f:
            json.dump(monitoring_config, f)
        
        # Create security config
        security_config = {
            "security": {
                "api_key_required": False,
                "rate_limit": {
                    "enabled": True,
                    "requests_per_minute": 100
                }
            }
        }
        
        with open(self.config_dir / "security-config.json", 'w') as f:
            json.dump(security_config, f)
        
        # Create environment configs
        dev_config = {
            "api": {
                "reload": True,
                "log_level": "debug"
            }
        }
        
        with open(self.config_dir / "development.json", 'w') as f:
            json.dump(dev_config, f)
        
        prod_config = {
            "api": {
                "reload": False,
                "log_level": "info"
            },
            "security": {
                "api_key_required": True
            }
        }
        
        with open(self.config_dir / "production.json", 'w') as f:
            json.dump(prod_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_base_config(self):
        """Test loading base configuration."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        config = manager.load_base_config()
        
        assert config["version"] == "5.5.0"
        assert config["api"]["host"] == "0.0.0.0"
        assert config["database"]["type"] == "postgresql"

    def test_load_config_file(self):
        """Test loading individual config file."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        config = manager.load_config_file("router-config.json", "router")
        
        assert "router" in config
        assert config["router"]["enabled"] is True
        assert config["router"]["type"] == "enhanced"

    def test_load_all_configs(self):
        """Test loading and merging all configurations."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        config = manager.load_all_configs()
        
        # Check base config is present
        assert "version" in config
        assert config["version"] == "5.5.0"
        
        # Check merged sections
        assert "router" in config
        assert config["router"]["enabled"] is True
        
        assert "ood" in config
        assert config["ood"]["enabled"] is True
        
        assert "monitoring" in config
        assert config["monitoring"]["enabled"] is True
        
        assert "security" in config
        assert config["security"]["api_key_required"] is False

    def test_get_config_with_dot_notation(self):
        """Test getting config values with dot notation."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        manager.load_all_configs()
        
        assert manager.get_config("api.host") == "0.0.0.0"
        assert manager.get_config("api.port") == 8000
        assert manager.get_config("router.enabled") is True
        assert manager.get_config("nonexistent.key", "default") == "default"

    def test_environment_config_loading(self):
        """Test loading environment-specific configuration."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        dev_config = manager.get_environment_config("development")
        
        assert "api" in dev_config
        assert dev_config["api"]["reload"] is True
        
        prod_config = manager.get_environment_config("production")
        assert "api" in prod_config
        assert prod_config["api"]["reload"] is False

    def test_validation_schemas_registered(self):
        """Test that all schemas are registered."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        # Schemas should be registered during initialization
        # This test just ensures no exception is raised
        assert manager is not None

    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        # Create invalid JSON file
        invalid_path = self.config_dir / "invalid.json"
        with open(invalid_path, 'w') as f:
            f.write("{ invalid json")
        
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ValidatorError):
            manager.load_config_file("invalid.json", "router")

    def test_missing_base_config(self):
        """Test handling of missing base configuration."""
        # Remove base config
        (self.config_dir / "base.json").unlink()
        
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ConfigurationError):
            manager.load_base_config()

    def test_reload_config(self):
        """Test configuration reloading."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        # Initial load
        config1 = manager.load_all_configs()
        
        # Reload
        config2 = manager.reload_config()
        
        # Should have same structure
        assert config1.keys() == config2.keys()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])