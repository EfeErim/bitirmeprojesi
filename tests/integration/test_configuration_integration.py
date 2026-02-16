#!/usr/bin/env python3
"""
Integration tests for the complete configuration system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.core.config_manager import ConfigurationManager, get_config
from src.core.configuration_validator import ConfigurationError

class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()
        
        # Create comprehensive test configuration
        self._create_test_configs()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_test_configs(self):
        """Create all test configuration files."""
        # Base config
        base_config = {
            "version": "5.5.0-test",
            "description": "Test base configuration",
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            },
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432
            },
            "ml": {
                "device": "cuda",
                "model": {
                    "backbone": "dinov2_vits14"
                }
            }
        }
        
        with open(self.config_dir / "base.json", 'w') as f:
            json.dump(base_config, f)
        
        # Router config
        router_config = {
            "router": {
                "enabled": True,
                "type": "enhanced",
                "strategy": "vlm_based",
                "crop_mapping": {
                    "tomato": {
                        "parts": ["leaf", "fruit", "stem", "whole"],
                        "model_path": "models/tomato_adapter",
                        "priority": 1
                    },
                    "potato": {
                        "parts": ["leaf", "tuber", "stem", "whole"],
                        "model_path": "models/potato_adapter",
                        "priority": 2
                    }
                },
                "fallback_strategy": "best_available",
                "confidence_threshold": 0.7,
                "vlm": {
                    "enabled": True,
                    "confidence_threshold": 0.8,
                    "max_detections": 10
                },
                "caching": {
                    "enabled": True,
                    "max_size": 1000
                }
            }
        }
        
        with open(self.config_dir / "router-config.json", 'w') as f:
            json.dump(router_config, f)
        
        # OOD config
        ood_config = {
            "ood": {
                "enabled": True,
                "method": "mahalanobis",
                "threshold": 0.95,
                "confidence_level": 0.99,
                "prototype": {
                    "update_rate": 0.1,
                    "min_samples": 10
                },
                "mahalanobis": {
                    "eps": 1e-6,
                    "batch_size": 64
                }
            }
        }
        
        with open(self.config_dir / "ood-config.json", 'w') as f:
            json.dump(ood_config, f)
        
        # Monitoring config
        monitoring_config = {
            "monitoring": {
                "enabled": True,
                "prometheus": {
                    "enabled": True,
                    "port": 9090
                },
                "logging": {
                    "format": "json",
                    "rotate": True
                },
                "metrics": {
                    "enabled": True,
                    "track_requests": True
                }
            }
        }
        
        with open(self.config_dir / "monitoring-config.json", 'w') as f:
            json.dump(monitoring_config, f)
        
        # Security config
        security_config = {
            "security": {
                "api_key_required": False,
                "rate_limit": {
                    "enabled": True,
                    "requests_per_minute": 100
                },
                "cors": {
                    "allow_origins": ["*"]
                },
                "input_validation": {
                    "max_request_size_mb": 10
                }
            }
        }
        
        with open(self.config_dir / "security-config.json", 'w') as f:
            json.dump(security_config, f)
        
        # Environment configs
        dev_config = {
            "api": {
                "reload": True,
                "log_level": "debug"
            },
            "security": {
                "api_key_required": False
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
                "api_key_required": True,
                "rate_limit": {
                    "requests_per_minute": 50
                }
            }
        }
        
        with open(self.config_dir / "production.json", 'w') as f:
            json.dump(prod_config, f)

    def test_full_configuration_loading(self):
        """Test complete configuration loading and merging."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        config = manager.load_all_configs()
        
        # Verify base config
        assert config["version"] == "5.5.0-test"
        assert config["api"]["host"] == "0.0.0.0"
        assert config["database"]["type"] == "postgresql"
        
        # Verify merged router config
        assert "router" in config
        assert config["router"]["enabled"] is True
        assert config["router"]["type"] == "enhanced"
        assert "tomato" in config["router"]["crop_mapping"]
        assert config["router"]["crop_mapping"]["tomato"]["priority"] == 1
        
        # Verify merged OOD config
        assert "ood" in config
        assert config["ood"]["enabled"] is True
        assert config["ood"]["method"] == "mahalanobis"
        assert config["ood"]["prototype"]["update_rate"] == 0.1
        
        # Verify merged monitoring config
        assert "monitoring" in config
        assert config["monitoring"]["enabled"] is True
        assert config["monitoring"]["prometheus"]["port"] == 9090
        
        # Verify merged security config
        assert "security" in config
        assert config["security"]["api_key_required"] is False
        assert config["security"]["rate_limit"]["requests_per_minute"] == 100

    def test_environment_override(self):
        """Test environment-specific configuration overrides."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        # Load development config
        dev_config = manager.get_environment_config("development")
        assert dev_config["api"]["reload"] is True
        assert dev_config["security"]["api_key_required"] is False
        
        # Load production config
        prod_config = manager.get_environment_config("production")
        assert prod_config["api"]["reload"] is False
        assert prod_config["security"]["api_key_required"] is True
        assert prod_config["security"]["rate_limit"]["requests_per_minute"] == 50

    def test_configuration_get_with_dot_notation(self):
        """Test getting configuration values using dot notation."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        manager.load_all_configs()
        
        # Test various access patterns
        assert manager.get_config("api.host") == "0.0.0.0"
        assert manager.get_config("api.port") == 8000
        assert manager.get_config("router.enabled") is True
        assert manager.get_config("router.crop_mapping.tomato.priority") == 1
        assert manager.get_config("ood.threshold") == 0.95
        assert manager.get_config("monitoring.prometheus.port") == 9090
        assert manager.get_config("security.rate_limit.requests_per_minute") == 100
        
        # Test default values
        assert manager.get_config("nonexistent.key", "default") == "default"
        assert manager.get_config("router.nonexistent", 42) == 42

    def test_configuration_validation(self):
        """Test configuration validation."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        config = manager.load_all_configs()
        
        # Should pass validation
        assert manager.validate_merged_config() is True

    def test_configuration_used_by_pipeline(self):
        """Test that configuration can be used by pipeline components."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        config = manager.load_all_configs()
        
        # Simulate pipeline initialization
        from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
        
        try:
            pipeline = IndependentMultiCropPipeline(config, device='cpu')
            
            # Check configuration is properly accessible
            assert pipeline.config is not None
            assert "router" in pipeline.config
            assert "ood" in pipeline.config
            assert "monitoring" in pipeline.config
            assert "security" in pipeline.config
            
            # Test configuration values
            assert pipeline.config["router"]["enabled"] is True
            assert pipeline.config["ood"]["method"] == "mahalanobis"
            
        except Exception as e:
            pytest.fail(f"Pipeline initialization failed: {e}")

    def test_configuration_reload(self):
        """Test configuration reloading."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        # Initial load
        config1 = manager.load_all_configs()
        
        # Reload
        config2 = manager.reload_config()
        
        # Should have same structure
        assert config1.keys() == config2.keys()
        assert config1["version"] == config2["version"]

    def test_missing_base_config_error(self):
        """Test error handling for missing base configuration."""
        # Remove base.json
        (self.config_dir / "base.json").unlink()
        
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ConfigurationError):
            manager.load_base_config()

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON in config files."""
        # Create invalid JSON
        invalid_path = self.config_dir / "invalid.json"
        with open(invalid_path, 'w') as f:
            f.write("{ invalid json content")
        
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        with pytest.raises(ConfigurationError):
            manager.load_config_file("invalid.json", "router")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])