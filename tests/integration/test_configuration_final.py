#!/usr/bin/env python3
"""
Final integration test to verify complete configuration system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.core.config_manager import ConfigurationManager, get_config
from src.core.configuration_validator import ConfigurationError

class TestConfigurationFinal:
    """Final integration tests for complete configuration system."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()
        
        # Create comprehensive test configuration
        self._create_complete_config()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def _create_complete_config(self):
        """Create complete test configuration."""
        # Create all required configuration files
        self._create_base_config()
        self._create_router_config()
        self._create_ood_config()
        self._create_monitoring_config()
        self._create_security_config()
        self._create_environment_configs()

    def _create_base_config(self):
        """Create base configuration."""
        base_config = {
            "version": "5.5.0-final",
            "description": "Complete configuration test",
            "application": {
                "name": "AADS-ULoRA",
                "environment": "test",
                "debug": false,
                "log_level": "INFO"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "reload": false
            },
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "name": "aads_test",
                "username": "test_user",
                "password": "test_password"
            },
            "ml": {
                "device": "cuda",
                "model": {
                    "backbone": "dinov2_vits14",
                    "num_classes": 10,
                    "adapter": {
                        "type": "lora",
                        "r": 8,
                        "lora_alpha": 16
                    }
                },
                "inference": {
                    "batch_size": 32,
                    "num_workers": 4
                }
            },
            "storage": {
                "type": "local",
                "local": {
                    "base_path": "./data",
                    "upload_dir": "uploads",
                    "model_dir": "models",
                    "results_dir": "results"
                }
            },
            "cache": {
                "enabled": true,
                "backend": "redis",
                "ttl": 3600,
                "max_size": 1000
            },
            "feedback": {
                "enabled": true,
                "storage": "database",
                "retention_days": 365
            },
            "paths": {
                "data": "./data",
                "logs": "./logs",
                "models": "./models",
                "temp": "./temp",
                "exports": "./exports"
            }
        }
        
        with open(self.config_dir / "base.json", 'w') as f:
            json.dump(base_config, f)

    def _create_router_config(self):
        """Create router configuration."""
        router_config = {
            "router": {
                "enabled": true,
                "type": "enhanced",
                "strategy": "vlm_based",
                "crop_mapping": {
                    "tomato": {
                        "parts": ["leaf", "fruit", "stem", "whole"],
                        "model_path": "models/tomato_adapter",
                        "priority": 1,
                        "timeout_ms": 100
                    },
                    "potato": {
                        "parts": ["leaf", "tuber", "stem", "whole"],
                        "model_path": "models/potato_adapter",
                        "priority": 2,
                        "timeout_ms": 100
                    },
                    "wheat": {
                        "parts": ["leaf", "ear", "stem", "whole"],
                        "model_path": "models/wheat_adapter",
                        "priority": 3,
                        "timeout_ms": 100
                    }
                },
                "fallback_strategy": "best_available",
                "confidence_threshold": 0.7,
                "max_retries": 3,
                "timeout_ms": 5000,
                "vlm": {
                    "enabled": true,
                    "use_diagnostic_scouting": true,
                    "confidence_threshold": 0.8,
                    "max_detections": 10,
                    "min_crop_confidence": 0.6
                },
                "caching": {
                    "enabled": true,
                    "ttl_seconds": 3600,
                    "max_size": 1000,
                    "key_prefix": "router"
                },
                "metrics": {
                    "enabled": true,
                    "track_latency": true,
                    "track_accuracy": true,
                    "track_cache_hits": true
                }
            }
        }
        
        with open(self.config_dir / "router-config.json", 'w') as f:
            json.dump(router_config, f)

    def _create_ood_config(self):
        """Create OOD configuration."""
        ood_config = {
            "ood": {
                "enabled": true,
                "method": "mahalanobis",
                "threshold": 0.95,
                "confidence_level": 0.99,
                "prototype": {
                    "enabled": true,
                    "update_rate": 0.1,
                    "min_samples": 10,
                    "max_prototypes": 1000,
                    "distance_metric": "euclidean",
                    "adaptive_threshold": true
                },
                "mahalanobis": {
                    "enabled": true,
                    "eps": 1e-6,
                    "batch_size": 64,
                    "use_shared_covariance": false,
                    "regularization": 1e-5,
                    "num_samples": 10000
                },
                "thresholding": {
                    "method": "dynamic",
                    "factor": 2.0,
                    "min_val_samples_per_class": 10,
                    "fallback_threshold": 25.0,
                    "target_fpr": 0.05,
                    "calibration_enabled": true,
                    "quantile_method": "empirical"
                },
                "fallback": {
                    "enabled": true,
                    "strategy": "conservative",
                    "default_confidence": 0.5,
                    "reject_uncertain": true,
                    "min_confidence_for_prediction": 0.3
                },
                "monitoring": {
                    "enabled": true,
                    "track_ood_scores": true,
                    "track_thresholds": true,
                    "alert_on_shift": true,
                    "window_size": 1000
                }
            }
        }
        
        with open(self.config_dir / "ood-config.json", 'w') as f:
            json.dump(ood_config, f)

    def _create_monitoring_config(self):
        """Create monitoring configuration."""
        monitoring_config = {
            "monitoring": {
                "enabled": true,
                "prometheus": {
                    "enabled": true,
                    "port": 9090,
                    "path": "/metrics",
                    "multiprocess_dir": null,
                    "max_metrics_age": 10000
                },
                "logging": {
                    "format": "json",
                    "rotate": true,
                    "max_size_mb": 100,
                    "backup_count": 5,
                    "level": "INFO",
                    "include_timestamp": true,
                    "include_hostname": true,
                    "structured": true
                },
                "metrics": {
                    "enabled": true,
                    "track_requests": true,
                    "track_latency": true,
                    "track_errors": true,
                    "track_cache": true,
                    "track_ood": true,
                    "track_gpu": true,
                    "track_memory": true,
                    "custom_labels": {}
                },
                "health": {
                    "enabled": true,
                    "detailed": true,
                    "include_memory": true,
                    "include_cuda": true,
                    "include_disk": true,
                    "check_interval": 30
                },
                "alerting": {
                    "enabled": false,
                    "channels": [],
                    "rules": {
                        "high_error_rate": {
                            "enabled": true,
                            "threshold": 0.05,
                            "window": 60,
                            "cooldown": 300
                        },
                        "high_latency": {
                            "enabled": true,
                            "threshold_ms": 1000,
                            "percentile": 95,
                            "window": 60
                        },
                        "low_accuracy": {
                            "enabled": false,
                            "threshold": 0.90,
                            "window": 100
                        }
                    }
                },
                "dashboard": {
                    "enabled": false,
                    "refresh_interval": 5,
                    "retention_days": 7
                }
            }
        }
        
        with open(self.config_dir / "monitoring-config.json", 'w') as f:
            json.dump(monitoring_config, f)

    def _create_security_config(self):
        """Create security configuration."""
        security_config = {
            "security": {
                "api_key_required": false,
                "api_keys": [],
                "rate_limit": {
                    "enabled": true,
                    "requests_per_minute": 100,
                    "burst": 200,
                    "by_ip": true,
                    "by_endpoint": false
                },
                "auth": {
                    "enabled": false,
                    "jwt_secret": "CHANGE_IN_PRODUCTION",
                    "jwt_algorithm": "HS256",
                    "token_expire_minutes": 1440,
                    "refresh_token_enabled": true,
                    "refresh_token_expire_days": 30,
                    "password_min_length": 8,
                    "require_special_chars": false
                },
                "cors": {
                    "allow_origins": ["*"],
                    "allow_credentials": true,
                    "allow_methods": ["*"],
                    "allow_headers": ["*"],
                    "expose_headers": ["X-Process-Time"],
                    "max_age": 600
                },
                "input_validation": {
                    "max_request_size_mb": 10,
                    "max_image_size_mb": 20,
                    "max_image_dimensions": [4096, 4096],
                    "allowed_image_formats": ["jpg", "jpeg", "png", "webp"],
                    "sanitize_inputs": true,
                    "validate_json_schema": true
                },
                "headers": {
                    "strict_transport_security": true,
                    "content_security_policy": "default-src 'self'",
                    "x_frame_options": "DENY",
                    "x_content_type_options": "nosniff",
                    "server_header": false
                },
                "logging": {
                    "enabled": true,
                    "log_auth_events": true,
                    "log_security_events": true,
                    "log_request_headers": false,
                    "log_response_headers": false,
                    "redact_sensitive_fields": ["password", "token", "secret", "authorization"]
                },
                "encryption": {
                    "enabled": true,
                    "algorithm": "AES-256-GCM",
                    "key_rotation_days": 90,
                    "use_hsm": false
                }
            }
        }
        
        with open(self.config_dir / "security-config.json", 'w') as f:
            json.dump(security_config, f)

    def _create_environment_configs(self):
        """Create environment-specific configurations."""
        # Development config
        dev_config = {
            "version": "5.5.0-final-dev",
            "description": "Development environment",
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "reload": true,
                "log_level": "debug",
                "timeout_keep_alive": 30,
                "timeout_graceful_shutdown": 10
            },
            "security": {
                "api_key_required": false,
                "api_keys": [],
                "rate_limit_requests": 1000,
                "rate_limit_window": 60,
                "max_request_size_mb": 20,
                "allowed_origins": ["*"],
                "https_enforced": false
            },
            "caching": {
                "enabled": false,
                "redis_url": "redis://localhost:6379",
                "default_ttl": 3600,
                "diagnosis_ttl": 1800,
                "max_cache_size": 1000
            },
            "compression": {
                "enabled": false,
                "minimum_size": 1024,
                "compression_level": 6
            },
            "monitoring": {
                "enabled": true,
                "metrics_port": 9090,
                "log_requests": true,
                "audit_logging": true,
                "track_performance": true
            },
            "database": {
                "pool_size": 10,
                "max_overflow": 20,
                "pool_recycle": 3600,
                "pool_pre_ping": true
            },
            "health": {
                "detailed": true,
                "include_memory": true,
                "include_cuda": true
            }
        }
        
        with open(self.config_dir / "development.json", 'w') as f:
            json.dump(dev_config, f)
        
        # Production config
        prod_config = {
            "version": "5.5.0-final-prod",
            "description": "Production environment",
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "reload": false,
                "log_level": "info",
                "timeout_keep_alive": 30,
                "timeout_graceful_shutdown": 10
            },
            "security": {
                "api_key_required": true,
                "api_keys": ["prod_key_secure_token_12345"],
                "rate_limit_requests": 100,
                "rate_limit_window": 60,
                "max_request_size_mb": 10,
                "allowed_origins": ["https://yourdomain.com"],
                "https_enforced": true
            },
            "caching": {
                "enabled": true,
                "redis_url": "redis://localhost:6379",
                "default_ttl": 3600,
                "diagnosis_ttl": 1800,
                "max_cache_size": 1000
            },
            "compression": {
                "enabled": true,
                "minimum_size": 1024,
                "compression_level": 6
            },
            "monitoring": {
                "enabled": true,
                "metrics_port": 9090,
                "log_requests": true,
                "audit_logging": true,
                "track_performance": true
            },
            "database": {
                "pool_size": 20,
                "max_overflow": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": true
            },
            "health": {
                "detailed": true,
                "include_memory": true,
                "include_cuda": true
            }
        }
        
        with open(self.config_dir / "production.json", 'w') as f:
            json.dump(prod_config, f)

    def test_complete_configuration_system(self):
        """Test complete configuration system integration."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        # Load and validate all configurations
        config = manager.load_all_configs()
        assert manager.validate_merged_config() is True
        
        # Test configuration values
        assert config["version"] == "5.5.0-final"
        assert config["application"]["name"] == "AADS-ULoRA"
        assert config["api"]["host"] == "0.0.0.0"
        assert config["api"]["port"] == 8000
        assert config["database"]["type"] == "postgresql"
        assert config["ml"]["device"] == "cuda"
        
        # Test router configuration
        assert config["router"]["enabled"] is True
        assert config["router"]["type"] == "enhanced"
        assert "tomato" in config["router"]["crop_mapping"]
        assert config["router"]["crop_mapping"]["tomato"]["priority"] == 1
        assert config["router"]["vlm"]["enabled"] is True
        assert config["router"]["caching"]["enabled"] is True
        
        # Test OOD configuration
        assert config["ood"]["enabled"] is True
        assert config["ood"]["method"] == "mahalanobis"
        assert config["ood"]["prototype"]["update_rate"] == 0.1
        assert config["ood"]["mahalanobis"]["eps"] == 1e-6
        
        # Test monitoring configuration
        assert config["monitoring"]["enabled"] is True
        assert config["monitoring"]["prometheus"]["port"] == 9090
        assert config["monitoring"]["logging"]["format"] == "json"
        assert config["monitoring"]["metrics"]["enabled"] is True
        
        # Test security configuration
        assert config["security"]["api_key_required"] is False
        assert config["security"]["rate_limit"]["enabled"] is True
        assert config["security"]["cors"]["allow_origins"] == ["*"]
        assert config["security"]["input_validation"]["max_request_size_mb"] == 10
        
        # Test environment-specific overrides
        env_config = manager.get_environment_config("development")
        assert env_config["version"] == "5.5.0-final-dev"
        assert env_config["api"]["reload"] is True
        assert env_config["security"]["api_key_required"] is False
        
        env_config = manager.get_environment_config("production")
        assert env_config["version"] == "5.5.0-final-prod"
        assert env_config["api"]["reload"] is False
        assert env_config["security"]["api_key_required"] is True

    def test_configuration_access_patterns(self):
        """Test various configuration access patterns."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        manager.load_all_configs()
        
        # Test dot notation access
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

    def test_configuration_validation_errors(self):
        """Test configuration validation error handling."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        # Test invalid configuration
        invalid_config = {
            "router": {
                "enabled": "invalid_type",  # Should be boolean
                "confidence_threshold": 1.5  # Should be between 0 and 1
            }
        }
        
        # This would fail validation - testing error handling
        with pytest.raises(ConfigurationError):
            manager.load_config_file("invalid.json", "router")

    def test_configuration_reload(self):
        """Test configuration reloading."""
        manager = ConfigurationManager(config_dir=str(self.config_dir))
        
        # Initial load
        config1 = manager.load_all_configs()
        
        # Modify a config file
        with open(self.config_dir / "base.json", 'r') as f:
            base_config = json.load(f)
        
        base_config["version"] = "5.5.0-final-modified"
        
        with open(self.config_dir / "base.json", 'w') as f:
            json.dump(base_config, f)
        
        # Reload
        config2 = manager.reload_config()
        
        # Should reflect changes
        assert config2["version"] == "5.5.0-final-modified"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])