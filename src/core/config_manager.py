#!/usr/bin/env python3
"""
Configuration Management System for AADS-ULoRA
Provides centralized loading, validation, and management of all configuration files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .configuration_validator import config_validator, ConfigurationError

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Centralized configuration management system.
    Handles loading, validation, and merging of configuration files.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._validated_configs: Dict[str, Dict[str, Any]] = {}
        self._base_config: Optional[Dict[str, Any]] = None
        
        # Register all schemas
        self._register_schemas()
    
    def _register_schemas(self):
        """Register all configuration schemas with the validator."""
        from .schemas import (
            router_schema,
            ood_schema,
            monitoring_schema,
            security_schema
        )
        
        # Register each schema (call functions to get actual schemas)
        config_validator.register_schema(
            "router",
            router_schema(),
            default_values={
                "enabled": True,
                "type": "enhanced",
                "strategy": "vlm_based",
                "fallback_strategy": "best_available",
                "confidence_threshold": 0.7,
                "max_retries": 3,
                "timeout_ms": 5000
            }
        )
        
        config_validator.register_schema(
            "ood",
            ood_schema(),
            default_values={
                "enabled": True,
                "method": "mahalanobis",
                "threshold": 0.95,
                "confidence_level": 0.99
            }
        )
        
        config_validator.register_schema(
            "monitoring",
            monitoring_schema(),
            default_values={
                "enabled": True,
                "prometheus": {"enabled": True, "port": 9090},
                "logging": {"format": "json", "rotate": True},
                "metrics": {"enabled": True},
                "health": {"enabled": True, "detailed": True}
            }
        )
        
        config_validator.register_schema(
            "security",
            security_schema(),
            default_values={
                "api_key_required": False,
                "rate_limit": {"enabled": True, "requests_per_minute": 100},
                "cors": {"allow_origins": ["*"]},
                "input_validation": {"max_request_size_mb": 10}
            }
        )
        
        logger.info("All configuration schemas registered")
    
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file."""
        base_path = self.config_dir / "base.json"
        try:
            with open(base_path, 'r') as f:
                self._base_config = json.load(f)
            logger.info(f"Loaded base configuration from {base_path}")
            return self._base_config
        except Exception as e:
            logger.error(f"Failed to load base configuration: {e}")
            raise ConfigurationError(f"Base configuration loading failed: {e}")
    
    def load_config_file(self, filename: str, schema_name: str = None) -> Dict[str, Any]:
        """
        Load and optionally validate a configuration file.
        
        Args:
            filename: Name of the config file
            schema_name: Optional schema name for validation
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract the section to validate if schema_name provided
            config_to_validate = config
            if schema_name and schema_name in config:
                config_to_validate = config[schema_name]
            
            # Validate if schema provided
            if schema_name:
                config_to_validate = config_validator.validate(schema_name, config_to_validate)
                logger.info(f"Validated {filename} against schema '{schema_name}'")
            
            # Store the full config, but replace the section with validated version
            if schema_name and schema_name in config:
                config[schema_name] = config_to_validate
            else:
                config = config_to_validate
            
            self._configs[filename] = config
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_path}: {e}")
            raise ConfigurationError(f"Configuration file {filename} is invalid JSON: {e}")
        except ConfigurationError as e:
            logger.error(f"Validation failed for {filename}: {e}")
            raise
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configuration files and merge them into a single configuration.
        
        Returns:
            Complete merged configuration
        """
        # Load base config first
        if not self._base_config:
            self.load_base_config()
        
        # Load all specific configs
        config_files = [
            ("router-config.json", "router"),
            ("ood-config.json", "ood"),
            ("monitoring-config.json", "monitoring"),
            ("security-config.json", "security")
        ]
        
        for filename, schema_name in config_files:
            try:
                self.load_config_file(filename, schema_name)
            except ConfigurationError as e:
                logger.warning(f"Failed to load {filename}: {e}. Using defaults.")
        
        # Merge configurations
        merged_config = self._base_config.copy()
        
        # Merge each config file into the base
        for config_name, config_data in self._configs.items():
            # Extract the top-level key from each config file
            for key, value in config_data.items():
                if key not in merged_config:
                    merged_config[key] = {}
                if isinstance(value, dict) and isinstance(merged_config.get(key), dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
        
        self._validated_configs["merged"] = merged_config
        logger.info("Configuration merge completed")
        
        return merged_config
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (dot notation supported)."""
        config = self._validated_configs.get("merged", {})
        
        if '.' in key:
            # Support dot notation for nested access
            parts = key.split('.')
            current = config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        
        return config.get(key, default)
    
    def validate_merged_config(self) -> bool:
        """Validate the complete merged configuration."""
        try:
            # Custom validation rules can be added here
            config = self._validated_configs.get("merged", {})
            
            # Validate critical sections exist
            required_sections = ['api', 'database', 'ml', 'router', 'ood', 'monitoring', 'security']
            for section in required_sections:
                if section not in config:
                    logger.warning(f"Missing required configuration section: {section}")
            
            logger.info("Merged configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Merged configuration validation failed: {e}")
            return False
    
    def reload_config(self):
        """Reload all configuration files."""
        self._configs.clear()
        self._validated_configs.clear()
        self._base_config = None
        logger.info("Configuration cleared for reload")
        return self.load_all_configs()
    
    def get_environment_config(self, env: str) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_config_path = self.config_dir / f"{env}.json"
        
        if not env_config_path.exists():
            logger.warning(f"Environment configuration not found: {env_config_path}")
            return {}
        
        try:
            with open(env_config_path, 'r') as f:
                env_config = json.load(f)
            
            # Environment configs override base settings
            return env_config
            
        except Exception as e:
            logger.error(f"Failed to load environment config {env}: {e}")
            return {}


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config() -> Dict[str, Any]:
    """Get the complete merged configuration."""
    if "merged" not in config_manager._validated_configs:
        return config_manager.load_all_configs()
    return config_manager._validated_configs["merged"]


def reload_configuration():
    """Reload configuration (useful for hot-reloading)."""
    return config_manager.reload_config()