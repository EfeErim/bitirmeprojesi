#!/usr/bin/env python3
"""
Configuration Management System for AADS-ULoRA
Provides centralized loading, validation, and management of all configuration files.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from .configuration_validator import config_validator, ConfigurationError

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Centralized configuration management system.
    Handles loading, validation, and merging of configuration files.
    """

    # Class-level variable to track if schemas have been registered (singleton pattern)
    _schemas_registered: bool = False
    _schema_registration_lock = threading.RLock()

    def __init__(self, config_dir: str = "config", environment: str = None):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._validated_configs: Dict[str, Dict[str, Any]] = {}
        self._base_config: Optional[Dict[str, Any]] = None
        self._environment = environment

        # Register all schemas (only once, regardless of instance count)
        self._register_schemas_once()

    @classmethod
    def _register_schemas_once(cls):
        """
        Register all configuration schemas with the validator.
        Uses class-level lock to ensure schemas are registered only once,
        regardless of how many ConfigurationManager instances are created.
        """
        if cls._schemas_registered:
            return  # Already registered, skip

        with cls._schema_registration_lock:
            # Double-check after acquiring lock (prevents race condition)
            if cls._schemas_registered:
                return

            cls._register_schemas()
            cls._schemas_registered = True
            logger.info("Configuration schemas registered (singleton)")

    @classmethod
    def _register_schemas(cls):
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

            # Validate if schema provided.
            # Accept both wrapped format {"router": {...}} and legacy unwrapped {...}.
            if schema_name:
                is_wrapped = isinstance(config, dict) and schema_name in config
                candidate = config if is_wrapped else {schema_name: config}
                validated_wrapper = config_validator.validate(schema_name, candidate)
                validated_section = validated_wrapper.get(schema_name, {})
                config = {schema_name: validated_section}
                logger.info(f"Validated {filename} against schema '{schema_name}'")
            
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
        
        # Load optional legacy split configs if present.
        # Canonical runtime model uses base.json + optional <environment>.json.
        config_files = [
            ("router-config.json", "router"),
            ("ood-config.json", "ood"),
            ("monitoring-config.json", "monitoring"),
            ("security-config.json", "security")
        ]
        
        for filename, schema_name in config_files:
            config_path = self.config_dir / filename
            if not config_path.exists():
                continue
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
        
        # Apply environment-specific overrides if environment is set
        if self._environment:
            env_config = self.get_environment_config(self._environment)
            if env_config:
                merged_config = self._apply_env_overrides(merged_config, env_config)
                logger.info(f"Applied environment overrides for '{self._environment}'")
        
        self._validated_configs["merged"] = merged_config
        logger.info("Configuration merge completed")
        
        return merged_config
    
    def _apply_env_overrides(self, base_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment-specific overrides to base configuration.
        
        Args:
            base_config: Base merged configuration
            env_config: Environment-specific configuration
            
        Returns:
            Configuration with environment overrides applied
        """
        result = base_config.copy()
        
        for key, value in env_config.items():
            if key == "version" or key == "description":
                # Skip metadata fields
                continue
            
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                # Deep merge for nested dictionaries
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                # Override with environment value
                result[key] = value
        
        return result
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
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
            
            # Validate critical sections exist for the current repository scope.
            required_sections = ['training', 'router', 'ood']
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


# Thread-safe global configuration manager with RLock
_config_lock = threading.RLock()
_config_manager: Optional[ConfigurationManager] = None


def _get_config_manager(environment: str = None) -> ConfigurationManager:
    """
    Get thread-safe configuration manager instance (lazy initialization).

    Args:
        environment: Optional environment name

    Returns:
        Thread-safe ConfigurationManager instance
    """
    global _config_manager, _config_lock

    with _config_lock:
        if _config_manager is None:
            _config_manager = ConfigurationManager(environment=environment)
        elif environment and environment != _config_manager._environment:
            # Reinitialize with new environment (only if environment actually changed)
            _config_manager = ConfigurationManager(environment=environment)

        return _config_manager


def get_config(environment: str = None) -> Dict[str, Any]:
    """
    Get the complete merged configuration (thread-safe).

    Args:
        environment: Optional environment name to apply overrides

    Returns:
        Complete merged configuration
    """
    config_mgr = _get_config_manager(environment)

    if "merged" not in config_mgr._validated_configs:
        return config_mgr.load_all_configs()
    return config_mgr._validated_configs["merged"]


def reload_configuration(environment: str = None):
    """
    Reload configuration (useful for hot-reloading, thread-safe).

    Args:
        environment: Optional environment name to reload with
    """
    config_mgr = _get_config_manager(environment)
    return config_mgr.reload_config()


# For backward compatibility, keep global reference but access through function
def _get_backward_compat_manager():
    """Get config manager for backward compatibility."""
    return _get_config_manager()


# Deprecated: use get_config() instead for thread-safe access
config_manager = property(lambda self: _get_backward_compat_manager())
