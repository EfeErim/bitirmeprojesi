"""
Schema validation for configuration files and runtime parameters.
Provides centralized validation for all configuration inputs.
"""

import os
import json
import re
from typing import Dict, Any, List, Union, Callable
import logging

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover - environment-dependent fallback
    jsonschema = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigurationValidator:
    """Centralized configuration validation system."""
    
    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._default_values: Dict[str, Dict[str, Any]] = {}
        self._validation_rules: Dict[str, List[Callable]] = {}
        
    def register_schema(
        self,
        schema_name: str,
        schema: Dict[str, Any],
        default_values: Dict[str, Any] = None,
        validation_rules: List[Callable] = None
    ):
        """Register a new schema for validation."""
        self._schemas[schema_name] = schema
        self._default_values[schema_name] = default_values or {}
        self._validation_rules[schema_name] = validation_rules or []
        
        logger.info(f"Registered schema: {schema_name}")
    
    def validate(
        self,
        schema_name: str,
        config: Dict[str, Any],
        allow_extra_fields: bool = False
    ) -> Dict[str, Any]:
        """Validate configuration against a schema."""
        if schema_name not in self._schemas:
            raise ConfigurationError(f"Schema {schema_name} not registered")
        
        schema = self._schemas[schema_name]
        # If the registered schema expects a top-level wrapper (e.g., a 'router'
        # property) but the provided config is the inner object, wrap it so the
        # JSON schema validation matches the schema shape. This keeps tests and
        # real config files compatible.
        properties = schema.get('properties', {})
        if schema_name in properties and schema_name not in config:
            config_to_validate = {schema_name: config}
        else:
            config_to_validate = config

        # Apply default values
        validated_config = self._apply_defaults(schema_name, config_to_validate)
        
        # Validate against JSON schema
        try:
            if jsonschema is not None:
                jsonschema.validate(instance=validated_config, schema=schema)
            else:
                self._validate_schema_subset(validated_config, schema, path=f"${schema_name}")
        except Exception as e:
            # Normalize all schema failures under ConfigurationError.
            raise ConfigurationError(
                f"Schema validation failed for '{schema_name}': {str(e)}"
            ) from e

        # Run custom validation rules
        for rule in self._validation_rules[schema_name]:
            try:
                rule(validated_config)
            except Exception as e:
                raise ConfigurationError(
                    f"Custom validation failed for '{schema_name}': {str(e)}"
                ) from e
        
        # Check for extra fields if not allowed
        if not allow_extra_fields:
            self._check_extra_fields(schema_name, validated_config)
        
        return validated_config
    
    def _apply_defaults(self, schema_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration."""
        defaults = self._default_values.get(schema_name, {})
        validated_config = config.copy()

        schema = self._schemas.get(schema_name, {})
        properties = schema.get('properties', {})
        is_wrapped = (
            schema_name in properties
            and isinstance(validated_config.get(schema_name), dict)
        )

        target = validated_config[schema_name] if is_wrapped else validated_config

        for key, default_value in defaults.items():
            if key not in target:
                target[key] = default_value
        
        return validated_config
    
    def _check_extra_fields(self, schema_name: str, config: Dict[str, Any]):
        """Check for extra fields not defined in schema."""
        schema = self._schemas[schema_name]
        properties = schema.get('properties', {})
        
        extra_fields = [
            key for key in config.keys()
            if key not in properties
        ]
        
        if extra_fields:
            raise ConfigurationError(
                f"Extra fields found in configuration: {', '.join(extra_fields)}"
            )

    def _validate_schema_subset(self, value: Any, schema: Dict[str, Any], path: str = "$") -> None:
        """Fallback schema validator for environments where jsonschema import fails.

        Supports the subset used by this repository:
        - type, enum, minimum, maximum
        - required, properties, patternProperties
        - items, minItems, maxItems
        """
        expected_type = schema.get("type")
        if expected_type is not None:
            self._validate_type(value, expected_type, path)

        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path} must be one of {schema['enum']}, got {value!r}")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"{path} must be >= {schema['minimum']}, got {value}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"{path} must be <= {schema['maximum']}, got {value}")

        if isinstance(value, list):
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if min_items is not None and len(value) < int(min_items):
                raise ValueError(f"{path} requires at least {min_items} items")
            if max_items is not None and len(value) > int(max_items):
                raise ValueError(f"{path} allows at most {max_items} items")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for idx, item in enumerate(value):
                    self._validate_schema_subset(item, item_schema, f"{path}[{idx}]")

        if isinstance(value, dict):
            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    raise ValueError(f"{path}.{key} is required")

            properties = schema.get("properties", {})
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    self._validate_schema_subset(value[key], child_schema, f"{path}.{key}")

            pattern_properties = schema.get("patternProperties", {})
            if pattern_properties:
                for key, child_value in value.items():
                    for pattern, child_schema in pattern_properties.items():
                        if re.match(pattern, str(key)):
                            self._validate_schema_subset(child_value, child_schema, f"{path}.{key}")

    def _validate_type(self, value: Any, expected: Union[str, List[str]], path: str) -> None:
        expected_list = [expected] if isinstance(expected, str) else list(expected)
        match = False
        for t in expected_list:
            if t == "object" and isinstance(value, dict):
                match = True
            elif t == "array" and isinstance(value, list):
                match = True
            elif t == "string" and isinstance(value, str):
                match = True
            elif t == "boolean" and isinstance(value, bool):
                match = True
            elif t == "integer" and isinstance(value, int) and not isinstance(value, bool):
                match = True
            elif t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                match = True
            elif t == "null" and value is None:
                match = True
        if not match:
            raise ValueError(f"{path} expected type {expected_list}, got {type(value).__name__}")
    
    def load_and_validate_file(
        self,
        schema_name: str,
        file_path: str,
        allow_extra_fields: bool = False
    ) -> Dict[str, Any]:
        """Load configuration from file and validate it."""
        if not os.path.exists(file_path):
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file '{file_path}': {e}"
            ) from e

        return self.validate(schema_name, config, allow_extra_fields)
    
    def create_schema(
        self,
        title: str,
        description: str,
        type_: str = "object",
        properties: Dict[str, Any] = None,
        required: List[str] = None,
        additional_properties: bool = False
    ) -> Dict[str, Any]:
        """Create a JSON schema."""
        return {
            "title": title,
            "description": description,
            "type": type_,
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": additional_properties
        }
    
    def create_property_schema(
        self,
        name: str,
        type_: Union[str, List[str]],
        description: str = "",
        default: Any = None,
        required: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a property schema."""
        property_schema = {
            "type": type_,
            "description": description
        }
        
        if default is not None:
            property_schema["default"] = default
        
        property_schema.update(kwargs)
        
        return property_schema
    
    def add_validation_rule(
        self,
        schema_name: str,
        rule: Callable[[Dict[str, Any]], None]
    ):
        """Add a custom validation rule."""
        if schema_name not in self._validation_rules:
            self._validation_rules[schema_name] = []
        
        self._validation_rules[schema_name].append(rule)
        logger.info(f"Added validation rule to schema: {schema_name}")


# Global configuration validator instance
config_validator = ConfigurationValidator()
