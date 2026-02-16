# AADS-ULoRA Configuration System

## Overview

The AADS-ULoRA configuration system provides a centralized, validated approach to managing all application settings. It uses a modular architecture with separate configuration files for different concerns.

## Configuration Files

### Core Configuration Files

- **base.json**: Base configuration with core settings (API, database, ML models)
- **router-config.json**: Crop routing configuration
- **ood-config.json**: Out-of-distribution detection configuration
- **monitoring-config.json**: System monitoring and metrics configuration
- **security-config.json**: Security and authentication configuration

### Environment-Specific Configuration

- **development.json**: Development environment overrides
- **production.json**: Production environment overrides
- **staging.json**: Staging environment overrides
- **test.json**: Test environment overrides

## Configuration Structure

### Base Configuration (`base.json`)

Contains fundamental application settings:

```json
{
  "version": "5.5.0",
  "application": {
    "name": "AADS-ULoRA",
    "environment": "development",
    "debug": false,
    "log_level": "INFO"
  },
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
      "backbone": "dinov2_vits14",
      "num_classes": 10
    }
  }
}
```

### Router Configuration (`router-config.json`)

Configures the crop routing system:

```json
{
  "router": {
    "enabled": true,
    "type": "enhanced",
    "strategy": "vlm_based",
    "crop_mapping": {
      "tomato": {
        "parts": ["leaf", "fruit", "stem", "whole"],
        "model_path": "models/tomato_adapter",
        "priority": 1
      }
    },
    "fallback_strategy": "best_available",
    "confidence_threshold": 0.7,
    "vlm": {
      "enabled": true,
      "confidence_threshold": 0.8,
      "max_detections": 10
    },
    "caching": {
      "enabled": true,
      "max_size": 1000
    }
  }
}
```

### OOD Configuration (`ood-config.json`)

Configures out-of-distribution detection:

```json
{
  "ood": {
    "enabled": true,
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
```

### Monitoring Configuration (`monitoring-config.json`)

Configures system monitoring:

```json
{
  "monitoring": {
    "enabled": true,
    "prometheus": {
      "enabled": true,
      "port": 9090
    },
    "logging": {
      "format": "json",
      "rotate": true
    },
    "metrics": {
      "enabled": true,
      "track_requests": true
    }
  }
}
```

### Security Configuration (`security-config.json`)

Configures security settings:

```json
{
  "security": {
    "api_key_required": false,
    "rate_limit": {
      "enabled": true,
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
```

## Usage

### In Python Code

```python
from src.core.config_manager import get_config, config_manager

# Get complete merged configuration
config = get_config()

# Get specific values
api_host = config_manager.get_config("api.host")
router_enabled = config_manager.get_config("router.enabled")

# Access nested configuration
crop_mapping = config["router"]["crop_mapping"]
```

### Configuration Loading Process

1. Load base configuration from `base.json`
2. Load all modular configs (`router-config.json`, `ood-config.json`, etc.)
3. Validate each config against its schema
4. Merge all configurations (modular configs override base)
5. Load environment-specific overrides
6. Apply environment overrides to merged config

### Command-Line Utilities

```bash
# Validate all configuration files
python scripts/config_utils.py validate

# Show complete configuration
python scripts/config_utils.py show

# Show specific configuration key
python scripts/config_utils.py show --key router.crop_mapping

# Initialize configuration directory
python scripts/config_utils.py init
```

## Validation

All configuration files are validated against JSON schemas defined in `src/core/schemas.py`. The validation ensures:

- Required fields are present
- Data types are correct
- Value ranges are valid
- Enum values are from allowed sets

## Environment Override

Set the `APP_ENV` environment variable to load environment-specific configuration:

```bash
export APP_ENV=production
# or
set APP_ENV=development  # Windows
```

Environment configurations override the merged settings, allowing for different settings in development, staging, and production.

## Schema Definitions

Schemas are defined in `src/core/schemas.py` and registered with the configuration validator. Each schema includes:

- Property definitions with types and constraints
- Required field lists
- Enum restrictions
- Default values

## Best Practices

1. **Never commit secrets**: Use environment variables for sensitive data
2. **Validate before deployment**: Run `scripts/config_utils.py validate` before deploying
3. **Use environment overrides**: Keep environment-specific settings in separate files
4. **Document changes**: Update this README when modifying configuration structure
5. **Test configurations**: Include configuration tests in your test suite

## Migration from Old Structure

The new configuration system is backward compatible. Existing code that reads from `CONFIG` dictionary will continue to work. The new system provides:

- Better organization with separate files
- Schema validation
- Environment-specific overrides
- Configuration utilities
- Hot-reloading support

To migrate:

1. Update imports to use `from src.core.config_manager import get_config`
2. Replace direct file reading with `get_config()` calls
3. Use `config_manager.get_config("key")` for dynamic access
4. Add configuration validation to your deployment process