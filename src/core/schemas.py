#!/usr/bin/env python3
"""
Configuration schemas for AADS-ULoRA
Defines JSON schemas for all configuration files.
"""

from typing import Dict, Any

def router_schema() -> Dict[str, Any]:
    """Schema for router configuration."""
    inner = {
        "title": "Router Configuration",
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "type": {"type": "string", "enum": ["basic", "enhanced", "vlm"]},
            "strategy": {"type": "string", "enum": ["vlm_based", "classifier_based", "hybrid"]},
            "crop_mapping": {
                "type": "object",
                "patternProperties": {
                    "^[a-z_]+$": {
                        "type": "object",
                        "properties": {
                            "parts": {"type": "array", "items": {"type": "string"}},
                            "model_path": {"type": "string"},
                            "priority": {"type": "integer", "minimum": 1},
                            "timeout_ms": {"type": "integer", "minimum": 1}
                        },
                        "required": ["parts", "model_path"]
                    }
                }
            },
            "fallback_strategy": {"type": "string", "enum": ["best_available", "first_available", "reject"]},
            "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "max_retries": {"type": "integer", "minimum": 0},
            "timeout_ms": {"type": "integer", "minimum": 1},
            "vlm": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "use_diagnostic_scouting": {"type": "boolean"},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_detections": {"type": "integer", "minimum": 1},
                    "min_crop_confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "caching": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "ttl_seconds": {"type": "integer", "minimum": 0},
                    "max_size": {"type": "integer", "minimum": 1},
                    "key_prefix": {"type": "string"}
                }
            },
            "metrics": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "track_latency": {"type": "boolean"},
                    "track_accuracy": {"type": "boolean"},
                    "track_cache_hits": {"type": "boolean"}
                }
            }
        },
        "required": ["enabled", "type"]
    }
    
    # Wrap in outer schema with "router" property
    return {
        "title": "Router Configuration Schema",
        "type": "object",
        "properties": {
            "router": inner
        },
        "required": ["router"]
    }

def ood_schema() -> Dict[str, Any]:
    """Schema for OOD configuration."""
    inner = {
        "title": "Out-of-Distribution Configuration",
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "method": {"type": "string", "enum": ["mahalanobis", "prototype", "ensemble"]},
            "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
            "prototype": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "update_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    "min_samples": {"type": "integer", "minimum": 1},
                    "max_prototypes": {"type": "integer", "minimum": 1},
                    "distance_metric": {"type": "string", "enum": ["euclidean", "cosine", "mahalanobis"]},
                    "adaptive_threshold": {"type": "boolean"}
                }
            },
            "mahalanobis": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "eps": {"type": "number", "minimum": 0},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "use_shared_covariance": {"type": "boolean"},
                    "regularization": {"type": "number", "minimum": 0},
                    "num_samples": {"type": "integer", "minimum": 1}
                }
            },
            "thresholding": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["static", "dynamic", "quantile"]},
                    "factor": {"type": "number", "minimum": 0},
                    "min_val_samples_per_class": {"type": "integer", "minimum": 1},
                    "fallback_threshold": {"type": "number"},
                    "target_fpr": {"type": "number", "minimum": 0, "maximum": 1},
                    "calibration_enabled": {"type": "boolean"},
                    "quantile_method": {"type": "string", "enum": ["empirical", "theoretical"]}
                }
            },
            "fallback": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "strategy": {"type": "string", "enum": ["conservative", "aggressive", "reject"]},
                    "default_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reject_uncertain": {"type": "boolean"},
                    "min_confidence_for_prediction": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "monitoring": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "track_ood_scores": {"type": "boolean"},
                    "track_thresholds": {"type": "boolean"},
                    "alert_on_shift": {"type": "boolean"},
                    "window_size": {"type": "integer", "minimum": 1}
                }
            }
        },
        "required": ["enabled", "method"]
    }
    
    # Wrap in outer schema with "ood" property
    return {
        "title": "OOD Configuration Schema",
        "type": "object",
        "properties": {
            "ood": inner
        },
        "required": ["ood"]
    }

def monitoring_schema() -> Dict[str, Any]:
    """Schema for monitoring configuration."""
    inner = {
        "title": "Monitoring Configuration",
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "prometheus": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    "path": {"type": "string"},
                    "multiprocess_dir": {"type": ["string", "null"]},
                    "max_metrics_age": {"type": "integer", "minimum": 0}
                }
            },
            "logging": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["json", "text", "structured"]},
                    "rotate": {"type": "boolean"},
                    "max_size_mb": {"type": "integer", "minimum": 1},
                    "backup_count": {"type": "integer", "minimum": 0},
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "include_timestamp": {"type": "boolean"},
                    "include_hostname": {"type": "boolean"},
                    "structured": {"type": "boolean"}
                }
            },
            "metrics": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "track_requests": {"type": "boolean"},
                    "track_latency": {"type": "boolean"},
                    "track_errors": {"type": "boolean"},
                    "track_cache": {"type": "boolean"},
                    "track_ood": {"type": "boolean"},
                    "track_gpu": {"type": "boolean"},
                    "track_memory": {"type": "boolean"},
                    "custom_labels": {"type": "object"}
                }
            },
            "health": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "detailed": {"type": "boolean"},
                    "include_memory": {"type": "boolean"},
                    "include_cuda": {"type": "boolean"},
                    "include_disk": {"type": "boolean"},
                    "check_interval": {"type": "integer", "minimum": 1}
                }
            },
            "alerting": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "channels": {"type": "array"},
                    "rules": {
                        "type": "object",
                        "properties": {
                            "high_error_rate": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean"},
                                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                    "window": {"type": "integer", "minimum": 1},
                                    "cooldown": {"type": "integer", "minimum": 0}
                                }
                            },
                            "high_latency": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean"},
                                    "threshold_ms": {"type": "integer", "minimum": 1},
                                    "percentile": {"type": "number", "minimum": 0, "maximum": 100},
                                    "window": {"type": "integer", "minimum": 1}
                                }
                            },
                            "low_accuracy": {
                                "type": "object",
                                "properties": {
                                    "enabled": {"type": "boolean"},
                                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                    "window": {"type": "integer", "minimum": 1}
                                }
                            }
                        }
                    }
                }
            },
            "dashboard": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "refresh_interval": {"type": "number", "minimum": 1},
                    "retention_days": {"type": "integer", "minimum": 1}
                }
            }
        },
        "required": ["enabled"]
    }
    
    # Wrap in outer schema with "monitoring" property
    return {
        "title": "Monitoring Configuration Schema",
        "type": "object",
        "properties": {
            "monitoring": inner
        },
        "required": ["monitoring"]
    }

def security_schema() -> Dict[str, Any]:
    """Schema for security configuration."""
    inner = {
        "title": "Security Configuration",
        "type": "object",
        "properties": {
            "api_key_required": {"type": "boolean"},
            "api_keys": {
                "type": "array",
                "items": {"type": "string"}
            },
            "rate_limit": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "requests_per_minute": {"type": "integer", "minimum": 1},
                    "burst": {"type": "integer", "minimum": 1},
                    "by_ip": {"type": "boolean"},
                    "by_endpoint": {"type": "boolean"}
                }
            },
            "auth": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "jwt_secret": {"type": "string"},
                    "jwt_algorithm": {"type": "string"},
                    "token_expire_minutes": {"type": "integer", "minimum": 1},
                    "refresh_token_enabled": {"type": "boolean"},
                    "refresh_token_expire_days": {"type": "integer", "minimum": 1},
                    "password_min_length": {"type": "integer", "minimum": 1},
                    "require_special_chars": {"type": "boolean"}
                }
            },
            "cors": {
                "type": "object",
                "properties": {
                    "allow_origins": {"type": "array", "items": {"type": "string"}},
                    "allow_credentials": {"type": "boolean"},
                    "allow_methods": {"type": "array", "items": {"type": "string"}},
                    "allow_headers": {"type": "array", "items": {"type": "string"}},
                    "expose_headers": {"type": "array", "items": {"type": "string"}},
                    "max_age": {"type": "integer", "minimum": 0}
                }
            },
            "input_validation": {
                "type": "object",
                "properties": {
                    "max_request_size_mb": {"type": "number", "minimum": 0.1},
                    "max_image_size_mb": {"type": "number", "minimum": 0.1},
                    "max_image_dimensions": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "allowed_image_formats": {"type": "array", "items": {"type": "string"}},
                    "sanitize_inputs": {"type": "boolean"},
                    "validate_json_schema": {"type": "boolean"}
                }
            },
            "headers": {
                "type": "object",
                "properties": {
                    "strict_transport_security": {"type": "boolean"},
                    "content_security_policy": {"type": "string"},
                    "x_frame_options": {"type": "string"},
                    "x_content_type_options": {"type": "string"},
                    "server_header": {"type": "boolean"}
                }
            },
            "logging": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "log_auth_events": {"type": "boolean"},
                    "log_security_events": {"type": "boolean"},
                    "log_request_headers": {"type": "boolean"},
                    "log_response_headers": {"type": "boolean"},
                    "redact_sensitive_fields": {"type": "array", "items": {"type": "string"}}
                }
            },
            "encryption": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "algorithm": {"type": "string"},
                    "key_rotation_days": {"type": "integer", "minimum": 1},
                    "use_hsm": {"type": "boolean"}
                }
            }
        },
        "required": ["api_key_required"]
    }
    
    # Wrap in outer schema with "security" property
    return {
        "title": "Security Configuration Schema",
        "type": "object",
        "properties": {
            "security": inner
        },
        "required": ["security"]
    }
