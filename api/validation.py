"""
Backward compatibility wrapper for validation utilities.
This module re-exports from src.core.validation to maintain backward compatibility.
"""

# Re-export all validation functions from the new location
from src.core.validation import (
    validate_base64_image,
    validate_image_file,
    validate_uuid,
    sanitize_input,
    validate_location_data,
    validate_crop_hint,
    validate_metadata,
    validate_batch_images
)

__all__ = [
    'validate_base64_image',
    'validate_image_file',
    'validate_uuid',
    'sanitize_input',
    'validate_location_data',
    'validate_crop_hint',
    'validate_metadata',
    'validate_batch_images'
]
