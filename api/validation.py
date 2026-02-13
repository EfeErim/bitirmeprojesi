"""
Shared validation utilities for AADS-ULoRA API endpoints.
"""

import base64
import re
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image
import imghdr
import logging

logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
VALID_IMAGE_FORMATS = {'jpeg', 'png', 'bmp', 'gif', 'tiff'}


def validate_base64_image(b64_string: str) -> Tuple[bytes, str]:
    """
    Validate and decode a base64 image string.
    
    Returns:
        Tuple of (decoded_bytes, image_format)
    
    Raises:
        ValueError: If validation fails
    """
    try:
        # Check if valid base64
        decoded = base64.b64decode(b64_string, validate=True)
        
        # Check size limit
        if len(decoded) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image size exceeds {MAX_IMAGE_SIZE_MB}MB limit")
        
        # Check if valid image format
        image_format = imghdr.what(None, h=decoded)
        if image_format not in VALID_IMAGE_FORMATS:
            raise ValueError(f"Invalid image format: {image_format}. Supported formats: {VALID_IMAGE_FORMATS}")
        
        # Check image integrity
        try:
            image = Image.open(BytesIO(decoded))
            image.verify()  # Verify image integrity
        except Exception as e:
            raise ValueError(f"Corrupted image: {str(e)}")
        
        return decoded, image_format
        
    except base64.binascii.Error as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")
    except Exception as e:
        raise ValueError(f"Image validation failed: {str(e)}")


def validate_image_file(file: bytes) -> str:
    """
    Validate an image file from multipart upload.
    
    Returns:
        Image format
    
    Raises:
        ValueError: If validation fails
    """
    try:
        # Check size limit
        if len(file) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image size exceeds {MAX_IMAGE_SIZE_MB}MB limit")
        
        # Check if valid image format
        image_format = imghdr.what(None, h=file)
        if image_format not in VALID_IMAGE_FORMATS:
            raise ValueError(f"Invalid image format: {image_format}. Supported formats: {VALID_IMAGE_FORMATS}")
        
        # Check image integrity
        try:
            image = Image.open(BytesIO(file))
            image.verify()  # Verify image integrity
        except Exception as e:
            raise ValueError(f"Corrupted image: {str(e)}")
        
        return image_format
        
    except Exception as e:
        raise ValueError(f"Image validation failed: {str(e)}")


def validate_uuid(uuid_str: str) -> bool:
    """
    Validate UUID format.
    
    Returns:
        True if valid UUID
    
    Raises:
        ValueError: If invalid UUID
    """
    uuid_regex = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    if not re.match(uuid_regex, uuid_str):
        raise ValueError(f"Invalid UUID format: {uuid_str}")
    return True


def sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """
    Sanitize input string to prevent injection attacks.
    
    Args:
        input_str: Input string to sanitize
        max_length: Maximum allowed length
    
    Returns:
        Sanitized string
    
    Raises:
        ValueError: If input is invalid
    """
    if not isinstance(input_str, str):
        raise ValueError("Input must be a string")
    
    if len(input_str) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")
    
    # Escape potentially dangerous characters to prevent XSS/injection
    sanitized = input_str.replace('<', '<').replace('>', '>').replace('"', '"').replace("'", '&#x27;')
    
    return sanitized


def validate_location_data(location: dict) -> dict:
    """
    Validate location data with proper bounds checking.
    
    Args:
        location: Dictionary with latitude and longitude
    
    Returns:
        Validated location dictionary
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(location, dict):
        raise ValueError("Location must be a dictionary")
    
    latitude = location.get('latitude')
    longitude = location.get('longitude')
    
    if latitude is None or longitude is None:
        raise ValueError("Location must contain both latitude and longitude")
    
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Latitude must be between -90 and 90, got: {latitude}")
    
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Longitude must be between -180 and 180, got: {longitude}")
    
    # Validate accuracy if present
    accuracy = location.get('accuracy_meters')
    if accuracy is not None:
        if not isinstance(accuracy, (int, float)) or accuracy < 0:
            raise ValueError("Accuracy must be a non-negative number")
    
    return {
        'latitude': float(latitude),
        'longitude': float(longitude),
        'accuracy_meters': float(accuracy) if accuracy is not None else None
    }


def validate_crop_hint(crop_hint: str, valid_crops: list) -> str:
    """
    Validate crop hint against valid crops.
    
    Args:
        crop_hint: Crop hint string
        valid_crops: List of valid crop names
    
    Returns:
        Validated crop hint
    
    Raises:
        ValueError: If validation fails
    """
    if not crop_hint:
        return crop_hint
    
    if crop_hint not in valid_crops:
        raise ValueError(f"Invalid crop hint: {crop_hint}. Must be one of: {valid_crops}")
    
    return crop_hint


def validate_metadata(metadata: dict) -> dict:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata dictionary
    
    Returns:
        Validated metadata dictionary
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")
    
    validated = {}
    
    # Validate capture_timestamp if present
    timestamp = metadata.get('capture_timestamp')
    if timestamp:
        # Basic timestamp validation (ISO 8601 format)
        try:
            from dateutil.parser import parse
            parse(timestamp)
            validated['capture_timestamp'] = timestamp
        except Exception:
            raise ValueError("Invalid timestamp format. Use ISO 8601 format")
    
    # Validate device_model if present
    device_model = metadata.get('device_model')
    if device_model:
        validated['device_model'] = sanitize_input(device_model, max_length=100)
    
    # Validate os_version if present
    os_version = metadata.get('os_version')
    if os_version:
        validated['os_version'] = sanitize_input(os_version, max_length=50)
    
    return validated


def validate_batch_images(images: list) -> list:
    """
    Validate a list of base64 images for batch processing.
    
    Args:
        images: List of base64 image strings
    
    Returns:
        List of validated image data
    
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(images, list):
        raise ValueError("Images must be a list")
    
    if len(images) < 1 or len(images) > 10:
        raise ValueError("Batch request must contain between 1 and 10 images")
    
    validated_images = []
    
    for i, image_b64 in enumerate(images):
        try:
            decoded, _ = validate_base64_image(image_b64)
            validated_images.append(decoded)
        except ValueError as e:
            raise ValueError(f"Image {i+1} validation failed: {str(e)}")
    
    return validated_images
