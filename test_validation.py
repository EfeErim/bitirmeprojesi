#!/usr/bin/env python3
"""
Comprehensive validation tests for AADS-ULoRA API endpoints.
Tests image size limits, format validation, and input sanitization.
"""

import sys
import os
import base64
from io import BytesIO
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api.validation import validate_base64_image, validate_image_file, validate_uuid, sanitize_input

def create_test_image(size_mb=1, format='JPEG'):
    """Create a test image of specified size."""
    img = Image.new('RGB', (100, 100), color='red')
    buffered = BytesIO()
    img.save(buffered, format=format)
    return buffered.getvalue()

def create_large_image(size_mb=11):
    """Create an image larger than the limit."""
    img = Image.new('RGB', (5000, 5000), color='red')
    buffered = BytesIO()
    img.save(buffered, format='JPEG', quality=95)
    data = buffered.getvalue()
    if len(data) < size_mb * 1024 * 1024:
        padding = b'\x00' * ((size_mb * 1024 * 1024) - len(data))
        data += padding
    return data

def test_validate_base64_image_valid():
    """Test validation of a valid base64 image."""
    img_data = create_test_image(1, 'JPEG')
    b64_string = base64.b64encode(img_data).decode('utf-8')
    decoded, format = validate_base64_image(b64_string)
    assert decoded == img_data
    assert format in ['jpeg', 'jpg']

def test_validate_base64_image_oversized():
    """Test validation rejects oversized images."""
    large_img = create_large_image(11)
    b64_string = base64.b64encode(large_img).decode('utf-8')
    try:
        validate_base64_image(b64_string)
        assert False, "Should have raised ValueError for oversized image"
    except ValueError as e:
        assert "exceeds 10MB limit" in str(e)

def test_validate_base64_image_invalid_format():
    """Test validation rejects invalid image formats."""
    invalid_b64 = base64.b64encode(b"not an image").decode('utf-8')
    try:
        validate_base64_image(invalid_b64)
        assert False, "Should have raised ValueError for invalid format"
    except ValueError as e:
        assert "Invalid image format" in str(e) or "Corrupted image" in str(e)

def test_validate_base64_image_invalid_base64():
    """Test validation rejects invalid base64."""
    invalid_b64 = "not-valid-base64!!!"
    try:
        validate_base64_image(invalid_b64)
        assert False, "Should have raised ValueError for invalid base64"
    except ValueError as e:
        assert "Invalid base64" in str(e) or "Image validation failed" in str(e)

def test_validate_uuid_valid():
    """Test validation of valid UUIDs."""
    valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
    result = validate_uuid(valid_uuid)
    assert result is True

def test_validate_uuid_invalid():
    """Test validation rejects invalid UUIDs."""
    invalid_uuid = "not-a-uuid"
    try:
        validate_uuid(invalid_uuid)
        assert False, "Should have raised ValueError for invalid UUID"
    except ValueError as e:
        assert "Invalid UUID format" in str(e)

def test_sanitize_input():
    """Test input sanitization."""
    dangerous = "<script>alert('xss')</script>"
    sanitized = sanitize_input(dangerous)
    assert '<' not in sanitized
    assert '>' not in sanitized
    assert '<' in sanitized
    assert '>' in sanitized
    assert "" in sanitized
    assert "&#x27;" in sanitized

def test_sanitize_input_long():
    """Test sanitization rejects overly long input."""
    long_input = "a" * 2000
    try:
        sanitize_input(long_input, max_length=1000)
        assert False, "Should have raised ValueError for long input"
    except ValueError as e:
        assert "exceeds maximum length" in str(e)

def test_validate_image_file():
    """Test validation of image files from multipart upload."""
    img_data = create_test_image(1, 'PNG')
    format = validate_image_file(img_data)
    assert format == 'png'

def test_validate_image_file_oversized():
    """Test validation rejects oversized file uploads."""
    large_img = create_large_image(11)
    try:
        validate_image_file(large_img)
        assert False, "Should have raised ValueError for oversized file"
    except ValueError as e:
        assert "exceeds 10MB limit" in str(e)

if __name__ == "__main__":
    print("Running validation tests...")
    tests = [
        test_validate_base64_image_valid,
        test_validate_base64_image_oversized,
        test_validate_base64_image_invalid_format,
        test_validate_base64_image_invalid_base64,
        test_validate_uuid_valid,
        test_validate_uuid_invalid,
        test_sanitize_input,
        test_sanitize_input_long,
        test_validate_image_file,
        test_validate_image_file_oversized,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print("[PASS] " + test.__name__)
            passed += 1
        except Exception as e:
            print("[FAIL] " + test.__name__ + ": " + str(e))
            failed += 1
    print("\nResults: " + str(passed) + " passed, " + str(failed) + " failed")
    sys.exit(0 if failed == 0 else 1)
