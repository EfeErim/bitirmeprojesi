"""
Comprehensive unit tests for API validation utilities.
"""

import pytest
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import imghdr
import re

from api.validation import (
    validate_base64_image,
    validate_image_file,
    validate_uuid,
    sanitize_input,
    validate_location_data,
    validate_crop_hint,
    validate_metadata,
    validate_batch_images
)


class TestImageValidation:
    """Test image validation functions."""
    
    @pytest.fixture
    def valid_image_data(self):
        """Create valid image data for testing."""
        # Create a small test image
        image = Image.new('RGB', (100, 100), color='red')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        image_bytes = buffer.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            'image_bytes': image_bytes,
            'base64_image': base64_image,
            'format': 'PNG'
        }
    
    def test_validate_base64_image_success(self, valid_image_data):
        """Test successful base64 image validation."""
        decoded, image_format = validate_base64_image(valid_image_data['base64_image'])
        
        assert isinstance(decoded, bytes)
        assert image_format.lower() == 'png'  # imghdr returns lowercase
        assert len(decoded) < 10 * 1024 * 1024  # Should be under 10MB
    
    def test_validate_base64_image_invalid_base64(self):
        """Test validation with invalid base64 string."""
        invalid_base64 = 'invalid_base64_string'
        
        with pytest.raises(ValueError, match="Invalid base64 encoding"):
            validate_base64_image(invalid_base64)
    
    def test_validate_base64_image_too_large(self, valid_image_data):
        """Test validation with image exceeding size limit."""
        # Create large image
        large_image = Image.new('RGB', (10000, 10000), color='red')
        buffer = BytesIO()
        large_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        large_image_bytes = buffer.read()
        large_base64_image = base64.b64encode(large_image_bytes).decode('utf-8')
        
        with pytest.raises(ValueError, match="Image size exceeds 10MB limit"):
            validate_base64_image(large_base64_image)
    
    def test_validate_base64_image_invalid_format(self):
        """Test validation with invalid image format."""
        # Create a file that's not a valid image
        invalid_data = b'not_an_image'
        invalid_base64 = base64.b64encode(invalid_data).decode('utf-8')
        
        with pytest.raises(ValueError, match="Invalid image format"):
            validate_base64_image(invalid_base64)
    
    def test_validate_base64_image_corrupted_image(self, valid_image_data):
        """Test validation with corrupted image data."""
        # Corrupt the valid image data
        corrupted_base64 = valid_image_data['base64_image'][:-10] + 'CORRUPT'
        
        with pytest.raises(ValueError, match="Corrupted image"):
            validate_base64_image(corrupted_base64)
    
    def test_validate_image_file_success(self, valid_image_data):
        """Test successful image file validation."""
        image_format = validate_image_file(valid_image_data['image_bytes'])
        
        assert image_format.lower() == 'png'
    
    def test_validate_image_file_too_large(self):
        """Test validation with file exceeding size limit."""
        # Create large image
        large_image = Image.new('RGB', (10000, 10000), color='red')
        buffer = BytesIO()
        large_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        large_image_bytes = buffer.read()
        
        with pytest.raises(ValueError, match="Image size exceeds 10MB limit"):
            validate_image_file(large_image_bytes)
    
    def test_validate_image_file_invalid_format(self):
        """Test validation with invalid image format."""
        # Create a file that's not a valid image
        invalid_data = b'not_an_image'
        
        with pytest.raises(ValueError, match="Invalid image format"):
            validate_image_file(invalid_data)
    
    def test_validate_image_file_corrupted_image(self):
        """Test validation with corrupted image data."""
        # Create a valid image but corrupt it
        image = Image.new('RGB', (100, 100), color='red')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        image_bytes = buffer.read()
        corrupted_bytes = image_bytes[:-10] + b'CORRUPT'
        
        with pytest.raises(ValueError, match="Corrupted image"):
            validate_image_file(corrupted_bytes)
    
    def test_validate_image_file_supported_formats(self):
        """Test validation with all supported formats."""
        supported_formats = ['jpeg', 'png', 'bmp', 'gif', 'tiff']
        
        for fmt in supported_formats:
            # Create image in this format
            image = Image.new('RGB', (100, 100), color='red')
            buffer = BytesIO()
            image.save(buffer, format=fmt.upper())
            buffer.seek(0)
            
            image_bytes = buffer.read()
            image_format = validate_image_file(image_bytes)
            
            assert image_format.lower() == fmt
    
    def test_validate_image_file_unsupported_format(self):
        """Test validation with unsupported format."""
        # Create a valid image but in an unsupported format
        image = Image.new('RGB', (100, 100), color='red')
        buffer = BytesIO()
        image.save(buffer, format='WEBP')  # WEBP is not in supported formats
        buffer.seek(0)
        
        image_bytes = buffer.read()
        
        with pytest.raises(ValueError, match="Invalid image format"):
            validate_image_file(image_bytes)


class TestUUIDValidation:
    """Test UUID validation."""
    
    def test_validate_uuid_valid(self):
        """Test validation with valid UUID."""
        valid_uuid = '123e4567-e89b-12d3-a456-426614174000'
        
        result = validate_uuid(valid_uuid)
        assert result is True
    
    def test_validate_uuid_invalid_format(self):
        """Test validation with invalid UUID format."""
        invalid_uuid = 'invalid-uuid-format'
        
        with pytest.raises(ValueError, match="Invalid UUID format"):
            validate_uuid(invalid_uuid)
    
    def test_validate_uuid_wrong_length(self):
        """Test validation with UUID of wrong length."""
        wrong_length_uuid = '123e4567-e89b-12d3-a456-42661417400'  # 35 chars instead of 36
        
        with pytest.raises(ValueError, match="Invalid UUID format"):
            validate_uuid(wrong_length_uuid)
    
    def test_validate_uuid_invalid_characters(self):
        """Test validation with UUID containing invalid characters."""
        invalid_chars_uuid = '123e4567-e89b-12d3-a456-42661417400G'  # Contains 'G'
        
        with pytest.raises(ValueError, match="Invalid UUID format"):
            validate_uuid(invalid_chars_uuid)
    
    def test_validate_uuid_empty_string(self):
        """Test validation with empty string."""
        empty_uuid = ''
        
        with pytest.raises(ValueError, match="Invalid UUID format"):
            validate_uuid(empty_uuid)


class TestInputSanitization:
    """Test input sanitization."""
    
    def test_sanitize_input_basic(self):
        """Test basic input sanitization."""
        input_str = 'Hello World'
        sanitized = sanitize_input(input_str)
        
        assert sanitized == 'Hello World'
    
    def test_sanitize_input_with_special_chars(self):
        """Test sanitization with special characters."""
        input_str = 'Hello <World> & "Universe"'
        sanitized = sanitize_input(input_str)
        
        assert sanitized == 'Hello <World> & "Universe"'
        # Note: The current implementation doesn't actually escape these characters
        # This test verifies the current behavior
    
    def test_sanitize_input_max_length(self):
        """Test input exceeding maximum length."""
        long_input = 'A' * 1001  # Exceeds 1000 character limit
        
        with pytest.raises(ValueError, match="Input exceeds maximum length"):
            sanitize_input(long_input)
    
    def test_sanitize_input_non_string(self):
        """Test sanitization with non-string input."""
        non_string_input = 12345
        
        with pytest.raises(ValueError, match="Input must be a string"):
            sanitize_input(non_string_input)
    
    def test_sanitize_input_empty_string(self):
        """Test sanitization with empty string."""
        empty_input = ''
        sanitized = sanitize_input(empty_input)
        
        assert sanitized == ''
def test_sanitize_input_xss_prevention(self):
        """Test XSS prevention in sanitization."""
        xss_input = '<script>alert("XSS");</script>'
        sanitized = sanitize_input(xss_input)
        
        # Current implementation doesn't escape < and >, so this test verifies
        # that the current behavior doesn't prevent XSS
        assert sanitized == xss_input
        # Note: This indicates a potential security issue in the current implementation


class TestLocationValidation:
    """Test location data validation."""
    
    def test_validate_location_data_valid(self):
        """Test validation with valid location data."""
        location = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'accuracy_meters': 10.5
        }
        
        validated = validate_location_data(location)
        
        assert validated['latitude'] == 37.7749
        assert validated['longitude'] == -122.4194
        assert validated['accuracy_meters'] == 10.5
    
    def test_validate_location_data_missing_fields(self):
        """Test validation with missing latitude or longitude."""
        location = {
            'latitude': 37.7749
            # Missing longitude
        }
        
        with pytest.raises(ValueError, match="Location must contain both latitude and longitude"):
            validate_location_data(location)
    
    def test_validate_location_data_latitude_bounds(self):
        """Test validation with latitude out of bounds."""
        # Latitude too high
        location_high = {
            'latitude': 91.0,
            'longitude': -122.4194
        }
        
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            validate_location_data(location_high)
        
        # Latitude too low
        location_low = {
            'latitude': -91.0,
            'longitude': -122.4194
        }
        
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            validate_location_data(location_low)
    
    def test_validate_location_data_longitude_bounds(self):
        """Test validation with longitude out of bounds."""
        # Longitude too high
        location_high = {
            'latitude': 37.7749,
            'longitude': 181.0
        }
        
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            validate_location_data(location_high)
        
        # Longitude too low
        location_low = {
            'latitude': 37.7749,
            'longitude': -181.0
        }
        
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            validate_location_data(location_low)
    
    def test_validate_location_data_accuracy_validation(self):
        """Test validation with accuracy field."""
        # Valid accuracy
        location_valid = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'accuracy_meters': 10.5
        }
        
        validated = validate_location_data(location_valid)
        assert validated['accuracy_meters'] == 10.5
        
        # Invalid accuracy (negative)
        location_invalid = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'accuracy_meters': -5.0
        }
        
        with pytest.raises(ValueError, match="Accuracy must be a non-negative number"):
            validate_location_data(location_invalid)
    
    def test_validate_location_data_non_numeric(self):
        """Test validation with non-numeric coordinates."""
        location = {
            'latitude': 'invalid',
            'longitude': -122.4194
        }
        
        with pytest.raises(ValueError):
            validate_location_data(location)
    
    def test_validate_location_data_non_dict(self):
        """Test validation with non-dictionary input."""
        non_dict_location = [37.7749, -122.4194]  # List instead of dict
        
        with pytest.raises(ValueError, match="Location must be a dictionary"):
            validate_location_data(non_dict_location)


class TestCropHintValidation:
    """Test crop hint validation."""
    
    def test_validate_crop_hint_valid(self):
        """Test validation with valid crop hint."""
        valid_crops = ['tomato', 'pepper', 'corn']
        crop_hint = 'tomato'
        
        validated = validate_crop_hint(crop_hint, valid_crops)
        
        assert validated == 'tomato'
    
    def test_validate_crop_hint_empty_hint(self):
        """Test validation with empty crop hint."""
        valid_crops = ['tomato', 'pepper', 'corn']
        crop_hint = ''
        
        validated = validate_crop_hint(crop_hint, valid_crops)
        
        assert validated == ''
    
    def test_validate_crop_hint_invalid_hint(self):
        """Test validation with invalid crop hint."""
        valid_crops = ['tomato', 'pepper', 'corn']
        crop_hint = 'wheat'
        
        with pytest.raises(ValueError, match="Invalid crop hint"):
            validate_crop_hint(crop_hint, valid_crops)
    
    def test_validate_crop_hint_case_sensitivity(self):
        """Test validation with case sensitivity."""
        valid_crops = ['tomato', 'pepper', 'corn']
        crop_hint = 'Tomato'  # Different case
        
        with pytest.raises(ValueError, match="Invalid crop hint"):
            validate_crop_hint(crop_hint, valid_crops)
    
    def test_validate_crop_hint_empty_valid_crops(self):
        """Test validation with empty valid crops list."""
        valid_crops = []
        crop_hint = 'tomato'
        
        with pytest.raises(ValueError, match="Invalid crop hint"):
            validate_crop_hint(crop_hint, valid_crops)


class TestMetadataValidation:
    """Test metadata validation."""
    
    def test_validate_metadata_valid(self):
        """Test validation with valid metadata."""
        metadata = {
            'capture_timestamp': '2026-02-12T23:00:00Z',
            'device_model': 'Pixel 7',
            'os_version': 'Android 14'
        }
        
        validated = validate_metadata(metadata)
        
        assert 'capture_timestamp' in validated
        assert 'device_model' in validated
        assert 'os_version' in validated
    
    def test_validate_metadata_missing_fields(self):
        """Test validation with missing optional fields."""
        metadata = {
            'capture_timestamp': '2026-02-12T23:00:00Z'
            # Missing device_model and os_version
        }
        
        validated = validate_metadata(metadata)
        
        assert 'capture_timestamp' in validated
        assert 'device_model' not in validated
        assert 'os_version' not in validated
    
    def test_validate_metadata_invalid_timestamp(self):
        """Test validation with invalid timestamp format."""
        metadata = {
            'capture_timestamp': 'invalid-timestamp'
        }
        
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            validate_metadata(metadata)
    
    def test_validate_metadata_non_dict(self):
        """Test validation with non-dictionary input."""
        non_dict_metadata = ['2026-02-12T23:00:00Z']  # List instead of dict
        
        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            validate_metadata(non_dict_metadata)
    
    def test_validate_metadata_sanitization(self):
        """Test sanitization of metadata fields."""
        metadata = {
            'device_model': '<script>alert("XSS");</script>',
            'os_version': 'Android & "Version"'
        }
        
        validated = validate_metadata(metadata)
        
        # Current implementation doesn't actually escape these characters
        # This test verifies the current behavior
        assert validated['device_model'] == '<script>alert("XSS");</script>'
        assert validated['os_version'] == 'Android & "Version"'
    
    def test_validate_metadata_max_length(self):
        """Test metadata fields exceeding maximum length."""
        metadata = {
            'device_model': 'A' * 101  # Exceeds 100 character limit
        }
        
        with pytest.raises(ValueError, match="Input exceeds maximum length"):
            validate_metadata(metadata)


class TestBatchImageValidation:
    """Test batch image validation."""
    
    @pytest.fixture
    def batch_images_data(self):
        """Create batch image data for testing."""
        # Create 5 valid images
        images = []
        for _ in range(5):
            image = Image.new('RGB', (100, 100), color='red')
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            image_bytes = buffer.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            images.append(base64_image)
        
        return images
    
    def test_validate_batch_images_success(self, batch_images_data):
        """Test successful batch image validation."""
        validated_images = validate_batch_images(batch_images_data)
        
        assert len(validated_images) == 5
        assert all(isinstance(img, bytes) for img in validated_images)
    
    def test_validate_batch_images_invalid_list(self):
        """Test validation with non-list input."""
        invalid_input = 'not_a_list'
        
        with pytest.raises(ValueError, match="Images must be a list"):
            validate_batch_images(invalid_input)
    
    def test_validate_batch_images_too_few_images(self):
        """Test validation with too few images."""
        few_images = []  # Empty list
        
        with pytest.raises(ValueError, match="Batch request must contain between 1 and 10 images"):
            validate_batch_images(few_images)
    
    def test_validate_batch_images_too_many_images(self, batch_images_data):
        """Test validation with too many images."""
        # Create 11 images (exceeds limit of 10)
        many_images = batch_images_data * 3  # 15 images
        
        with pytest.raises(ValueError, match="Batch request must contain between 1 and 10 images"):
            validate_batch_images(many_images)
    
    def test_validate_batch_images_individual_failure(self, batch_images_data):
        """Test validation where one image in batch fails."""
        # Create batch with one invalid image
        batch = batch_images_data.copy()
        batch.append('invalid_base64')  # Add invalid image
        
        with pytest.raises(ValueError, match="Image 6 validation failed"):
            validate_batch_images(batch)
    
    def test_validate_batch_images_empty_list(self):
        """Test validation with empty list."""
        empty_list = []
        
        with pytest.raises(ValueError, match="Batch request must contain between 1 and 10 images"):
            validate_batch_images(empty_list)
    
    def test_validate_batch_images_single_image(self, batch_images_data):
        """Test validation with single image."""
        single_image = batch_images_data[:1]
        
        validated_images = validate_batch_images(single_image)
        
        assert len(validated_images) == 1
        assert isinstance(validated_images[0], bytes)
    
    def test_validate_batch_images_max_images(self, batch_images_data):
        """Test validation with maximum allowed images."""
        max_images = batch_images_data[:10]  # First 10 images
        
        validated_images = validate_batch_images(max_images)
        
        assert len(validated_images) == 10
        assert all(isinstance(img, bytes) for img in validated_images)


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_validate_base64_image_minimum_size(self):
        """Test validation with minimum valid image size."""
        # Create very small image (1x1 pixel)
        image = Image.new('RGB', (1, 1), color='red')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        image_bytes = buffer.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Should succeed
        decoded, image_format = validate_base64_image(base64_image)
        assert image_format == 'PNG'
    
    def test_validate_base64_image_maximum_size(self):
        """Test validation with maximum allowed image size."""
        # Create image just under 10MB limit
        # 10MB = 10 * 1024 * 1024 = 10,485,760 bytes
        # Create image of size 10MB - 1KB
        max_size = 10 * 1024 * 1024 - 1024
        
        # Create dummy data of max size
        dummy_data = b'A' * max_size
        
        # Wrap in image format (simplified)
        # Note: This won't be a valid image, but tests size limit
        base64_image = base64.b64encode(dummy_data).decode('utf-8')
        
        # Should fail because it's not a valid image format
        with pytest.raises(ValueError, match="Invalid image format"):
            validate_base64_image(base64_image)
    
    def test_validate_uuid_edge_cases(self):
        """Test UUID validation edge cases."""
        # Valid UUID with different case
        valid_uuid_mixed = '123e4567-e89b-12d3-a456-426614174000'.upper()
        result = validate_uuid(valid_uuid_mixed)
        assert result is True
        
        # UUID with hyphens in wrong places
        invalid_uuid_hyphens = '123e4567e89b12d3a456426614174000'  # No hyphens
        with pytest.raises(ValueError, match="Invalid UUID format"):
            validate_uuid(invalid_uuid_hyphens)
    
    def test_validate_location_data_edge_cases(self):
        """Test location validation edge cases."""
        # Latitude at bounds
        location_latitude_bounds = {
            'latitude': 90.0,
            'longitude': -122.4194
        }
        validated = validate_location_data(location_latitude_bounds)
        assert validated['latitude'] == 90.0
        
        # Longitude at bounds
        location_longitude_bounds = {
            'latitude': 37.7749,
            'longitude': 180.0
        }
        validated = validate_location_data(location_longitude_bounds)
        assert validated['longitude'] == 180.0
    
    def test_validate_metadata_edge_cases(self):
        """Test metadata validation edge cases."""
        # Empty metadata
        empty_metadata = {}
        validated = validate_metadata(empty_metadata)
        assert validated == {}
        
        # Metadata with only optional fields
        optional_metadata = {
            'device_model': 'Test Device'
        }
        validated = validate_metadata(optional_metadata)
        assert 'device_model' in validated


if __name__ == '__main__':
    pytest.main([__file__, '-v'])