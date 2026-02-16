#!/usr/bin/env python3
"""
Unit tests for VLM Pipeline (replacing SimpleCropRouter)
"""

import pytest
import torch
from src.router.vlm_pipeline import VLMPipeline, DiagnosticScoutingAnalyzer
from src.utils.data_loader import CropDataset

class TestVLMPipeline:
    """Test cases for VLM Pipeline."""
    
    def test_initialization(self):
        """Test VLM pipeline initialization."""
        config = {
            'vlm_enabled': True,
            'vlm_confidence_threshold': 0.8,
            'vlm_max_detections': 10
        }
        pipeline = VLMPipeline(config, device='cpu')
        
        assert pipeline.device.type == 'cpu'
        assert pipeline.enabled == True
        assert pipeline.confidence_threshold == 0.8
        assert pipeline.max_detections == 10
    
    def test_process_image_returns_correct_format(self):
        """Test that process_image returns expected output structure."""
        config = {'vlm_enabled': True}
        pipeline = VLMPipeline(config, device='cpu')
        pipeline.load_models()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        result = pipeline.process_image(dummy_input)
        
        # Check result structure
        assert 'status' in result
        assert 'scenario' in result
        assert result['scenario'] == 'diagnostic_scouting'
    
    def test_analyzer_initialization(self):
        """Test DiagnosticScoutingAnalyzer initialization."""
        config = {'vlm_enabled': True}
        analyzer = DiagnosticScoutingAnalyzer(config, device='cpu')
        
        assert analyzer.vlm_pipeline is not None
        assert analyzer.device.type == 'cpu'
    
    def test_analyzer_quick_assessment(self):
        """Test quick assessment method."""
        config = {'vlm_enabled': True}
        analyzer = DiagnosticScoutingAnalyzer(config, device='cpu')
        analyzer.vlm_pipeline.enabled = False  # Disable for testing
        
        dummy_input = torch.randn(1, 3, 224, 224)
        result = analyzer.quick_assessment(dummy_input)
        
        assert 'status' in result
        assert 'explanation' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])