# Crop Router Architecture - VLM Pipeline

## Overview

The AADS-ULoRA system now uses the **VLM (Vision-Language Model) Pipeline** as its definitive router for plant disease diagnosis. This represents the culmination of research into optimal detection methods for agricultural applications.

## Current Architecture: VLM Pipeline

### Multi-Stage Detection Approach

The VLM pipeline implements a three-stage process as recommended in "Researching Plant Detection Methods.pdf" (Scenario B: Diagnostic Scouting):

```
Input Image
    ↓
┌─────────────────────────────────────┐
│  Stage 1: Grounding DINO            │
│  - Open-set object detection        │
│  - Finds all plant parts            │
│  - Returns bounding boxes           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Stage 2: SAM-2                     │
│  - Zero-shot segmentation           │
│  - Isolates plant tissue            │
│  - Creates precise masks            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Stage 3: BioCLIP 2                 │
│  - Taxonomic identification         │
│  - Hierarchical classification      │
│  - Species, disease, organ          │
└─────────────────────────────────────┘
    ↓
Comprehensive Analysis Report
```

### Performance Metrics

- **Accuracy**: 97.27% on tomato disease identification
- **Hardware Requirements**: >24GB VRAM (32GB recommended)
- **Throughput**: <5 FPS (suitable for scouting applications)
- **Output**: Detailed analysis with segmentation masks and taxonomic classifications

### Key Advantages

1. **High Accuracy**: Multi-stage approach ensures precise detection and classification
2. **Explainability**: Provides natural language explanations and visual segmentation
3. **Comprehensive**: Identifies species, disease, and plant organ in single pass
4. **Research-Backed**: Based on latest findings in vision-language models for agriculture

## Deprecated Alternatives

### Simple Crop Router (Deprecated)
- **Status**: No longer used in production
- **Reason**: Limited to crop classification only; lower accuracy for complex disease identification
- **Legacy**: Kept for reference in `src/router/simple_crop_router.py`

### Enhanced Crop Router (Deprecated)
- **Status**: Not actively used
- **Reason**: VLM pipeline supersedes its dual-classification approach with more sophisticated multi-stage detection
- **Legacy**: Kept for reference in `src/router/enhanced_crop_router.py`

## Integration

The VLM pipeline is integrated into the main system through:

```python
from src.router.vlm_pipeline import VLMPipeline, DiagnosticScoutingAnalyzer

# Direct usage
pipeline = VLMPipeline(config, device='cuda')
pipeline.load_models()
result = pipeline.process_image(image_tensor)

# Or via high-level analyzer
analyzer = DiagnosticScoutingAnalyzer(config, device='cuda')
result = analyzer.analyze(image_tensor, crop_type='tomato')
```

The pipeline is used by [`IndependentMultiCropPipeline`](src/pipeline/independent_multi_crop_pipeline.py) which maintains API compatibility while internally leveraging VLM for routing and diagnosis.

## Migration from Legacy Routers

### Before (Simple Crop Router)
```python
from src.router.simple_crop_router import SimpleCropRouter

router = SimpleCropRouter(crops=['tomato', 'pepper', 'corn'])
crop, confidence = router.route(image_tensor)
```

### After (VLM Pipeline)
```python
from src.router.vlm_pipeline import DiagnosticScoutingAnalyzer

analyzer = DiagnosticScoutingAnalyzer(config)
result = analyzer.analyze(image_tensor)
# result contains: classifications, detections, segmented_objects, explanation
```

## Configuration

The VLM pipeline is configured through the main config file (e.g., `config/adapter-spec.json`):

```json
{
  "vlm_enabled": true,
  "vlm_confidence_threshold": 0.8,
  "vlm_max_detections": 10
}
```

## Future Directions

- Optimize VLM pipeline for batch processing
- Quantization for reduced VRAM requirements
- Integration with edge deployment options

---

