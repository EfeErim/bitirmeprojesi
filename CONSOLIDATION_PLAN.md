# VLM Pipeline Consolidation Plan

## Objective
Make the VLM pipeline the definitive router implementation for the plant disease detection system, removing all dependencies on alternative router implementations.

## Current State Analysis

### Existing Router Implementations
1. **VLM Pipeline** (`src/router/vlm_pipeline.py`)
   - Multi-stage: Grounding DINO + SAM-2 + BioCLIP 2
   - High accuracy: 97.27% on tomato diseases
   - High resource requirements: >24GB VRAM
   - Comprehensive output with segmentation and taxonomic classification

2. **Enhanced Crop Router** (`src/router/enhanced_crop_router.py`)
   - Dual classification: crop + plant part
   - Routes to (crop, part) adapters
   - Currently NOT used in main pipeline

3. **Simple Crop Router** (`src/router/simple_crop_router.py`)
   - Single classification: crop only
   - Routes to crop-specific adapters
   - Currently ACTIVE in main pipeline

### Active Usage of SimpleCropRouter
- `src/pipeline/independent_multi_crop_pipeline.py` - imports and uses SimpleCropRouter
- `demo/app.py` - imports and uses SimpleCropRouter
- Test files:
  - `tests/unit/test_router.py`
  - `tests/unit/test_router_comprehensive.py`
  - `tests/unit/test_router_minimal.py`
  - `tests/unit/verify_optimizations.py`
  - `tests/unit/verify_optimizations_simple.py`
  - `tests/integration/test_full_pipeline.py`
  - `tests/fixtures/test_fixtures.py`
  - `tests/unit/test_imports.py`

## Consolidation Strategy

### Phase 1: Update Core Pipeline (Priority 1)
**Target: `src/pipeline/independent_multi_crop_pipeline.py`**
- Replace SimpleCropRouter with VLMPipeline
- Adapt the routing logic to use VLM's multi-stage approach
- Maintain compatibility with existing adapter system
- Ensure the pipeline interface remains unchanged

### Phase 2: Update Demo Application (Priority 2)
**Target: `demo/app.py`**
- Switch from SimpleCropRouter to VLMPipeline
- Update initialization and usage patterns

### Phase 3: Update Test Suite (Priority 3)
**Targets: All test files that import SimpleCropRouter**
- Replace imports with VLMPipeline or DiagnosticScoutingAnalyzer
- Update test cases to match VLM pipeline's output format
- Adjust mock data and expectations

### Phase 4: Update Documentation (Priority 4)
**Targets:**
- `docs/architecture/crop-router-explanation.md`
- `docs/architecture/crop-router-technical-guide.md`
- Any other docs referencing old routers

### Phase 5: Deprecate Old Implementations (Priority 5)
**Targets:**
- `src/router/simple_crop_router.py` - Add deprecation warnings, mark as legacy
- `src/router/enhanced_crop_router.py` - Add deprecation warnings, mark as legacy
- Keep files for reference but clearly mark as deprecated

## Implementation Details

### API Compatibility
The current API (`api/main.py`) uses `IndependentMultiCropPipeline`. By updating that pipeline to use VLM internally, no API changes are required.

### Output Format Differences
**SimpleCropRouter output:**
```python
{
    'crop': 'tomato',
    'crop_confidence': 0.95,
    'disease': {...},
    'ood_analysis': {...}
}
```

**VLM Pipeline output:**
```python
{
    'status': 'success',
    'scenario': 'diagnostic_scouting',
    'detections': [...],
    'segmented_objects': [...],
    'classifications': [...],
    'explanation': '...',
    'num_objects': 5,
    'pipeline_components': [...]
}
```

The pipeline will need to adapt VLM's output to match the expected API response format.

## Success Criteria
- ✅ All imports of SimpleCropRouter removed from production code
- ✅ All tests pass with VLM pipeline
- ✅ API endpoints work unchanged (internal implementation swapped)
- ✅ Documentation clearly states VLM pipeline is the definitive choice
- ✅ No references to "alternative" routers in user-facing documentation

## Rollback Plan
- Keep old router files (simple_crop_router.py, enhanced_crop_router.py) in place but marked as deprecated
- Git history provides full restore capability
- Feature flag could be added if needed (but not in scope)

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| VLM pipeline too slow for production | High | Optimize with caching, batch processing |
| VLM output format incompatible | Medium | Create adapter layer in pipeline |
| Tests fail due to output differences | Medium | Update test expectations systematically |
| Missing adapter support | Low | Enhanced router's adapter registry can be integrated |

## Timeline
- Day 1: Phase 1 (Core Pipeline)
- Day 1: Phase 2 (Demo App)
- Day 2: Phase 3 (Tests)
- Day 2: Phase 4 (Documentation)
- Day 2: Phase 5 (Deprecation)

---

**Decision**: VLM Pipeline is the definitive choice. All other router implementations are legacy alternatives and will be deprecated.