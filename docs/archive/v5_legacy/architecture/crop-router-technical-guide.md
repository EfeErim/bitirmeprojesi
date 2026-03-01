# Crop Router Technical Guide

## Scope

This guide documents the router path currently implemented in:

- `src/pipeline/independent_multi_crop_pipeline.py`
- `src/router/vlm_pipeline.py`

## Router Responsibilities

The router stage is responsible for:

1. estimating crop identity,
2. optionally identifying plant part,
3. providing confidence scores used downstream,
4. enabling adapter selection.

## Implementation Summary

### Pipeline Integration

`IndependentMultiCropPipeline`:

- initializes `VLMPipeline` and `DiagnosticScoutingAnalyzer` in `initialize_router()`,
- routes images when `crop` is not explicitly provided,
- propagates router confidence into final response (`router_confidence`, `crop_confidence`).

### Cache Behavior

The pipeline includes two LRU caches:

- `router_cache`
- `adapter_cache`

Cache characteristics:

- optional enable/disable via config,
- configurable capacity,
- optional TTL,
- hit/miss counters.

## Configuration Inputs

Router behavior is primarily configured under `router` in config files (for example `config/base.json` and `config/colab.json`):

- `router.enabled`
- `router.crop_mapping`
- `router.confidence_threshold`
- cache-related settings (`cache_enabled`, `cache_size`, optional TTL)

## Routing Flow

1. Input image arrives (tensor, PIL image, or path).
2. Cache key is generated.
3. If cache miss, image is preprocessed to tensor.
4. `_route_image(...)` executes router logic.
5. Pipeline selects corresponding crop adapter.
6. Adapter result and router metadata are merged in output.

## Failure and Fallback Strategy

- If router is uninitialized, runtime errors are surfaced.
- If no adapter exists for predicted crop, pipeline returns an error-annotated result.
- OOD metadata is still propagated where available.

## Operational Guidance

- Keep `router.crop_mapping` synchronized with available adapters.
- Validate router confidence distributions during model/version updates.
- Use integration tests in `tests/integration/` after routing logic changes.
