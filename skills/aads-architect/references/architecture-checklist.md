# Architecture Checklist

## Key Architectural Surfaces

- Contracts/config: `src/core/*`, `config/*.json`, `src/core/schemas.py`
- Routing/policy: `src/router/*`
- OOD decision path: `src/ood/*`, adapter integration points
- Training lifecycle: `src/training/*`, `src/training/phase3_components.py`, `src/training/phase3_runtime.py`, `src/adapter/independent_crop_adapter.py`
- End-to-end assembly: `src/pipeline/independent_multi_crop_pipeline.py`
- User-facing orchestration: `colab_notebooks/*`, `scripts/*`, `README.md`, `docs/REPO_FILE_RELATIONS.md`

## Interface Change Template

For each changed interface, specify:
- Current behavior
- New behavior
- Compatibility status (compatible/incompatible)
- Migration strategy
- Validation commands

## Compatibility Classes

- Compatible: old callers/configs work without edits.
- Soft-incompatible: old callers/configs work with warnings or aliases.
- Incompatible: migration steps required before runtime success.

## Suggested Validation For Contract Changes

```bash
pytest tests/unit/validation -v
pytest tests/integration/test_configuration_integration.py -v --runintegration
pytest tests/integration/test_full_pipeline.py -v --runintegration
```

## Migration Deliverables

- Config key rename/add/remove table.
- Adapter artifact compatibility notes (load, fallback, migration).
- Rollback path with concrete restore commands.

## Risk Prompts

- Does this change alter stage ordering or threshold semantics?
- Does this change require docs/notebook/script entrypoint updates?
- Is there a measurable performance risk requiring Phase 5 guardrail checks?
- Can old configs/artifacts still load without manual edits?
