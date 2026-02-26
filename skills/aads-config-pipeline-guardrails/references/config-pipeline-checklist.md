# Config and Pipeline Contract Checklist

## Critical Surfaces

- Config schemas and validation: `src/core/schemas.py`, `src/core/configuration_validator.py`, `src/core/validation.py`
- Config loading and defaults: `src/core/config_manager.py`, `config/base.json`, `config/colab.json`, `config/plant_taxonomy.json`
- Pipeline assembly and orchestration: `src/pipeline/independent_multi_crop_pipeline.py`, `src/core/pipeline_manager.py`
- Artifact and registry contracts: `src/core/artifact_manifest.py`, `src/core/model_registry.py`, `src/core/colab_contract.py`

## Surface To Validation Map

| Change surface | Minimum validation | Extra checks |
|---|---|---|
| Schema/validation rules | `pytest tests/unit/validation -v` | `pytest tests/integration/test_configuration_integration.py -v --runintegration` |
| Config defaults/key paths | `pytest tests/unit/validation -v` | `pytest tests/integration/test_configuration_final.py -v --runintegration` |
| Pipeline assembly behavior | `pytest tests/unit/pipeline -v` | `pytest tests/integration/test_full_pipeline.py -v --runintegration` |
| Artifact manifest/registry contracts | `pytest tests/unit/validation -v` | `pytest tests/integration/test_configuration_final.py -v --runintegration` |

## Compatibility Rules

- Compatible: Existing config files and artifacts work without edits.
- Soft-incompatible: Existing config files and artifacts still work with aliasing or warnings.
- Incompatible: Existing config files or artifacts require explicit migration steps.

## Migration Checklist

- List every added, renamed, and removed config key.
- Define fallback behavior for missing/legacy keys.
- Confirm legacy artifacts still load or provide conversion steps.
- Update docs only when user-facing behavior or paths changed.

## Command Bundles

- Focused contract checks:

```bash
pytest tests/unit/validation -v
pytest tests/integration/test_configuration_integration.py -v --runintegration
```

- Full config and pipeline checks:

```bash
pytest tests/integration/test_configuration_final.py -v --runintegration
pytest tests/integration/test_full_pipeline.py -v --runintegration
```

- Optional broad sanity:

```bash
python scripts/run_python_sanity_bundle.py
```
