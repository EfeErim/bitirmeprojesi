# Coding Playbook

## Module To Test Mapping

| Module area | Primary tests | Extra checks |
|---|---|---|
| `src/router/*` | `pytest tests/unit/router -v` | `python scripts/run_policy_regression_bundle.py` |
| `src/ood/*` | `pytest tests/unit/ood -v` | `pytest tests/unit/adapters -v` |
| `src/training/*` | `pytest tests/unit/training -v` | `pytest tests/colab/test_smoke_training.py -v`, `pytest tests/unit/adapters -v` |
| `src/adapter/*` | `pytest tests/unit/adapters -v` | `pytest tests/unit/training -v`, `pytest tests/unit/ood -v` |
| `src/dataset/*` | `pytest tests/unit/dataset -v` | `pytest tests/unit/pipeline -v`, `pytest tests/colab/test_data_pipeline.py -v` |
| `src/pipeline/*` | `pytest tests/unit/pipeline -v` | `pytest tests/integration/test_full_pipeline.py -v --runintegration` |
| `src/core/*` | `pytest tests/unit/validation -v` | `pytest tests/integration/test_configuration_integration.py -v --runintegration` |
| `src/evaluation/*` | `pytest tests/unit/evaluation -v` | `pytest tests/integration/test_full_pipeline.py -v --runintegration` |
| `src/monitoring/*` | `pytest tests/unit/monitoring -v` | `pytest tests/unit/debugging -v` |
| `src/debugging/*` | `pytest tests/unit/debugging -v` | `pytest tests/unit/monitoring -v` |
| `src/utils/*` | `pytest tests/unit/utils -v` | `python scripts/run_python_sanity_bundle.py` |
| `src/visualization/*` | `pytest tests/unit/visualization -v` | `pytest tests/unit/utils -v` |
| `config/*.json` | `pytest tests/unit/validation -v` | `pytest tests/integration/test_configuration_integration.py -v --runintegration` |
| docs/notebooks/scripts path changes | `python scripts/check_markdown_links.py --root .` | `python scripts/validate_notebook_imports.py` |

## Common Command Bundles

- Fast sanity:

```bash
python scripts/run_python_sanity_bundle.py
```

- Docs and notebook sync sanity:

```bash
python scripts/validate_notebook_imports.py
python scripts/check_markdown_links.py --root .
```

- Full unit set (when change is broad):

```bash
pytest tests/unit -v
```

- Performance guardrails (router runtime path changes):

```bash
python scripts/benchmark_router_phase5.py
python scripts/check_phase5_perf_regression.py
```

- Integration bundle (use only for cross-module risk):

```bash
pytest tests/integration -v --runintegration
```

## Guardrails

- Avoid changing behavior and interfaces together unless required.
- Keep root wrappers as compatibility aliases; prefer canonical `scripts/` paths.
- Use `--runintegration` and `--runheavymodel` only when relevant.
- Prefer one focused check per touched module before broad suite execution.
