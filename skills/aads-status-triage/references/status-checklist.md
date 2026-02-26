# AADS Status Checklist

## Quick Snapshot Commands

```bash
git status --short --branch
git diff --name-only
git log --oneline -n 8
```

## Source Area To Validation Map

| Area | Primary Files | Minimum Validation |
|---|---|---|
| Router policy/stage logic | `src/router/vlm_pipeline.py`, `src/router/roi_pipeline.py`, `src/router/policy_taxonomy_utils.py`, `src/router/simple_crop_router.py` | `pytest tests/unit/router -v`, `python scripts/run_policy_regression_bundle.py` |
| OOD behavior | `src/ood/dynamic_thresholds.py`, `src/ood/prototypes.py`, `src/ood/mahalanobis.py` | `pytest tests/unit/ood -v`, `pytest tests/unit/adapters -v` |
| Training phase logic | `src/training/phase1_training.py`, `src/training/phase2_sd_lora.py`, `src/training/phase3_conec_lora.py`, `src/training/phase3_components.py`, `src/training/phase3_runtime.py`, and `src/training/colab_*` | `pytest tests/unit/training -v`, `pytest tests/colab/test_smoke_training.py -v` |
| Adapter lifecycle + eval metrics | `src/adapter/independent_crop_adapter.py`, `src/evaluation/v55_metrics.py` | `pytest tests/unit/adapters -v`, `pytest tests/unit/evaluation -v` |
| Dataset and pipeline assembly | `src/dataset/*`, `src/pipeline/independent_multi_crop_pipeline.py` | `pytest tests/unit/dataset -v`, `pytest tests/unit/pipeline -v` |
| Core config/contracts | `src/core/*`, `config/*.json` | `pytest tests/unit/validation -v`, `pytest tests/integration/test_configuration_integration.py -v --runintegration` |
| Monitoring/debugging/utilities | `src/monitoring/*`, `src/debugging/*`, `src/utils/*` | `pytest tests/unit/monitoring -v`, `pytest tests/unit/debugging -v`, `pytest tests/unit/utils -v` |
| Visualization surfaces | `src/visualization/*` | `pytest tests/unit/visualization -v` |
| Docs and notebook path updates | `README.md`, `docs/**`, `colab_notebooks/**`, `scripts/README.md` | `python scripts/check_markdown_links.py --root .`, `python scripts/validate_notebook_imports.py` |

## CI-Alignment Bundles

- Fast local sanity:

```bash
python scripts/run_python_sanity_bundle.py
```

- Policy/profile regression:

```bash
python scripts/run_policy_regression_bundle.py
```

- Performance guardrails (router runtime related changes):

```bash
python scripts/benchmark_router_phase5.py
python scripts/check_phase5_perf_regression.py
```

## Optional Expensive Checks

- Heavy model tests:

```bash
pytest tests/ -v --runheavymodel
```

- Integration-marked tests:

```bash
pytest tests/integration -v --runintegration
```

## Triaging Rule Of Thumb

- Keep mandatory checks minimal and directly tied to touched surfaces.
- Keep expensive checks in optional section unless risk profile requires them.
- Prefer one focused behavior check per impacted module over broad suite runs.
