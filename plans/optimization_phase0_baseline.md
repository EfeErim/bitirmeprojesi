# Phase 0 Implementation: Baseline & Relations

Date: 2026-02-24  
Scope: Repository baseline for optimization, de-bloat, and redundancy reduction.

## 1) Visual Flow Chart

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 0: BASELINE & RELATIONS                     │
└──────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ INPUT SOURCES                                                                │
│ src/  scripts/  tests/  colab_notebooks/  config/  docs/                    │
└──────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP A: FILE RELATION MAPPING                                                │
│ imports, call chains, ownership boundaries                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP B: ENTRYPOINT INVENTORY                                                 │
│ classify execution paths as canonical vs compatibility                       │
└──────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP C: BASELINE QUALITY SNAPSHOT                                            │
│ python/test/docs checks to establish before-vs-after reference               │
└──────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT PACKAGE                                                               │
│ 1) Relation Matrix  2) Entrypoint Matrix  3) Baseline Metrics               │
└──────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌───────────────────────────────────────────────────────┐
        │ GATE CHECK: all 3 outputs complete and reviewable?   │
        └───────────────────────────────────────────────────────┘
                 │Yes                                   │No
                 ▼                                      ▼
┌───────────────────────────────┐         ┌───────────────────────────────────┐
│ Proceed to Phase 1            │         │ Resolve gaps in A/B/C and rerun   │
│ (Consistency & De-bloat)      │         │ Phase 0 checks                    │
└───────────────────────────────┘         └───────────────────────────────────┘
```

## 2) Relation Matrix (Implemented)

Primary baseline source: `docs/REPO_FILE_RELATIONS.md`

| Area | Responsibility | Key Relations | Main Risks (for optimization) |
|---|---|---|---|
| `src/router/` | VLM routing and policy dispatch | Used by pipeline and regression scripts | Large monolith (`vlm_pipeline.py`) increases coupling and bloat risk |
| `src/pipeline/` | End-to-end multi-crop orchestration | Consumes router + adapter + config | Sensitive to config/source-of-truth drift |
| `src/training/` | Phase 1/2/3 trainers + colab variants | Called by notebooks and local scripts | Duplicate patterns across colab/non-colab implementations |
| `src/dataset/` | Data prep/load/cache/error handling | Used by training and preparation flows | Multiple parallel utilities can drift |
| `scripts/` | Operational setup/smoke/regression tooling | Wrap/validate core modules | Potential overlap among setup and test scripts |
| `tests/` | Unit/integration/colab verification | Mirrors `src/` responsibilities | Selection/collection mismatch can hide gaps |
| `colab_notebooks/` | User-facing Colab execution paths | Depend on scripts/config and training modules | Duplication with script flows and bootstrap variants |
| `config/` | Runtime and testing configuration | Consumed by router/pipeline/tests | Source-of-truth ambiguity (`base.json`/`colab.json` vs split configs) |

## 3) Entrypoint Inventory (Implemented)

### Canonical script entrypoints
- `scripts/run_policy_regression_bundle.py`
- `scripts/run_python_sanity_bundle.py`
- `scripts/validate_notebook_imports.py`
- `scripts/test_dynamic_taxonomy.py`
- `scripts/test_pipeline_final_check.py`
- `scripts/colab_test_upload.py`
- `scripts/check_markdown_links.py`

### Compatibility wrappers (root)
- `validate_notebook_imports.py` -> delegates to `scripts/validate_notebook_imports.py`
- `test_dynamic_taxonomy.py` -> delegates to `scripts/test_dynamic_taxonomy.py`
- `test_pipeline_final_check.py` -> delegates to `scripts/test_pipeline_final_check.py`
- `colab_test_upload.py` -> delegates to `scripts/colab_test_upload.py`

### Notebook execution entrypoints
- `colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb`
- `colab_notebooks/1_data_preparation.ipynb`
- `colab_notebooks/2_phase1_training.ipynb`
- `colab_notebooks/3_phase2_training.ipynb`
- `colab_notebooks/4_phase3_training.ipynb`
- `colab_notebooks/5_testing_validation.ipynb`
- `colab_notebooks/6_performance_monitoring.ipynb`
- `colab_notebooks/7_VLM_ROUTER_ONECLICK.ipynb`
- `colab_notebooks/TEST_VLM_ROUTER.ipynb`
- `colab_notebooks/colab_bootstrap.ipynb`

### In-module runnable entrypoints (`if __name__ == "__main__"`)
- 18 matches in `src/` (training, dataset, ood, debugging, evaluation)
- 11 matches in `scripts/`

## 4) Baseline Quality Snapshot (Implemented)

Environment baseline:
- Python: `3.13.5`
- venv path: `.venv`

Executed checks:

1. `& "d:/bitirme projesi/.venv/Scripts/python.exe" -V`  
   Result: PASS (`Python 3.13.5`)

2. `& "d:/bitirme projesi/.venv/Scripts/python.exe" -m pytest -c config/pytest.ini tests/import_test.py`  
   Result: FAIL (exit code 1, collected 0 items)

3. `& "d:/bitirme projesi/.venv/Scripts/python.exe" scripts/check_markdown_links.py --root .`  
   Result: PASS (`no broken local markdown links found in 51 files`)

Phase 0 signal discovered:
- README quick-start references `pytest -c config/pytest.ini tests/import_test.py`, but current `pytest.ini` pattern (`python_files = test_*.py`) does not collect `import_test.py`.
- This is a documentation/validation mismatch and should be queued in Phase 1 consistency fixes.

## 5) Phase 1 Readiness Gate

Gate status: **READY WITH NOTED ISSUE**

Completed Phase 0 outputs:
- [x] Relation matrix
- [x] Entrypoint inventory
- [x] Baseline metrics snapshot

Must-carry issue into Phase 1:
- Align test invocation and discovery expectations around `tests/import_test.py` (rename file, adapt pytest config, or update README command).

## 6) Immediate Next Actions (Phase 1 kickoff backlog)

1. Fix README/pytest collection mismatch for import smoke test.
2. Build stale-reference list from docs/CI/setup metadata and mark each item keep/deprecate/remove.
3. Draft dependency canonicalization proposal (`requirements.txt` + `requirements_colab.txt` strategy).