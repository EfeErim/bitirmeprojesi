# Optimization Phase 4 — Prioritized Backlog (Necessary + Low-Risk)

Date: 2026-02-24

## 1) Scope
Phase 4 covers both:
- **Necessary** tasks (correctness, stability, and contract consistency)
- **Low-risk** tasks (maintainability and readability improvements with minimal behavior risk)

---

## 2) Necessary Tasks (Completed in kickoff)

### N1. Stabilize `SimpleCropRouter` constructor contract
- **Why**: Unit tests and external usage expected configurable backbone id via `model_name`.
- **Change**:
  - Added `model_name` parameter to `SimpleCropRouter.__init__` in `src/router/simple_crop_router.py`.
  - Kept backward compatibility with default `facebook/dinov2-giant`.
  - Derived `feature_dim` from model config (`hidden_size`) when available.
- **Outcome**: Constructor API mismatch resolved.

### N2. Restore backward-compatible save/load aliases
- **Why**: Existing tests/usage referenced `save_model` and `load_model`.
- **Change**:
  - Added `save_model` -> `save_checkpoint` alias.
  - Added `load_model` -> `load_checkpoint` alias.
  - Updated summary backbone value to reflect actual `model_name`.
- **Outcome**: Legacy API calls remain functional.

### N3. Remove import-time heavy model execution from router tests
- **Why**: Script-style test module executed model loading during pytest collection and caused instability.
- **Change**:
  - Replaced `tests/unit/router/test_router_comprehensive.py` with proper pytest tests.
  - Disabled heavy model path in that test contract by using `vlm_enabled=False`.
- **Outcome**: `tests/unit/router` now collects and runs safely.

### N4. Make `tests/unit/test_router.py` deterministic and contract-accurate
- **Why**: Previous tests relied on external model download and mismatched method/return contracts.
- **Change**:
  - Added local dummy backbone patch via `monkeypatch`.
  - Aligned route test to actual return type (`str` crop label).
  - Validated save/load via aliases and deterministic checkpoint path.
- **Outcome**: Fast, offline, reliable SimpleCropRouter unit tests.

Validation snapshot:
- `pytest tests/unit/test_router.py -v` -> **3 passed**.
- `pytest -c config/pytest.ini tests/unit/router -v` -> **18 passed**.

---

## 3) Necessary Tasks (Completed)

### N5. Resolve AMP deprecation warnings in Phase 3 trainer paths
- **Priority**: High
- **Implemented**:
  - Migrated Phase 3 AMP usage to `torch.amp` API in:
    - `src/training/colab_phase3_conec_lora.py`
    - `src/training/phase3_conec_lora.py`
    - `src/training/phase3_runtime.py`
  - Replaced deprecated `torch.cuda.amp.GradScaler` / `torch.cuda.amp.autocast` calls.
- **Validation**:
  - `pytest -c config/pytest.ini tests/colab/test_smoke_training.py -k "phase3" -v` -> **8 passed**.
- **Risk note**: Medium (training-path changes), mitigated by smoke coverage.

### N6. Normalize heavy-model tests to explicit marker/opt-in policy
- **Priority**: High
- **Implemented**:
  - Added `heavy_model` marker in `config/pytest.ini`.
  - Added `--runheavymodel` CLI option and default skip behavior in `tests/conftest.py`.
  - Marked heavy end-to-end Colab integration tests with `@pytest.mark.heavy_model` in `tests/integration/test_colab_integration.py`.
- **Validation**:
  - `pytest -c config/pytest.ini tests/integration/test_colab_integration.py -q` -> **11 skipped** (as expected by default policy).
- **Risk note**: Low-medium (test-organization only).

---

## 4) Low-Risk Backlog (Completed)

### L1. Router module docstrings and type-hint tightening
- Added concise contract docstrings, explicit exports (`__all__`), and local type aliases in:
  - `src/router/policy_taxonomy_utils.py`
  - `src/router/roi_helpers.py`
  - `src/router/roi_pipeline.py`

### L2. Validation command consistency in docs
- Normalized optimization docs to use the same validated command set for router checks:
  - `pytest -c config/pytest.ini tests/unit/router -v`
  - `python scripts/profile_policy_sanity.py`

### L3. Small cleanup pass for stale comments and naming consistency
- Cleaned stale wording/naming in optimization reporting context while preserving behavior.

---

## 5) Execution Order Recommendation
1. **N5** (AMP deprecation) — necessary runtime correctness/forward compatibility.
2. **N6** (heavy-model marker policy) — necessary test reliability hardening.
3. **L1-L3** low-risk polish in one compact pass.

---

## 6) Exit Criteria for Phase 4
- Necessary tasks N5-N6 completed and validated.
- No regression in router/unit smoke suites.
- Docs and validation commands reflect the stabilized workflow.

Status update:
- `pytest -c config/pytest.ini tests/unit/router -v` -> **18 passed**.
- `pytest -c config/pytest.ini tests/colab/test_smoke_training.py -k "phase3" -v` -> **8 passed**.
- `python scripts/check_markdown_links.py --root .` -> **no broken links**.
