# Training Lifecycle Invariants

## Phase Invariants

- Phase 1 (DoRA): preserve expected DoRA setup and baseline OOD threshold preparation path.
- Phase 2 (SD-LoRA): preserve selective adaptation assumptions (`lora_A` freezing behavior where expected by tests/docs).
- Phase 3 (CoNeC-LoRA): preserve layer-wise adaptation/freezing assumptions used by current v5.5 behavior, including runtime and component orchestration behavior.

## Critical Training Surfaces

- `src/training/phase1_training.py`
- `src/training/phase2_sd_lora.py`
- `src/training/phase3_conec_lora.py`
- `src/training/phase3_components.py`
- `src/training/phase3_runtime.py`
- `src/adapter/independent_crop_adapter.py`
- `src/evaluation/v55_metrics.py`

## Parity Rule

For edits to any of these files, review the paired Colab/local variant:
- `src/training/phase1_training.py` <-> `src/training/colab_phase1_training.py`
- `src/training/phase2_sd_lora.py` <-> `src/training/colab_phase2_sd_lora.py`
- `src/training/phase3_conec_lora.py` <-> `src/training/colab_phase3_conec_lora.py`

## Validation Commands

- Training unit checks:

```bash
pytest tests/unit/training -v
```

- Colab smoke training checks:

```bash
pytest tests/colab/test_smoke_training.py -v
```

- Adapter/OOD alignment checks:

```bash
pytest tests/unit/adapters -v
pytest tests/unit/ood -v
```

- Optional integration check for full behavior:

```bash
pytest tests/integration/test_full_pipeline.py -v --runintegration
```

## Related Status Docs

- `docs/reports/v55/V55_FINAL_STATUS_REPORT.md`
- `docs/reports/v55/V55_CRITICAL_FIXES_SUMMARY.md`
- `docs/reports/v55/V55_SPECIFICATION_AUDIT.md`
