# Automation Scripts Guide

Quick-win automation improvements added to the repository.

## 1. `cleanup_old_artifacts.py`

Removes old temporary artifacts older than a retention period (default: 90 days). It does not touch `runs/`.

**Usage:**
```powershell
# Dry-run to see what would be deleted
.\scripts\python.cmd scripts/cleanup_old_artifacts.py --dry-run

# Actually delete artifacts older than 90 days
.\scripts\python.cmd scripts/cleanup_old_artifacts.py --days 90

# Custom retention (e.g., 30 days)
.\scripts\python.cmd scripts/cleanup_old_artifacts.py --days 30
```

**What it deletes:**
- `.runtime_tmp/*` files/folders (except caches) modified >N days ago

**CI Integration:**
- Runs daily via GitHub Actions scheduled job (midnight UTC)
- Safe to run manually before large training campaigns

## 2. `check_config_drift.py`

Detects config schema changes and warns if documentation/tests are not updated.

**Usage:**
```powershell
# Check for config inconsistencies
.\scripts\python.cmd scripts/check_config_drift.py
```

**Checks:**
- `base.json` vs `colab.json` schema version mismatch
- New keys in base not in colab
- Missing documentation references
- Provides guidance for schema versioning

**CI Integration:**
- Runs on every push to ensure config consistency
- Prevents silent config drift across environments

## 3. `generate_artifact_manifest.py`

Creates manifest metadata for exported adapters, tracking version, commit, and training info.

**Usage:**
```powershell
# Generate manifest for an adapter
.\scripts\python.cmd scripts/generate_artifact_manifest.py `
  --adapter-root models/adapters/tomato/leaf/continual_sd_lora_adapter `
  --crop tomato `
  --part leaf `
  --dataset-key "dataset_v1::sha256hash" `
  --ood-available `
  --readiness ready `
  --output models/adapters/tomato/leaf/MANIFEST.json
```

**Manifest contents:**
- Git commit hash and branch
- Adapter files availability (weights, classifier, OOD state, config)
- Training metadata (start time, duration, dataset lineage)
- Evaluation status (OOD available, readiness verdict)

**Next Steps:**
- Integrate into `TrainingWorkflow` to auto-generate on export
- Use manifest for deployment traceability

## 4. Enhanced CI Workflow

New validation steps in `.github/workflows/ci.yml`:

**Added checks:**
- **Config drift detection** (`check_config_drift.py`) — After schema validation
- **Dataset layout validation** (`validate_dataset_layout.py`) — On prepared runtime datasets if present
- **Artifact cleanup** (scheduled) — Monthly cleanup job (1st of month at 00:00 UTC)

**Ruff/MyPy now lint:**
- `cleanup_old_artifacts.py`
- `check_config_drift.py`
- `generate_artifact_manifest.py`

## Example: Integration into Training Workflow

In `src/workflows/training.py`, after adapter export:

```python
from scripts.generate_artifact_manifest import build_artifact_manifest, persist_manifest

# After adapter export...
manifest = build_artifact_manifest(
    adapter_bundle_root=exported_adapter_path,
    crop=crop_name,
    part=part_name,
    training_start_time=training_start,
    training_duration_seconds=training_duration,
    dataset_lineage_key=dataset_key,
    ood_evidence_available=ood_calibration_done,
    production_readiness_verdict=readiness_status,
)
persist_manifest(manifest, exported_adapter_path / "MANIFEST.json")
```

## Benefits Summary

| Script | Benefit | Effort |
|--------|---------|--------|
| Cleanup | Saves disk space, reduces clutter | Automated (cron) |
| Config drift | Catches misalignment early | Zero effort (CI gate) |
| Artifact manifest | Deployment traceability | Auto-generated |
| Dataset validation | Prevents wasted training | Zero effort (CI gate) |

## Future Enhancements

- [ ] Integrate artifact manifest generation into `TrainingWorkflow`
- [ ] Add adapter health monitoring (check for OOD calibration artifacts)
- [ ] Router performance regression detection in CI
- [ ] Notebook cell execution testing for Notebooks 0, 2, 3
