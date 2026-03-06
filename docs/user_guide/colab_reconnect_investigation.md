# Colab Reconnect Investigation (Notebook 2)

Use this guide when `colab_notebooks/2_interactive_adapter_training.ipynb` appears stuck on `Connecting...` after closing and reopening the browser tab.

## Scope

- Root-cause diagnosis only.
- No training logic or runtime policy changes.
- Evidence is stored under the same telemetry run folder.

## Prerequisites

1. Run Cell 1 and Cell 2 of `colab_notebooks/2_interactive_adapter_training.ipynb`.
2. Ensure `TELEMETRY` and `CHECKPOINT_MANAGER` are initialized.
3. Keep canonical helper path: `scripts/colab_reconnect_probe.py`.

## Preflight Identity Cell

Run this cell after Cell 2:

```python
from scripts.colab_reconnect_probe import capture_preflight_identity

preflight = capture_preflight_identity(
    state=STATE,
    telemetry=TELEMETRY,
    checkpoint_manager=CHECKPOINT_MANAGER,
    browser_info="Chrome <version> / Edge <version>",
    extra={
        "colab_gpu_tier": "Pro+",
        "investigation_date": "2026-03-05",
    },
)
print(preflight["run_id"])
print(preflight["drive_run_dir"])
```

Expected artifact:
- `telemetry/<run_id>/investigation/preflight_identity.json`

## Evidence Probe Cell (Before Training Cell)

Run this cell before the training cell:

```python
from scripts.colab_reconnect_probe import ReconnectEvidenceProbe, log_trial_timestamp

probe = ReconnectEvidenceProbe.from_runtime(
    state=STATE,
    telemetry=TELEMETRY,
    checkpoint_manager=CHECKPOINT_MANAGER,
    interval_sec=15.0,
    echo=True,
)
probe.start()
probe.snapshot_once()  # immediate baseline sample
```

What it captures every 15 seconds:
- `events.jsonl` line count
- `latest_status.json` mtime + latest phase/epoch/batch/global_step
- `latest_checkpoint.json` epoch/global_step/name

Expected artifacts:
- `telemetry/<run_id>/investigation/reconnect_probe.jsonl`
- `telemetry/<run_id>/investigation/trial_timestamps.jsonl`

## Trial Timestamp Logging

Use these calls during each reconnect experiment:

```python
from datetime import datetime, timezone

close_ts = datetime.now(timezone.utc).isoformat()
log_trial_timestamp(
    probe,
    label="trial_a_2min_normal_window",
    close_ts_utc=close_ts,
    ui_state="tab_closed",
)
```

After reopening:

```python
reopen_ts = datetime.now(timezone.utc).isoformat()
log_trial_timestamp(
    probe,
    label="trial_a_2min_normal_window",
    reopen_ts_utc=reopen_ts,
    first_reconnect_ts_utc="",  # fill when connection is restored
    ui_state="connecting_stuck",
    ram_disk_busy=True,         # set based on Colab RAM/Disk panel once visible
    note="UI kept showing Connecting",
)
```

When reconnect succeeds, append:

```python
first_ok_ts = datetime.now(timezone.utc).isoformat()
log_trial_timestamp(
    probe,
    label="trial_a_2min_normal_window",
    first_reconnect_ts_utc=first_ok_ts,
    ui_state="reconnected",
)
```

## Required Trial Set

1. `trial_a_2min_normal_window`
2. `trial_b_10min_normal_window`
3. `trial_c_2min_incognito`

Optional control:
4. `trial_d_reduced_output` (lower visible UI churn)

## Stop Probe and Final Snapshot

After each trial block:

```python
probe.stop()
probe.snapshot_once()
```

## Diagnosis Rules

Classify with artifact evidence:

- `frontend_attach_issue`:
  - `reconnect_probe.jsonl` keeps increasing `events_lines` and `latest_global_step`
  - checkpoints continue to advance
  - browser UI remains on `Connecting...`
- `runtime_stopped_or_restarted`:
  - event lines and status/checkpoint values stop advancing
  - runtime process no longer active
- `drive_sync_issue`:
  - local evidence advances but Drive-run artifacts lag or fail
  - `latest_status_mtime_utc` is stale on Drive side

## Resume Integrity Check

After forced detach/reconnect:

1. Set notebook resume mode to `Resume latest`.
2. Rerun training cell.
3. Confirm resumed epoch/global_step matches latest checkpoint manifest.
4. Confirm no backward jump in `global_step`.

