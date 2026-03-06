import json
from pathlib import Path

from scripts.colab_reconnect_probe import (
    ReconnectEvidenceProbe,
    capture_preflight_identity,
    log_trial_timestamp,
)


class _FakeTelemetry:
    def __init__(self, run_id: str, drive_run_dir: Path, local_run_dir: Path) -> None:
        self.run_id = run_id
        self.drive_run_dir = drive_run_dir
        self.local_run_dir = local_run_dir


class _FakeCheckpointManager:
    def __init__(self, latest_manifest_path: Path, latest: dict) -> None:
        self.latest_manifest_path = latest_manifest_path
        self._latest = dict(latest)

    def get_latest(self):
        return dict(self._latest)


def test_capture_preflight_identity_persists_payload(tmp_path):
    drive_run_dir = tmp_path / "drive" / "telemetry" / "run_010"
    local_run_dir = tmp_path / "local" / "run_010"
    drive_run_dir.mkdir(parents=True, exist_ok=True)
    local_run_dir.mkdir(parents=True, exist_ok=True)

    latest_manifest = drive_run_dir / "latest_checkpoint.json"
    latest_payload = {"name": "ckpt_latest", "epoch": 3, "global_step": 400}
    latest_manifest.write_text(json.dumps(latest_payload), encoding="utf-8")

    telemetry = _FakeTelemetry("run_010", drive_run_dir=drive_run_dir, local_run_dir=local_run_dir)
    manager = _FakeCheckpointManager(latest_manifest, latest_payload)
    state = {
        "widget_config": {"resume_mode": "resume"},
        "resume_manifest": latest_payload,
        "telemetry_run_dir": str(drive_run_dir),
    }

    payload = capture_preflight_identity(
        state=state,
        telemetry=telemetry,
        checkpoint_manager=manager,
        browser_info="Chrome 125",
        extra={"gpu_runtime": "A100"},
    )

    assert payload["run_id"] == "run_010"
    assert payload["resume_mode"] == "resume"
    assert payload["latest_checkpoint_meta"]["global_step"] == 400
    assert payload["browser_info"] == "Chrome 125"

    drive_file = drive_run_dir / "investigation" / "preflight_identity.json"
    local_file = local_run_dir / "investigation" / "preflight_identity.json"
    assert drive_file.exists()
    assert local_file.exists()


def test_probe_snapshot_and_trial_event_logging(tmp_path):
    drive_run_dir = tmp_path / "drive" / "telemetry" / "run_011"
    local_run_dir = tmp_path / "local" / "run_011"
    drive_run_dir.mkdir(parents=True, exist_ok=True)
    local_run_dir.mkdir(parents=True, exist_ok=True)

    events_path = drive_run_dir / "events.jsonl"
    events_path.write_text('{"event":"run_started"}\n{"event":"train_batch"}\n', encoding="utf-8")

    latest_status = {
        "status": {"phase": "training", "epoch": 2, "batch": 5, "global_step": 205},
    }
    (drive_run_dir / "latest_status.json").write_text(json.dumps(latest_status), encoding="utf-8")
    latest_manifest = {"name": "ckpt_e2", "epoch": 2, "global_step": 200}
    latest_manifest_path = drive_run_dir / "latest_checkpoint.json"
    latest_manifest_path.write_text(json.dumps(latest_manifest), encoding="utf-8")

    probe = ReconnectEvidenceProbe(
        drive_run_dir=drive_run_dir,
        local_run_dir=local_run_dir,
        latest_manifest_path=latest_manifest_path,
        interval_sec=15.0,
        echo=False,
    )
    snapshot = probe.snapshot_once()
    assert snapshot["events_lines"] == 2
    assert snapshot["latest_phase"] == "training"
    assert snapshot["latest_global_step"] == 205
    assert snapshot["checkpoint_global_step"] == 200

    record = log_trial_timestamp(
        probe,
        label="trial_a",
        close_ts_utc="2026-03-05T12:00:00Z",
        reopen_ts_utc="2026-03-05T12:02:00Z",
        first_reconnect_ts_utc="",
        ui_state="connecting_stuck",
        ram_disk_busy=True,
        note="ui stayed in connecting",
    )
    assert record["event_type"] == "trial_timestamp"
    assert record["payload"]["label"] == "trial_a"
    assert record["payload"]["ram_disk_busy"] is True

    drive_probe_log = drive_run_dir / "investigation" / "reconnect_probe.jsonl"
    local_probe_log = local_run_dir / "investigation" / "reconnect_probe.jsonl"
    trial_log = drive_run_dir / "investigation" / "trial_timestamps.jsonl"
    assert drive_probe_log.exists()
    assert local_probe_log.exists()
    assert trial_log.exists()
