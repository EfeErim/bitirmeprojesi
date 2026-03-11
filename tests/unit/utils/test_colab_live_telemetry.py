import json
import re
import sys
from datetime import datetime

import pytest

from scripts.colab_live_telemetry import ColabLiveTelemetry


def test_live_telemetry_writes_events_logs_and_artifacts(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_001",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )
    telemetry.emit_event("train_batch", {"global_step": 1}, phase="train")
    telemetry.emit_log("batch completed", phase="train", level="info")
    telemetry.update_latest({"epoch": 1, "batch": 1})
    telemetry.write_json_artifact("training/history.json", {"train_loss": [0.1]})
    telemetry.close({"status": "ok"})

    run_dir = drive_root / "telemetry" / "run_001"
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "runtime.log").exists()
    assert (run_dir / "latest_status.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "artifact_index.json").exists()
    assert (run_dir / "artifacts" / "training" / "history.json").exists()


def test_live_telemetry_recovers_after_drive_write_failure(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_002",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    def _fail_sync_events(*args, **kwargs):
        raise OSError("drive unavailable")

    monkeypatch.setattr(telemetry, "_sync_events", _fail_sync_events)
    telemetry.emit_event("train_batch", {"global_step": 1}, phase="train")
    local_events = telemetry.local_events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(local_events) >= 2  # includes run_started + train_batch

    monkeypatch.undo()
    telemetry.sync_pending()
    drive_events = telemetry.drive_events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(drive_events) >= 2
    telemetry.close({"status": "ok"})

    sync_state = json.loads(telemetry.local_sync_state_path.read_text(encoding="utf-8"))
    assert sync_state["events_synced"] >= 2


def test_live_telemetry_throttles_batch_latest_status_writes(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_002b",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=60.0,
        latest_status_min_interval_sec=15.0,
    )

    clock = {"value": 1.0}
    monkeypatch.setattr("scripts.colab_live_telemetry.time.monotonic", lambda: clock["value"])

    telemetry.update_latest({"event_type": "batch_end", "epoch": 1, "batch": 1})
    latest_path = drive_root / "telemetry" / "run_002b" / "latest_status.json"
    first = json.loads(latest_path.read_text(encoding="utf-8"))
    assert first["status"]["batch"] == 1

    clock["value"] = 5.0
    telemetry.update_latest({"event_type": "batch_end", "epoch": 1, "batch": 2})
    second = json.loads(latest_path.read_text(encoding="utf-8"))
    assert second["status"]["batch"] == 1

    clock["value"] = 20.0
    telemetry.update_latest({"event_type": "batch_end", "epoch": 1, "batch": 3})
    third = json.loads(latest_path.read_text(encoding="utf-8"))
    assert third["status"]["batch"] == 3


def test_capture_cell_output_writes_timestamped_drive_artifact(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_003",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    with telemetry.capture_cell_output("Cell 3: Parameters") as artifact_path:
        print("hello from stdout")
        print("warning from stderr", file=sys.stderr)

    assert artifact_path.exists()
    assert re.search(r"cell_outputs[\\/]\d{8}_\d{6}_\d{6}_cell_3_parameters\.log$", str(artifact_path))

    body = artifact_path.read_text(encoding="utf-8")
    assert "[CELL_CAPTURE] cell=Cell 3: Parameters" in body
    assert "run_id=run_003" in body
    assert "started_at=" in body
    assert "finished_at=" in body
    assert "duration_sec=" in body
    assert "hello from stdout" in body
    assert "[STDERR]" in body
    assert "warning from stderr" in body


def test_capture_cell_output_persists_output_on_exception(tmp_path):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_004",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )

    with pytest.raises(RuntimeError, match="boom"):
        with telemetry.capture_cell_output("Cell 4: Failure"):
            print("about to fail")
            raise RuntimeError("boom")

    artifacts = list((drive_root / "telemetry" / "run_004" / "artifacts" / "cell_outputs").glob("*.log"))
    assert len(artifacts) == 1
    body = artifacts[0].read_text(encoding="utf-8")
    assert "about to fail" in body
    assert "[EXCEPTION] RuntimeError: boom" in body


def test_live_telemetry_spools_locally_when_drive_mount_is_missing(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"
    monkeypatch.setattr("scripts.colab_live_telemetry._requires_google_drive_mount", lambda path: True)
    monkeypatch.setattr("scripts.colab_live_telemetry._google_drive_is_mounted", lambda: False)

    telemetry = ColabLiveTelemetry(
        notebook_name="nb2",
        run_id="run_005",
        drive_root=drive_root,
        local_root=local_root,
        sync_interval_sec=0.1,
    )
    telemetry.emit_event("train_batch", {"global_step": 1}, phase="train")
    telemetry.write_json_artifact("training/history.json", {"train_loss": [0.1]})
    telemetry.close({"status": "ok"})

    drive_run_dir = drive_root / "telemetry" / "run_005"
    local_run_dir = local_root / "run_005"
    assert not drive_run_dir.exists()
    assert (local_run_dir / "events.jsonl").exists()
    assert (local_run_dir / "summary.json").exists()
    assert (local_run_dir / "artifacts" / "training" / "history.json").exists()


def test_live_telemetry_auto_run_ids_do_not_collide_within_same_second(tmp_path, monkeypatch):
    drive_root = tmp_path / "drive"
    local_root = tmp_path / "local"

    class _FakeDateTime:
        values = [
            datetime(2026, 3, 11, 12, 0, 0, 1),
            datetime(2026, 3, 11, 12, 0, 0, 2),
        ]

        @classmethod
        def now(cls, tz=None):
            value = cls.values.pop(0) if cls.values else datetime(2026, 3, 11, 12, 0, 0, 3)
            if tz is not None:
                return value.replace(tzinfo=tz)
            return value

    monkeypatch.setattr("scripts.colab_live_telemetry.datetime", _FakeDateTime)
    monkeypatch.setattr("scripts.colab_live_telemetry._utc_now_iso", lambda: "2026-03-11T12:00:00Z")

    first = ColabLiveTelemetry(notebook_name="nb2", drive_root=drive_root, local_root=local_root)
    second = ColabLiveTelemetry(notebook_name="nb2", drive_root=drive_root, local_root=local_root)

    assert first.run_id == "20260311_120000_000001"
    assert second.run_id == "20260311_120000_000002"
