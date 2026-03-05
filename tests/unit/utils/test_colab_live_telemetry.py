import json

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
        heartbeat_sec=1.0,
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
        heartbeat_sec=1.0,
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
