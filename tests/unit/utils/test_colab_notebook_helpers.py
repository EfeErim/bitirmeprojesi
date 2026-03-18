from datetime import datetime
from pathlib import Path

from scripts.colab_checkpointing import TrainingCheckpointManager
from scripts.colab_notebook_helpers import (
    NotebookTrainingStatusPrinter,
    build_notebook_completion_report,
    ensure_notebook_checkpoint_manager,
    maybe_auto_disconnect_colab_runtime,
    persist_validation_artifacts,
)


def test_persist_validation_artifacts_writes_metric_gate(tmp_path: Path):
    result = persist_validation_artifacts(
        root=tmp_path,
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 1, 1],
        classes=["healthy", "disease_a"],
        telemetry=None,
        require_ood=False,
        context={"crop": "tomato"},
    )

    metric_gate_path = result["paths"]["metric_gate_json"]
    assert metric_gate_path.exists()
    assert result["metric_gate"]["evaluation"]["gating"]["status"] in {"soft", "ready"}


def test_persist_validation_artifacts_supports_custom_artifact_subdir(tmp_path: Path):
    result = persist_validation_artifacts(
        root=tmp_path,
        y_true=[0, 1],
        y_pred=[0, 1],
        classes=["healthy", "disease_a"],
        artifact_subdir="test",
        context={"crop": "tomato", "split_name": "test"},
    )

    assert result["paths"]["report_txt"].exists()
    assert result["paths"]["report_txt"].parent.name == "test"


def test_ensure_notebook_checkpoint_manager_returns_existing_instance(tmp_path: Path):
    existing = TrainingCheckpointManager(tmp_path / "telemetry" / "run_1", retention=2)

    resolved = ensure_notebook_checkpoint_manager(existing, run_id="ignored", drive_root=tmp_path / "ignored")

    assert resolved is existing


def test_ensure_notebook_checkpoint_manager_builds_manager_when_missing(tmp_path: Path):
    resolved = ensure_notebook_checkpoint_manager(
        None,
        run_id="run_2",
        drive_root=tmp_path,
        retention=4,
    )

    assert isinstance(resolved, TrainingCheckpointManager)
    assert resolved.root_dir == tmp_path / "telemetry" / "run_2"
    assert resolved.retention == 4


def test_ensure_notebook_checkpoint_manager_generates_microsecond_run_ids(tmp_path: Path, monkeypatch):
    class _FakeDateTime:
        values = [
            datetime(2026, 3, 11, 12, 0, 0, 1),
            datetime(2026, 3, 11, 12, 0, 0, 2),
        ]

        @classmethod
        def now(cls):
            return cls.values.pop(0)

    monkeypatch.setattr("scripts.colab_notebook_helpers.datetime", _FakeDateTime)

    first = ensure_notebook_checkpoint_manager(None, drive_root=tmp_path)
    second = ensure_notebook_checkpoint_manager(None, drive_root=tmp_path)

    assert first.root_dir.name == "20260311_120000_000001"
    assert second.root_dir.name == "20260311_120000_000002"


def test_notebook_training_status_printer_throttles_batch_updates():
    lines = []
    printer = NotebookTrainingStatusPrinter(
        total_epochs=5,
        batch_interval=50,
        min_interval_sec=15.0,
        print_fn=lines.append,
    )

    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 1,
            "total_batches": 100,
            "loss": 0.8,
            "lr": 0.0003,
            "samples_per_sec": 42.5,
            "elapsed_sec": 2.0,
            "eta_sec": 100.0,
        },
    )
    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 2,
            "total_batches": 100,
            "loss": 0.7,
            "lr": 0.0003,
            "samples_per_sec": 43.0,
            "elapsed_sec": 8.0,
            "eta_sec": 94.0,
        },
    )
    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 3,
            "total_batches": 100,
            "loss": 0.6,
            "lr": 0.0003,
            "samples_per_sec": 44.0,
            "elapsed_sec": 18.0,
            "eta_sec": 84.0,
        },
    )
    printer.handle(
        "batch_end",
        {
            "epoch": 1,
            "batch": 50,
            "total_batches": 100,
            "loss": 0.5,
            "lr": 0.0003,
            "samples_per_sec": 45.0,
            "elapsed_sec": 20.0,
            "eta_sec": 80.0,
        },
    )

    assert len(lines) == 3
    assert lines[0].startswith("[LIVE] 1/5 batch=1/100")
    assert "elapsed=2s" in lines[0]
    assert lines[1].startswith("[LIVE] 1/5 batch=3/100")
    assert "eta=1m24s" in lines[1]
    assert lines[2].startswith("[LIVE] 1/5 batch=50/100")


def test_notebook_training_status_printer_emits_validation_best_and_stop():
    lines = []
    printer = NotebookTrainingStatusPrinter(total_epochs=4, print_fn=lines.append)

    printer.handle(
        "validation_end",
        {
            "epoch_done": 2,
            "val_loss": 0.21,
            "val_accuracy": 0.93,
            "macro_f1": 0.91,
            "balanced_accuracy": 0.92,
            "generalization_gap": 0.04,
        },
    )
    printer.handle(
        "best_metric_updated",
        {"epoch_done": 2, "best_metric_name": "val_loss", "best_metric_value": 0.21},
    )
    printer.handle(
        "stop_requested",
        {"epoch": 3, "global_step": 120, "reason": "early_stopping"},
    )

    assert lines == [
        "[VALID] 2/4 val_loss=0.2100 val_acc=0.9300 macro_f1=0.9100 bal_acc=0.9200 gap=0.0400",
        "[BEST] 2/4 val_loss=0.2100",
        "[STOP] epoch=3 step=120 reason=early_stopping",
    ]


class _FakeTelemetry:
    def __init__(self, summary_path: Path):
        self.local_summary_path = summary_path
        self.latest_payloads = []
        self.sync_calls = 0

    def update_latest(self, payload, **_kwargs):
        self.latest_payloads.append(dict(payload))

    def sync_pending(self):
        self.sync_calls += 1


def test_build_notebook_completion_report_marks_ready_when_outputs_exist(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    notebook_export_path = tmp_path / "notebooks" / "executed.ipynb"
    notebook_export_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_export_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "production_readiness": {"status": "failed", "ood_evidence_source": "held_out_benchmark"},
        },
        telemetry=_FakeTelemetry(summary_path),
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_path,
    )

    assert report["ready"] is True
    assert report["missing"] == []
    assert report["soft_missing"] == []
    assert report["evaluation_splits"] == ["test"]
    assert report["production_readiness_status"] == "failed"


def test_build_notebook_completion_report_treats_missing_notebook_export_as_soft_missing(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "production_readiness": {"status": "ready", "ood_evidence_source": "real_ood_split"},
        },
        telemetry=_FakeTelemetry(summary_path),
        repo_run_exports=repo_run_exports,
        notebook_export_path=tmp_path / "missing.ipynb",
    )

    assert report["ready"] is True
    assert report["missing"] == []
    assert report["soft_missing"] == ["executed_notebook_export"]
    assert report["checks"]["executed_notebook_export"] is False


def test_maybe_auto_disconnect_colab_runtime_calls_unassign_when_ready(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    notebook_export_path = tmp_path / "executed.ipynb"
    notebook_export_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    telemetry = _FakeTelemetry(summary_path)
    completion_report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "production_readiness": {"status": "ready"},
        },
        telemetry=telemetry,
        repo_run_exports=repo_run_exports,
        notebook_export_path=notebook_export_path,
    )

    calls = []

    class _FakeRuntime:
        def unassign(self):
            calls.append("unassign")

    monkeypatch.setattr(
        "scripts.colab_notebook_helpers._resolve_colab_runtime_api",
        lambda: _FakeRuntime(),
    )

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=True,
        grace_period_sec=0.0,
        telemetry=telemetry,
        completion_report=completion_report,
        print_fn=lines.append,
    )

    assert result["disconnect_requested"] is True
    assert calls == ["unassign"]
    assert telemetry.sync_calls == 1
    assert telemetry.latest_payloads[-1]["phase"] == "auto_disconnect_pending"
    assert lines == ["[COLAB] Work complete. Disconnecting runtime now to avoid idle credit use."]


def test_maybe_auto_disconnect_colab_runtime_skips_when_checks_are_incomplete(tmp_path: Path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    telemetry = _FakeTelemetry(summary_path)
    monkeypatch.setattr(
        "scripts.colab_notebook_helpers._resolve_colab_runtime_api",
        lambda: None,
    )

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=True,
        grace_period_sec=0.0,
        state={"evaluation_artifacts": {}, "production_readiness": {}},
        telemetry=telemetry,
        repo_run_exports={},
        notebook_export_path=tmp_path / "missing.ipynb",
        print_fn=lines.append,
    )

    assert result["ready"] is False
    assert result["disconnect_requested"] is not True
    assert telemetry.sync_calls == 1
    assert telemetry.latest_payloads[-1]["phase"] == "auto_disconnect_skipped"
    assert telemetry.latest_payloads[-1]["completion_missing"] == [
        "evaluation_artifacts",
        "production_readiness",
        "repo_exports",
    ]
    assert telemetry.latest_payloads[-1]["completion_soft_missing"] == ["executed_notebook_export"]
    assert lines == [
        "[COLAB] Auto-disconnect skipped. Incomplete required checks: evaluation_artifacts, production_readiness, repo_exports",
        "[COLAB] Soft-missing checks: executed_notebook_export",
    ]


def test_maybe_auto_disconnect_colab_runtime_proceeds_when_only_notebook_export_is_soft_missing(
    tmp_path: Path,
    monkeypatch,
):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    repo_run_exports = {}
    for name in ("outputs", "telemetry", "checkpoint_state"):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        repo_run_exports[name] = str(path)

    telemetry = _FakeTelemetry(summary_path)
    completion_report = build_notebook_completion_report(
        state={
            "evaluation_artifacts": {"test": {"metric_gate": {}}},
            "production_readiness": {"status": "ready"},
        },
        telemetry=telemetry,
        repo_run_exports=repo_run_exports,
        notebook_export_path=tmp_path / "missing.ipynb",
    )

    calls = []

    class _FakeRuntime:
        def unassign(self):
            calls.append("unassign")

    monkeypatch.setattr(
        "scripts.colab_notebook_helpers._resolve_colab_runtime_api",
        lambda: _FakeRuntime(),
    )

    lines = []
    result = maybe_auto_disconnect_colab_runtime(
        enabled=True,
        grace_period_sec=0.0,
        telemetry=telemetry,
        completion_report=completion_report,
        print_fn=lines.append,
    )

    assert result["ready"] is True
    assert result["soft_missing"] == ["executed_notebook_export"]
    assert result["disconnect_requested"] is True
    assert calls == ["unassign"]
    assert telemetry.latest_payloads[-1]["phase"] == "auto_disconnect_pending"
    assert telemetry.latest_payloads[-1]["completion_soft_missing"] == ["executed_notebook_export"]
    assert lines == [
        "[COLAB] Proceeding despite soft-missing checks: executed_notebook_export",
        "[COLAB] Work complete. Disconnecting runtime now to avoid idle credit use.",
    ]
