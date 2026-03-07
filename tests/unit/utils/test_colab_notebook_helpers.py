from pathlib import Path

from scripts.colab_checkpointing import TrainingCheckpointManager
from scripts.colab_notebook_helpers import (
    NotebookTrainingStatusPrinter,
    ensure_notebook_checkpoint_manager,
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
