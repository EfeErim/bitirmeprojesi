from pathlib import Path

from src.training.services.reporting import (
    persist_batch_metrics_artifacts,
    persist_training_history_artifacts,
    persist_training_results_figure,
    persist_training_summary_artifact,
    persist_validation_artifacts,
)


def test_reporting_writes_training_and_validation_artifacts(tmp_path: Path):
    artifact_root = tmp_path / "training_metrics"
    history_snapshot = {
        "train_loss": [0.8, 0.4],
        "val_loss": [0.9, 0.5],
        "val_accuracy": [0.4, 0.75],
        "macro_precision": [0.3, 0.7],
        "macro_recall": [0.35, 0.72],
        "macro_f1": [0.32, 0.71],
        "weighted_f1": [0.34, 0.74],
        "balanced_accuracy": [0.35, 0.72],
        "generalization_gap": [0.1, 0.1],
        "best_metric_name": "val_loss",
        "best_metric_value": 0.5,
        "best_epoch": 2,
    }
    batch_history = [
        {"epoch": 1, "batch": 1, "global_step": 1, "optimizer_steps": 1, "loss": 0.8, "lr": 1e-3},
        {"epoch": 1, "batch": 2, "global_step": 2, "optimizer_steps": 2, "loss": 0.6, "lr": 8e-4},
    ]

    training_paths = persist_training_history_artifacts(
        artifact_root=artifact_root,
        history_snapshot=history_snapshot,
    )
    batch_paths = persist_batch_metrics_artifacts(
        artifact_root=artifact_root,
        batch_history=batch_history,
    )
    figure_paths = persist_training_results_figure(
        artifact_root=artifact_root,
        history_snapshot=history_snapshot,
        batch_history=batch_history,
    )
    summary_paths = persist_training_summary_artifact(
        artifact_root=artifact_root,
        summary_payload={"run_id": "run_1", "crop_name": "tomato"},
    )
    validation_paths = persist_validation_artifacts(
        artifact_root=artifact_root,
        y_true=[0, 0],
        y_pred=[0, 0],
        classes=["healthy", "disease_a"],
        context={"crop_name": "tomato"},
    )

    assert training_paths["history_json"].exists()
    assert training_paths["results_csv"].exists()
    assert batch_paths["batch_metrics_csv"].exists()
    assert figure_paths["results_png"].exists()
    assert summary_paths["summary_json"].exists()
    assert validation_paths["paths"]["cm_png"].exists()
    assert validation_paths["paths"]["cm_norm_png"].exists()
    assert validation_paths["paths"]["metric_gate_json"].exists()
