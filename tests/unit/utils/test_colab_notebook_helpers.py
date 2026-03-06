from pathlib import Path

from scripts.colab_notebook_helpers import persist_validation_artifacts


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
