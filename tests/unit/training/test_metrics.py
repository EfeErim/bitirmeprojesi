import json

from src.training.services.metrics import compute_plan_metrics, load_plan_targets
from src.training.services.reporting import persist_validation_artifacts


def test_load_plan_targets_reads_extended_metric_targets(tmp_path):
    spec_path = tmp_path / "metric_targets.json"
    spec_path.write_text(
        json.dumps(
            {
                "targets": {
                    "accuracy": 0.88,
                    "ood_auroc": 0.91,
                    "ood_false_positive_rate": 0.04,
                    "sure_ds_f1": 0.87,
                    "conformal_empirical_coverage": 0.94,
                }
            }
        ),
        encoding="utf-8",
    )

    targets = load_plan_targets(spec_path)

    assert targets == {
        "accuracy": 0.88,
        "ood_auroc": 0.91,
        "ood_false_positive_rate": 0.04,
        "sure_ds_f1": 0.87,
        "conformal_empirical_coverage": 0.94,
    }


def test_compute_plan_metrics_reports_fpr_at_95_tpr():
    perfect = compute_plan_metrics(
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        ood_labels=[0, 0, 0, 0, 1, 1, 1, 1],
        ood_scores=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
    )
    inverted = compute_plan_metrics(
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        ood_labels=[0, 0, 0, 0, 1, 1, 1, 1],
        ood_scores=[0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4],
    )

    assert perfect["ood_false_positive_rate"] == 0.0
    assert inverted["ood_false_positive_rate"] == 1.0


def test_persist_validation_artifacts_records_extended_gate_metrics(tmp_path):
    result = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        sure_ds_f1=0.91,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.25,
        gate_targets={
            "accuracy": 0.80,
            "ood_auroc": 0.90,
            "ood_false_positive_rate": 0.10,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
        },
    )

    assert result["metric_gate"]["metrics"]["sure_ds_f1"] == 0.91
    assert result["metric_gate"]["metrics"]["conformal_empirical_coverage"] == 0.97
    assert result["metric_gate"]["metrics"]["conformal_avg_set_size"] == 1.25
    assert result["metric_gate"]["evaluation"]["checks"]["sure_ds_f1"]["passed"] is True
    assert result["metric_gate"]["evaluation"]["checks"]["conformal_empirical_coverage"]["passed"] is True
