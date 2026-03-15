import json

from src.training.services.metrics import (
    build_production_readiness,
    compute_plan_metrics,
    load_plan_targets,
)
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
                    "ood_samples": 7,
                    "in_distribution_samples": 9,
                    "sure_ds_f1": 0.87,
                    "conformal_empirical_coverage": 0.94,
                    "conformal_avg_set_size": 1.5,
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
        "ood_samples": 7.0,
        "in_distribution_samples": 9.0,
        "sure_ds_f1": 0.87,
        "conformal_empirical_coverage": 0.94,
        "conformal_avg_set_size": 1.5,
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


def test_build_production_readiness_passes_with_real_ood_evidence(tmp_path):
    validation = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        ood_labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ood_scores=[0.1, 0.2, 0.15, 0.18, 0.22, 0.8, 0.9, 0.82, 0.87, 0.92],
        sure_ds_f1=0.93,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.25,
        gate_targets={
            "accuracy": 0.80,
            "ood_auroc": 0.90,
            "ood_false_positive_rate": 0.10,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
        },
        require_ood=True,
    )

    readiness = build_production_readiness(
        classification_metric_gate=validation["metric_gate"],
        classification_split="test",
        ood_evidence_source="real_ood_split",
        ood_metrics=validation["metric_gate"]["metrics"],
        targets={
            "accuracy": 0.80,
            "ood_auroc": 0.90,
            "ood_false_positive_rate": 0.10,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
        },
    )

    assert readiness["status"] == "ready"
    assert readiness["passed"] is True
    assert readiness["ood_evidence_source"] == "real_ood_split"
    assert readiness["missing_requirements"] == []
    assert readiness["ood_evidence"]["metrics"]["ood_samples"] == 5
    assert readiness["ood_evidence"]["metrics"]["in_distribution_samples"] == 5
    assert readiness["ood_evidence"]["metrics"]["conformal_avg_set_size"] == 1.25


def test_build_production_readiness_allows_missing_ood_evidence_when_not_required(tmp_path):
    validation = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
    )

    readiness = build_production_readiness(
        classification_metric_gate=validation["metric_gate"],
        classification_split="test",
        ood_evidence_source="unavailable",
        ood_metrics={},
        require_ood=False,
    )

    assert readiness["status"] == "ready"
    assert readiness["passed"] is True
    assert readiness["ood_evidence"]["evaluation"]["require_ood"] is False
    assert readiness["missing_requirements"] == []


def test_build_production_readiness_fails_when_ood_evidence_is_missing(tmp_path):
    validation = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
    )

    readiness = build_production_readiness(
        classification_metric_gate=validation["metric_gate"],
        classification_split="test",
        ood_evidence_source="unavailable",
        ood_metrics={},
        require_ood=True,
    )

    assert readiness["status"] == "failed"
    assert readiness["passed"] is False
    assert "ood_auroc" in readiness["missing_requirements"]
    assert "ood_false_positive_rate" in readiness["missing_requirements"]
    assert "ood_samples" in readiness["missing_requirements"]
    assert "in_distribution_samples" in readiness["missing_requirements"]
    assert "conformal_avg_set_size" in readiness["missing_requirements"]


def test_build_production_readiness_fails_when_ood_sample_counts_are_too_small(tmp_path):
    validation = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        ood_labels=[0, 1],
        ood_scores=[0.1, 0.9],
        sure_ds_f1=0.93,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.0,
        require_ood=True,
    )

    readiness = build_production_readiness(
        classification_metric_gate=validation["metric_gate"],
        classification_split="test",
        ood_evidence_source="real_ood_split",
        ood_metrics=validation["metric_gate"]["metrics"],
        targets={
            "accuracy": 0.80,
            "ood_auroc": 0.90,
            "ood_false_positive_rate": 0.10,
            "ood_samples": 5,
            "in_distribution_samples": 5,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
            "conformal_avg_set_size": 2.0,
        },
    )

    assert readiness["status"] == "failed"
    assert readiness["passed"] is False
    assert "ood_samples" in readiness["missing_requirements"]
    assert "in_distribution_samples" in readiness["missing_requirements"]


def test_build_production_readiness_fails_when_conformal_sets_are_too_large(tmp_path):
    validation = persist_validation_artifacts(
        artifact_root=tmp_path / "training_metrics",
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        classes=["healthy", "disease_a"],
        ood_labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ood_scores=[0.1, 0.2, 0.15, 0.18, 0.22, 0.8, 0.9, 0.82, 0.87, 0.92],
        sure_ds_f1=0.93,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=3.5,
        require_ood=True,
    )

    readiness = build_production_readiness(
        classification_metric_gate=validation["metric_gate"],
        classification_split="test",
        ood_evidence_source="real_ood_split",
        ood_metrics=validation["metric_gate"]["metrics"],
        targets={
            "accuracy": 0.80,
            "ood_auroc": 0.90,
            "ood_false_positive_rate": 0.10,
            "ood_samples": 5,
            "in_distribution_samples": 5,
            "sure_ds_f1": 0.90,
            "conformal_empirical_coverage": 0.95,
            "conformal_avg_set_size": 2.0,
        },
    )

    assert readiness["status"] == "failed"
    assert readiness["passed"] is False
    assert "conformal_avg_set_size" in readiness["missing_requirements"]
