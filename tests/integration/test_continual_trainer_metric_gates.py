import json
from pathlib import Path

import pytest

from src.training.continual_sd_lora import ContinualSDLoRATrainer


def test_plan_targets_loaded_from_adapter_spec_contract():
    targets = ContinualSDLoRATrainer.load_plan_targets(Path("specs/adapter-spec.json"))

    assert targets["accuracy"] == pytest.approx(0.93)
    assert targets["ood_auroc"] == pytest.approx(0.92)
    assert targets["ood_false_positive_rate"] == pytest.approx(0.05)


def test_validate_plan_metrics_soft_gate_when_ood_missing():
    metrics = {
        "accuracy": 0.95,
        "ood_auroc": None,
        "ood_false_positive_rate": None,
        "classification_samples": 10,
        "ood_samples": 0,
        "in_distribution_samples": 0,
    }

    result = ContinualSDLoRATrainer.validate_plan_metrics(metrics, require_ood=False)

    assert result["passed"] is True
    assert result["gating"]["status"] == "soft"
    assert set(result["gating"]["missing_metrics"]) == {"ood_auroc", "ood_false_positive_rate"}


def test_validate_plan_metrics_hard_gate_requires_ood_and_fails_when_missing():
    metrics = {
        "accuracy": 0.95,
        "ood_auroc": None,
        "ood_false_positive_rate": None,
        "classification_samples": 10,
        "ood_samples": 0,
        "in_distribution_samples": 0,
    }

    result = ContinualSDLoRATrainer.validate_plan_metrics(metrics, require_ood=True)

    assert result["passed"] is False
    assert result["gating"]["status"] == "failed"
    assert "ood_auroc" in result["gating"]["missing_metrics"]
    assert "ood_false_positive_rate" in result["gating"]["missing_metrics"]


def test_compute_and_write_metric_artifact_with_explicit_fail_conditions(tmp_path):
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 0]  # 5/6 = 0.8333 (below 0.93 target)

    ood_labels = [0, 0, 0, 1, 1, 1]
    ood_scores = [0.10, 0.20, 0.25, 0.85, 0.90, 0.95]  # separable -> high AUROC

    metrics = ContinualSDLoRATrainer.compute_plan_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ood_labels=ood_labels,
        ood_scores=ood_scores,
    )

    artifact_path = tmp_path / "plan_metric_gate.json"
    artifact = ContinualSDLoRATrainer.write_plan_metric_artifact(
        output_path=artifact_path,
        metrics=metrics,
        require_ood=True,
        context={"suite": "integration", "source": "unitized_e2e_gate"},
    )

    assert artifact_path.exists()
    on_disk = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "v6_plan_metric_gate"

    assert artifact["evaluation"]["checks"]["accuracy"]["passed"] is False
    assert artifact["evaluation"]["checks"]["ood_auroc"]["asserted"] is True
    assert artifact["evaluation"]["checks"]["ood_false_positive_rate"]["asserted"] is True
    assert artifact["evaluation"]["passed"] is False

