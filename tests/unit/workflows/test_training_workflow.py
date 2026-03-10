from pathlib import Path

from src.training.types import EvaluationArtifactsPayload, ValidationReport
from src.workflows.training import TrainingWorkflow, TrainingWorkflowResult


class FakeDataset:
    def __init__(self, classes):
        self.classes = list(classes)

    def __len__(self):
        return len(self.classes)


class FakeLoader(list):
    def __init__(self, classes):
        super().__init__([{"images": 1, "labels": 1}])
        self.dataset = FakeDataset(classes)


class FakeHistory:
    def to_dict(self):
        return {"train_loss": [0.1], "val_loss": [0.2], "global_step": 1}


class FakeSession:
    def __init__(self, observers, trainer=None):
        self.observers = list(observers)
        self.trainer = trainer if trainer is not None else object()

    def snapshot_state(self):
        return {"progress_state": {"epoch": 1, "global_step": 2}, "history": {"train_loss": [0.1]}}

    def run(self):
        for observer in self.observers:
            observer(
                {
                    "event_type": "checkpoint_requested",
                    "payload": {"reason": "batch_interval", "mark_best": False, "val_loss": 0.2},
                }
            )
        return FakeHistory()


class FakeAdapter:
    def __init__(self, crop_name, model_name="model", device="cpu"):
        self.crop_name = crop_name
        self.model_name = model_name
        self.device = device
        self.initialized = None

    def initialize_engine(self, *, class_names=None, config=None):
        self.initialized = {"class_names": list(class_names or []), "config": dict(config or {})}
        return {"status": "initialized"}

    def build_training_session(self, train_loader, **kwargs):
        evaluation = (
            self.initialized.get("config", {})
            .get("training", {})
            .get("continual", {})
            .get("evaluation", {})
        )
        trainer_config = type(
            "FakeTrainerConfig",
            (),
            {
                "evaluation_require_ood_for_gate": bool(evaluation.get("require_ood_for_gate", True)),
                "evaluation_emit_ood_gate": bool(evaluation.get("emit_ood_gate", True)),
            },
        )()
        trainer = type("FakeTrainer", (), {"config": trainer_config})()
        return FakeSession(kwargs.get("observers", []), trainer=trainer)

    def calibrate_ood(self, loader):
        return {"status": "calibrated", "ood_calibration": {"version": 1}}

    def save_adapter(self, output_dir):
        path = Path(output_dir) / "continual_sd_lora_adapter"
        path.mkdir(parents=True, exist_ok=True)
        return path


class FakeCheckpointManager:
    def __init__(self):
        self.calls = []

    def save_checkpoint(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"name": "ckpt_1", "reason": kwargs["reason"]}


def _fake_evaluation_result(include_ood: bool) -> EvaluationArtifactsPayload:
    report = ValidationReport(
        val_loss=0.05,
        val_accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        balanced_accuracy=1.0,
        per_class_accuracy={"healthy": 1.0, "disease_a": 1.0},
        per_class_support={"healthy": 2, "disease_a": 2},
        worst_classes=[],
    )
    return EvaluationArtifactsPayload(
        report=report,
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        ood_labels=[0, 0, 1, 1] if include_ood else None,
        ood_scores=[0.1, 0.2, 0.8, 0.9] if include_ood else None,
        sure_ds_f1=0.95,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.0,
    )


def test_training_workflow_runs_adapter_session_and_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy"]),
            "test": FakeLoader(["healthy"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)

    checkpoint_manager = FakeCheckpointManager()
    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {"ood_fallback_strategy": "none", "ood_benchmark_auto_run": False},
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False, "checkpoint_every_n_steps": 1}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
        checkpoint_manager=checkpoint_manager,
    )

    assert result.class_names == ["healthy", "disease_a"]
    assert result.history["train_loss"] == [0.1]
    assert result.adapter_dir.exists()
    assert result.artifact_dir is not None and result.artifact_dir.exists()
    assert (result.artifact_dir / "training" / "results.png").exists()
    assert checkpoint_manager.calls
    assert result.checkpoint_records[0]["reason"] == "batch_interval"


def test_training_workflow_result_to_dict_stringifies_nested_paths(tmp_path: Path):
    result = TrainingWorkflowResult(
        run_id="run_1",
        crop_name="tomato",
        class_names=["healthy"],
        history={"train_loss": [0.1]},
        loader_sizes={"train": 1},
        adapter_dir=tmp_path / "adapter",
        artifact_dir=tmp_path / "artifacts",
        artifacts={"training": {"summary": tmp_path / "artifacts" / "summary.json"}},
        ood_benchmark={"paths": {"report": tmp_path / "artifacts" / "ood.json"}},
        production_readiness={"status": "ready"},
    )

    payload = result.to_dict()

    assert payload["adapter_dir"] == str(tmp_path / "adapter")
    assert payload["artifact_dir"] == str(tmp_path / "artifacts")
    assert payload["artifacts"]["training"]["summary"] == str(tmp_path / "artifacts" / "summary.json")
    assert payload["ood_benchmark"]["paths"]["report"] == str(tmp_path / "artifacts" / "ood.json")


def test_training_workflow_prefers_real_ood_evidence(monkeypatch, tmp_path: Path):
    benchmark_calls = []
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy", "disease_a"]),
            "test": FakeLoader(["healthy", "disease_a"]),
            "ood": FakeLoader(["unknown"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=ood_loader is not None),
    )
    monkeypatch.setattr(
        "src.workflows.training.run_leave_one_class_out_benchmark",
        lambda **kwargs: benchmark_calls.append(dict(kwargs)) or {},
    )

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {"require_ood_for_gate": True},
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert benchmark_calls == []
    assert result.ood_evidence_source == "real_ood_split"
    assert result.production_readiness["passed"] is True
    assert (result.artifact_dir / "production_readiness.json").exists()


def test_training_workflow_uses_held_out_benchmark_when_real_ood_is_missing(monkeypatch, tmp_path: Path):
    benchmark_calls = []
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a", "disease_b"]),
            "val": FakeLoader(["healthy", "disease_a", "disease_b"]),
            "test": FakeLoader(["healthy", "disease_a", "disease_b"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=ood_loader is not None),
    )
    monkeypatch.setattr(
        "src.workflows.training.run_leave_one_class_out_benchmark",
        lambda **kwargs: benchmark_calls.append(dict(kwargs)) or {
            "status": "completed",
            "passed": True,
            "metrics": {
                "ood_auroc": 0.96,
                "ood_false_positive_rate": 0.03,
                "sure_ds_f1": 0.94,
                "conformal_empirical_coverage": 0.97,
            },
            "paths": {
                "summary_json": str(
                    tmp_path / "outputs" / "training_metrics" / "ood_benchmark" / "summary.json"
                )
            },
        },
    )

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {
                        "require_ood_for_gate": True,
                        "ood_fallback_strategy": "held_out_benchmark",
                        "ood_benchmark_auto_run": True,
                        "ood_benchmark_min_classes": 3,
                    },
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert len(benchmark_calls) == 1
    assert result.ood_evidence_source == "held_out_benchmark"
    assert result.production_readiness["passed"] is True
    assert result.production_readiness["ood_evidence_source"] == "held_out_benchmark"


def test_training_workflow_allows_missing_ood_when_gate_is_optional(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy", "disease_a"]),
            "test": FakeLoader(["healthy", "disease_a"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=False),
    )
    monkeypatch.setattr(
        "src.workflows.training.run_leave_one_class_out_benchmark",
        lambda **kwargs: {},
    )

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {
                        "require_ood_for_gate": False,
                        "ood_fallback_strategy": "none",
                        "ood_benchmark_auto_run": False,
                    },
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert result.ood_evidence_source == "unavailable"
    assert result.production_readiness["passed"] is True
    assert result.production_readiness["ood_evidence"]["evaluation"]["require_ood"] is False


def test_training_workflow_can_skip_split_metric_gate_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.workflows.training.create_training_loaders",
        lambda **kwargs: {
            "train": FakeLoader(["healthy", "disease_a"]),
            "val": FakeLoader(["healthy", "disease_a"]),
            "test": FakeLoader(["healthy", "disease_a"]),
        },
    )
    monkeypatch.setattr("src.workflows.training.IndependentCropAdapter", FakeAdapter)
    monkeypatch.setattr(
        "src.workflows.training.evaluate_model_with_artifact_metrics",
        lambda trainer, loader, *, ood_loader=None: _fake_evaluation_result(include_ood=False),
    )
    monkeypatch.setattr(
        "src.workflows.training.run_leave_one_class_out_benchmark",
        lambda **kwargs: {},
    )

    workflow = TrainingWorkflow(
        config={
            "training": {
                "continual": {
                    "backbone": {"model_name": "fake"},
                    "batch_size": 2,
                    "seed": 7,
                    "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
                    "evaluation": {
                        "emit_ood_gate": False,
                        "require_ood_for_gate": False,
                        "ood_fallback_strategy": "none",
                        "ood_benchmark_auto_run": False,
                    },
                }
            },
            "colab": {"training": {"num_workers": 0, "pin_memory": False}},
        },
        device="cpu",
    )

    result = workflow.run(
        crop_name="tomato",
        data_dir=tmp_path / "runtime_data",
        output_dir=tmp_path / "outputs",
    )

    assert result.production_readiness["passed"] is True
    assert "metric_gate_json" not in result.artifacts["validation"]
    assert "metric_gate_json" not in result.artifacts["test"]
    assert not (result.artifact_dir / "validation" / "metric_gate.json").exists()
    assert not (result.artifact_dir / "test" / "metric_gate.json").exists()
