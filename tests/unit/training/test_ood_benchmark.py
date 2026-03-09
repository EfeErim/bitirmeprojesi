from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.training.services.ood_benchmark import run_leave_one_class_out_benchmark
from src.training.types import EvaluationArtifactsPayload, ValidationReport


class ToyDataset(Dataset):
    def __init__(self, classes, items_per_class):
        self.classes = [str(name) for name in classes]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.samples = []
        self.labels = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            for item_idx in range(int(items_per_class)):
                self.samples.append(torch.tensor([float(class_idx), float(item_idx)], dtype=torch.float32))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


class FakeSession:
    def __init__(self, trainer):
        self.trainer = trainer

    def run(self):
        return None


class FakeAdapter:
    def __init__(self, crop_name, model_name="model", device="cpu"):
        self.crop_name = crop_name
        self.model_name = model_name
        self.device = device
        self._trainer = type("FakeTrainer", (), {})()

    def initialize_engine(self, *, class_names=None, config=None):
        del config
        self._trainer.class_names = list(class_names or [])
        return {"status": "initialized"}

    def build_training_session(self, train_loader, **kwargs):
        del train_loader, kwargs
        return FakeSession(self._trainer)

    def calibrate_ood(self, loader):
        del loader
        return {"status": "calibrated"}


def _fake_evaluate_model_with_artifact_metrics(_trainer, loader, *, ood_loader=None):
    y_true = [int(label) for label in getattr(loader.dataset, "labels", [])]
    y_pred = list(y_true)
    per_class_accuracy = {name: 1.0 for name in getattr(loader.dataset, "classes", [])}
    per_class_support = {name: 1 for name in getattr(loader.dataset, "classes", [])}
    report = ValidationReport(
        val_loss=0.05,
        val_accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        balanced_accuracy=1.0,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
        worst_classes=[],
    )
    ood_count = len(getattr(getattr(ood_loader, "dataset", None), "labels", [])) if ood_loader is not None else 0
    return EvaluationArtifactsPayload(
        report=report,
        y_true=y_true,
        y_pred=y_pred,
        ood_labels=([0] * len(y_true)) + ([1] * ood_count) if ood_loader is not None else None,
        ood_scores=([0.1] * len(y_true)) + ([0.9] * ood_count) if ood_loader is not None else None,
        sure_ds_f1=0.95,
        conformal_empirical_coverage=0.97,
        conformal_avg_set_size=1.0,
    )


def _build_loaders(classes, *, train_items=3, eval_items=2):
    return {
        "train": DataLoader(ToyDataset(classes, train_items), batch_size=2, shuffle=False),
        "val": DataLoader(ToyDataset(classes, eval_items), batch_size=2, shuffle=False),
        "test": DataLoader(ToyDataset(classes, eval_items), batch_size=2, shuffle=False),
    }


def test_run_leave_one_class_out_benchmark_writes_aggregate_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _fake_evaluate_model_with_artifact_metrics,
    )

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a", "disease_b"],
        loaders=_build_loaders(["healthy", "disease_a", "disease_b"]),
        config={"training": {"continual": {"backbone": {"model_name": "fake"}}}},
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_123",
        min_classes=3,
    )

    assert summary["status"] == "completed"
    assert summary["passed"] is True
    assert summary["successful_folds"] == 3
    assert summary["failed_folds"] == 0
    assert all(fold["status"] == "completed" for fold in summary["folds"])
    assert (tmp_path / "training_metrics" / "ood_benchmark" / "summary.json").exists()
    assert (tmp_path / "training_metrics" / "ood_benchmark" / "per_fold.csv").exists()


def test_run_leave_one_class_out_benchmark_fails_when_class_count_is_too_small(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.training.services.ood_benchmark.evaluate_model_with_artifact_metrics",
        _fake_evaluate_model_with_artifact_metrics,
    )

    summary = run_leave_one_class_out_benchmark(
        crop_name="tomato",
        class_names=["healthy", "disease_a"],
        loaders=_build_loaders(["healthy", "disease_a"]),
        config={"training": {"continual": {"backbone": {"model_name": "fake"}}}},
        device="cpu",
        artifact_root=tmp_path / "training_metrics",
        adapter_factory=FakeAdapter,
        run_id="run_456",
        min_classes=3,
    )

    assert summary["status"] == "failed"
    assert summary["passed"] is False
    assert summary["reason"] == "insufficient_classes_for_fallback"
    assert summary["successful_folds"] == 0
