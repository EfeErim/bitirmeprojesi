from datetime import datetime
from pathlib import Path

import pytest

from src.workflows.training_support import prepare_training_run


class FakeDataset:
    def __init__(self, classes, labels):
        self.classes = list(classes)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)


class FakeLoader:
    def __init__(self, classes, labels, batch_count):
        self.dataset = FakeDataset(classes, labels)
        self._batch_count = int(batch_count)

    def __len__(self):
        return self._batch_count


class FakeAdapter:
    def initialize_engine(self, *, class_names=None, config=None):
        self.initialized = {"class_names": list(class_names or []), "config": dict(config or {})}
        return self.initialized


def _base_config():
    return {
        "training": {
            "continual": {
                "backbone": {"model_name": "fake"},
                "batch_size": 2,
                "seed": 7,
                "data": {"target_size": 224, "cache_size": 10, "loader_error_policy": "tolerant"},
            }
        },
        "colab": {"training": {"num_workers": 0, "pin_memory": False}},
    }


def test_prepare_training_run_rejects_zero_train_batches():
    loaders = {
        "train": FakeLoader(["healthy"], [0], 0),
        "val": FakeLoader(["healthy"], [0], 1),
        "test": FakeLoader(["healthy"], [0], 1),
    }

    with pytest.raises(ValueError, match="zero batches"):
        prepare_training_run(
            config=_base_config(),
            device="cpu",
            crop_name="tomato",
            data_dir=Path("runtime"),
            class_names=None,
            num_workers=0,
            pin_memory=False,
            use_cache=False,
            sampler=None,
            error_policy=None,
            run_id="run_1",
            loader_factory=lambda **kwargs: loaders,
            adapter_factory=lambda **kwargs: FakeAdapter(),
        )


def test_prepare_training_run_rejects_eval_only_classes():
    loaders = {
        "train": FakeLoader(["healthy", "disease_a"], [0], 1),
        "val": FakeLoader(["healthy", "disease_a"], [1], 1),
        "test": FakeLoader(["healthy", "disease_a"], [0], 1),
    }

    with pytest.raises(ValueError, match="Validation/test splits contain classes"):
        prepare_training_run(
            config=_base_config(),
            device="cpu",
            crop_name="tomato",
            data_dir=Path("runtime"),
            class_names=None,
            num_workers=0,
            pin_memory=False,
            use_cache=False,
            sampler=None,
            error_policy=None,
            run_id="run_1",
            loader_factory=lambda **kwargs: loaders,
            adapter_factory=lambda **kwargs: FakeAdapter(),
        )


def test_prepare_training_run_generates_microsecond_resolution_run_ids(monkeypatch):
    loaders = {
        "train": FakeLoader(["healthy"], [0], 1),
        "val": FakeLoader(["healthy"], [0], 1),
        "test": FakeLoader(["healthy"], [0], 1),
    }

    class _FakeDateTime:
        values = [
            datetime(2026, 3, 11, 12, 0, 0, 1),
            datetime(2026, 3, 11, 12, 0, 0, 2),
        ]

        @classmethod
        def utcnow(cls):
            return cls.values.pop(0)

    monkeypatch.setattr("src.workflows.training_support.datetime", _FakeDateTime)

    first = prepare_training_run(
        config=_base_config(),
        device="cpu",
        crop_name="tomato",
        data_dir=Path("runtime"),
        class_names=None,
        num_workers=0,
        pin_memory=False,
        use_cache=False,
        sampler=None,
        error_policy=None,
        run_id="",
        loader_factory=lambda **kwargs: loaders,
        adapter_factory=lambda **kwargs: FakeAdapter(),
    )
    second = prepare_training_run(
        config=_base_config(),
        device="cpu",
        crop_name="tomato",
        data_dir=Path("runtime"),
        class_names=None,
        num_workers=0,
        pin_memory=False,
        use_cache=False,
        sampler=None,
        error_policy=None,
        run_id="",
        loader_factory=lambda **kwargs: loaders,
        adapter_factory=lambda **kwargs: FakeAdapter(),
    )

    assert first.run_id == "tomato_20260311_120000_000001"
    assert second.run_id == "tomato_20260311_120000_000002"
