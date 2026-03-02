"""Unit tests for v6 evaluation metric helpers."""

import pytest
import torch

from src.evaluation.metrics import compute_metrics, compute_protected_retention


def test_compute_metrics_returns_accuracy_and_counts():
    logits = torch.tensor(
        [
            [3.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.5, 0.2, 1.4],
            [0.2, 1.2, 0.4],
        ]
    )
    labels = torch.tensor([0, 1, 2, 1])

    metrics = compute_metrics(logits, labels)

    assert metrics["correct"] == 4.0
    assert metrics["total"] == 4.0
    assert metrics["accuracy"] == 1.0


def test_compute_metrics_accepts_non_1d_labels():
    logits = torch.tensor([[1.0, 0.0], [0.1, 0.9], [0.7, 0.2]])
    labels = torch.tensor([[0], [1], [0]])

    metrics = compute_metrics(logits, labels)

    assert metrics["correct"] == 3.0
    assert metrics["total"] == 3.0
    assert metrics["accuracy"] == 1.0


def test_compute_metrics_rejects_invalid_logit_shape():
    logits = torch.randn(2, 3, 4)
    labels = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="logits must have shape \\[N, C\\]"):
        compute_metrics(logits, labels)


def test_compute_protected_retention_matches_prediction_agreement():
    previous = torch.tensor(
        [
            [2.0, 0.1, 0.1],  # class 0
            [0.2, 1.5, 0.1],  # class 1
            [0.2, 0.3, 1.7],  # class 2
            [0.2, 0.9, 0.3],  # class 1
        ]
    )
    current = torch.tensor(
        [
            [1.8, 0.2, 0.1],  # class 0 (agree)
            [1.2, 0.4, 0.2],  # class 0 (disagree)
            [0.1, 0.3, 1.9],  # class 2 (agree)
            [0.1, 1.1, 0.4],  # class 1 (agree)
        ]
    )

    retention = compute_protected_retention(previous, current)

    assert retention == pytest.approx(0.75)


def test_compute_protected_retention_requires_same_shape():
    previous = torch.randn(4, 3)
    current = torch.randn(4, 2)

    with pytest.raises(ValueError, match="must share shape"):
        compute_protected_retention(previous, current)
