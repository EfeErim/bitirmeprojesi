from types import SimpleNamespace

import torch

from src.training.services.runtime import configure_runtime_reproducibility


def test_configure_runtime_reproducibility_enables_fast_cudnn_when_not_deterministic(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: calls.append(bool(enabled)))

    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        configure_runtime_reproducibility(SimpleNamespace(seed=7, deterministic=False))

        assert calls == [False]
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark


def test_configure_runtime_reproducibility_enforces_deterministic_cudnn(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: calls.append(bool(enabled)))

    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        configure_runtime_reproducibility(SimpleNamespace(seed=7, deterministic=True))

        assert calls == [True]
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark
