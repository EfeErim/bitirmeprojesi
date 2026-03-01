#!/usr/bin/env python3
"""Evaluation utilities for continual classifier training."""

from __future__ import annotations

from typing import Dict

import torch


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute loss-free classification metrics from logits and labels."""
    if logits.ndim != 2:
        raise ValueError('logits must have shape [N, C]')
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    pred = torch.argmax(logits, dim=1)
    correct = (pred == labels).sum().item()
    total = max(1, labels.numel())
    accuracy = float(correct / total)

    return {
        'accuracy': accuracy,
        'correct': float(correct),
        'total': float(total),
    }


def compute_protected_retention(previous_logits: torch.Tensor, current_logits: torch.Tensor) -> float:
    """Estimate retention based on agreement between prior/current predictions."""
    if previous_logits.shape != current_logits.shape:
        raise ValueError('previous_logits and current_logits must share shape')
    prev_pred = torch.argmax(previous_logits, dim=1)
    curr_pred = torch.argmax(current_logits, dim=1)
    retention = (prev_pred == curr_pred).float().mean().item()
    return float(retention)
