"""Shared adapter recommendations for Notebook 2 and Notebook 6."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ADAPTER_RECS: dict[str, dict[str, Any]] = {
    "grape__fruit": {
        "crop": "grape", "part": "fruit",
        "ood": "data/prepared_runtime_datasets/grape__fruit/ood",
        "oe": "data/prepared_runtime_datasets/grape__fruit/oe",
        "oe_enabled": True, "oe_w": 0.24, "allow_under_min": False,
        "defaults": {"EPOCHS": 34, "BATCH_SIZE": 48, "LEARNING_RATE": 1e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.18, "OOD_FACTOR": 4.0, "LABEL_SMOOTHING": 0.08},
    },
    "grape__leaf": {
        "crop": "grape", "part": "leaf",
        "ood": "data/prepared_runtime_datasets/grape__leaf/ood",
        "oe": "data/prepared_runtime_datasets/grape__leaf/oe",
        "oe_enabled": True, "oe_w": 0.24, "allow_under_min": False,
        "defaults": {"EPOCHS": 30, "BATCH_SIZE": 56, "LEARNING_RATE": 1e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.16, "OOD_FACTOR": 4.0, "LABEL_SMOOTHING": 0.08},
    },
    "strawberry__fruit": {
        "crop": "strawberry", "part": "fruit",
        "ood": "data/prepared_runtime_datasets/strawberry__fruit/ood",
        "oe": "data/prepared_runtime_datasets/strawberry__fruit/oe",
        "oe_enabled": True, "oe_w": 0.18, "allow_under_min": True,
        "defaults": {"EPOCHS": 34, "BATCH_SIZE": 48, "LEARNING_RATE": 6e-5, "LORA_R": 20, "LORA_ALPHA": 20, "LORA_DROPOUT": 0.22, "OOD_FACTOR": 3.2, "LABEL_SMOOTHING": 0.08},
    },
    "strawberry__leaf": {
        "crop": "strawberry", "part": "leaf",
        "ood": "data/prepared_runtime_datasets/strawberry__leaf/ood",
        "oe": "data/prepared_runtime_datasets/strawberry__leaf/oe",
        "oe_enabled": True, "oe_w": 0.30, "allow_under_min": False,
        "defaults": {"EPOCHS": 20, "BATCH_SIZE": 88, "LEARNING_RATE": 1.2e-4, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.20, "OOD_FACTOR": 4.5, "LABEL_SMOOTHING": 0.10, "EARLY_STOPPING_PATIENCE": 5},
    },
    "tomato__fruit": {
        "crop": "tomato", "part": "fruit",
        "ood": "data/prepared_runtime_datasets/tomato__fruit/ood",
        "oe": "data/prepared_runtime_datasets/tomato__fruit/oe",
        "oe_enabled": True, "oe_w": 0.18, "allow_under_min": False,
        "defaults": {"EPOCHS": 30, "BATCH_SIZE": 64, "LEARNING_RATE": 8e-5, "LORA_R": 24, "LORA_ALPHA": 24, "LORA_DROPOUT": 0.16, "OOD_FACTOR": 3.8, "LABEL_SMOOTHING": 0.08},
    },
    "tomato__leaf": {
        "crop": "tomato", "part": "leaf",
        "ood": "data/prepared_runtime_datasets/tomato__leaf/ood",
        "oe": "data/prepared_runtime_datasets/tomato__leaf/oe",
        "oe_enabled": True, "oe_w": 0.20, "allow_under_min": False,
        "defaults": {"EPOCHS": 24, "BATCH_SIZE": 88, "LEARNING_RATE": 1.1e-4, "LORA_R": 32, "LORA_ALPHA": 32, "LORA_DROPOUT": 0.14, "OOD_FACTOR": 4.0, "LABEL_SMOOTHING": 0.08},
    },
    "apricot__fruit": {
        "crop": "apricot", "part": "fruit",
        "ood": "data/ood_dataset/final/apricot__fruit_ood_final",
        "oe": "", "oe_enabled": False, "oe_w": 0.10, "allow_under_min": False,
        "defaults": {"EPOCHS": 36, "BATCH_SIZE": 40, "LEARNING_RATE": 1e-4, "LORA_R": 20, "LORA_ALPHA": 20, "LORA_DROPOUT": 0.25, "OOD_FACTOR": 5.0, "LABEL_SMOOTHING": 0.15},
    },
    "apricot__leaf": {
        "crop": "apricot", "part": "leaf",
        "ood": "data/ood_dataset/final/apricot__leaf_ood_final",
        "oe": "data/oe_dataset/apricot_leaf_oe_unsupported_leaf_candidates",
        "oe_enabled": True, "oe_w": 0.35, "allow_under_min": False,
        "defaults": {"EPOCHS": 38, "BATCH_SIZE": 52, "LEARNING_RATE": 1.2e-4, "LORA_R": 26, "LORA_ALPHA": 26, "LORA_DROPOUT": 0.24, "OOD_FACTOR": 6.0, "LABEL_SMOOTHING": 0.15},
    },
}


def get_adapter_recs() -> dict[str, dict[str, Any]]:
    """Return a copy so notebook cells can mutate defaults safely."""
    return deepcopy(ADAPTER_RECS)
