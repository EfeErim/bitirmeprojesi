"""Backward-compatible loader exports.

Legacy notebook cells imported training loader helpers from ``src.utils.data_loader``.
The canonical implementation now lives in ``src.data.loaders``; this module keeps
that import path working without duplicating logic.
"""

from __future__ import annotations

from src.data.loaders import (
    VALID_SAMPLERS,
    build_weighted_sampler,
    create_training_loaders,
    dict_collate_fn,
    seed_worker_factory,
)

__all__ = [
    "VALID_SAMPLERS",
    "build_weighted_sampler",
    "create_training_loaders",
    "dict_collate_fn",
    "seed_worker_factory",
]
