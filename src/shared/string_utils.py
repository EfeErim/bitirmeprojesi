"""Shared string normalization utilities."""

from __future__ import annotations


def normalize_class_name(name: str) -> str:
    """Normalize a class name for dataset layout consistency.
    
    Converts to lowercase, replaces separators with underscores,
    collapses consecutive underscores, and strips leading/trailing underscores.
    
    Args:
        name: Raw class name (e.g., 'Grape Leaf', 'berry-spot', 'fruit/type').
    
    Returns:
        Normalized class name (e.g., 'grape_leaf', 'berry_spot', 'fruit_type').
    """
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")
