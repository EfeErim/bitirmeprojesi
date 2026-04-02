"""Helpers for resolving runtime dataset roots for training workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.shared.json_utils import read_json


def normalize_runtime_crop_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    for token in (" ", "-", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _read_runtime_manifest(crop_root: Path) -> tuple[str, Dict[str, Any]]:
    for filename in ("split_manifest.json", "_split_metadata.json"):
        manifest_path = crop_root / filename
        if not manifest_path.exists():
            continue
        payload = read_json(manifest_path, default={}, expect_type=dict)
        if isinstance(payload, dict):
            return filename, dict(payload)
    return "", {}


@dataclass(frozen=True)
class ResolvedRuntimeDataset:
    dataset_key: str
    crop_root: Path
    resolution_source: str


def resolve_runtime_dataset(*, data_dir: str | Path, crop_name: str) -> ResolvedRuntimeDataset:
    resolved_data_dir = Path(data_dir)
    crop_key = normalize_runtime_crop_name(crop_name)
    direct_root = resolved_data_dir / crop_key
    if direct_root.is_dir():
        return ResolvedRuntimeDataset(
            dataset_key=crop_key,
            crop_root=direct_root,
            resolution_source="direct",
        )

    manifest_matches: list[tuple[Path, str]] = []
    prefix_matches: list[Path] = []
    if resolved_data_dir.is_dir():
        for candidate in sorted(resolved_data_dir.iterdir(), key=lambda item: item.name.lower()):
            if not candidate.is_dir():
                continue
            candidate_key = str(candidate.name).strip().lower()
            if candidate_key.startswith(f"{crop_key}__"):
                prefix_matches.append(candidate)
            manifest_name, manifest_payload = _read_runtime_manifest(candidate)
            manifest_crop_key = normalize_runtime_crop_name(manifest_payload.get("crop_name", ""))
            if manifest_name and manifest_crop_key == crop_key:
                manifest_matches.append((candidate, manifest_name))

    if len(manifest_matches) == 1:
        crop_root, manifest_name = manifest_matches[0]
        return ResolvedRuntimeDataset(
            dataset_key=str(crop_root.name),
            crop_root=crop_root,
            resolution_source=f"manifest:{manifest_name}",
        )
    if len(manifest_matches) > 1:
        matches = ", ".join(str(path.name) for path, _ in manifest_matches)
        raise ValueError(
            f"Multiple runtime datasets matched crop '{crop_key}' under {resolved_data_dir}: {matches}. "
            "Point data_dir at the specific runtime root or remove the ambiguity."
        )

    if len(prefix_matches) == 1:
        crop_root = prefix_matches[0]
        return ResolvedRuntimeDataset(
            dataset_key=str(crop_root.name),
            crop_root=crop_root,
            resolution_source="directory_prefix",
        )
    if len(prefix_matches) > 1:
        matches = ", ".join(str(path.name) for path in prefix_matches)
        raise ValueError(
            f"Multiple runtime dataset directories matched crop '{crop_key}' under {resolved_data_dir}: {matches}. "
            "Point data_dir at the specific runtime root or remove the ambiguity."
        )

    return ResolvedRuntimeDataset(
        dataset_key=crop_key,
        crop_root=direct_root,
        resolution_source="default",
    )
