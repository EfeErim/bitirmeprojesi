"""Helpers to write and validate Colab training output manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


def write_output_manifest(
    output_dir: Path,
    phase: str,
    artifacts: Dict[str, Path],
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    normalized_artifacts = {}
    for key, path in artifacts.items():
        artifact_path = Path(path)
        normalized_artifacts[key] = {
            "path": str(artifact_path),
            "exists": artifact_path.exists(),
            "is_dir": artifact_path.is_dir(),
        }

    payload = {
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": normalized_artifacts,
        "metadata": metadata or {},
    }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def validate_manifest_artifacts(manifest_path: Path, required_keys: Iterable[str]) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {})
    missing_keys = [key for key in required_keys if key not in artifacts]
    if missing_keys:
        raise RuntimeError(f"OUTPUT_SUITE_COMPLETE failed: missing manifest keys: {missing_keys}")

    missing_files = [key for key in required_keys if not artifacts[key].get("exists")]
    if missing_files:
        raise RuntimeError(f"OUTPUT_SUITE_COMPLETE failed: missing artifact files: {missing_files}")
