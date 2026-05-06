#!/usr/bin/env python3
"""Remove old temporary artifacts and run directories older than retention period."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cleanup_old_artifacts(
    root: Path, days_retention: int = 90, dry_run: bool = False
) -> dict[str, int]:
    """
    Remove artifacts older than retention period.

    Args:
        root: Repository root
        days_retention: Keep artifacts newer than this many days
        dry_run: If True, report what would be deleted without deleting

    Returns:
        Dict with counts of deleted directories and total freed space (bytes)
    """
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_retention)
    artifacts_root = root / "runs"
    runtime_tmp = root / ".runtime_tmp"

    deleted_count = 0
    freed_bytes = 0

    # Cleanup runs/<RUN_ID> directories
    if artifacts_root.exists():
        for run_dir in artifacts_root.iterdir():
            if run_dir.name in ("_index",):  # Skip index directory
                continue
            if not run_dir.is_dir():
                continue

            try:
                # Get modification time
                mtime = datetime.fromtimestamp(
                    run_dir.stat().st_mtime, tz=timezone.utc
                )
                if mtime < cutoff_time:
                    size = sum(
                        f.stat().st_size
                        for f in run_dir.rglob("*")
                        if f.is_file()
                    )
                    if dry_run:
                        print(f"[DRY RUN] Would delete: {run_dir} ({size} bytes, mtime={mtime})")
                    else:
                        shutil.rmtree(run_dir)
                        print(f"Deleted: {run_dir} ({size} bytes)")
                    deleted_count += 1
                    freed_bytes += size
            except Exception as e:
                print(f"Failed to process {run_dir}: {e}", file=sys.stderr)

    # Cleanup .runtime_tmp/* (but keep the folder itself)
    if runtime_tmp.exists():
        for item in runtime_tmp.iterdir():
            if item.name in ("__pycache__", ".pytest_cache"):
                continue
            try:
                mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff_time:
                    size = (
                        sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                        if item.is_dir()
                        else item.stat().st_size
                    )
                    if dry_run:
                        print(
                            f"[DRY RUN] Would delete: {item} ({size} bytes, mtime={mtime})"
                        )
                    else:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        print(f"Deleted: {item} ({size} bytes)")
                    deleted_count += 1
                    freed_bytes += size
            except Exception as e:
                print(f"Failed to process {item}: {e}", file=sys.stderr)

    return {"deleted_count": deleted_count, "freed_bytes": freed_bytes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove old temporary artifacts and run directories"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Retain artifacts newer than this many days (default: 90)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    result = cleanup_old_artifacts(ROOT, days_retention=args.days, dry_run=args.dry_run)
    freed_mb = result["freed_bytes"] / (1024 * 1024)
    print(
        f"\nCleanup complete: {result['deleted_count']} items deleted, "
        f"{freed_mb:.1f} MB freed"
    )
    sys.exit(0)
