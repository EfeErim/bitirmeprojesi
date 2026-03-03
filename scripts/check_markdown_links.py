#!/usr/bin/env python3
"""Check local markdown links and fail on broken targets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
IGNORED_SCHEMES = ("http://", "https://", "mailto:", "tel:")
DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    ".runtime_tmp",
    ".tmp",
    ".kilocode",
    "venv",
    "archive",
    "node_modules",
    "site-packages",
    "__pycache__",
    ".pytest_cache",
}


def should_skip_target(target: str) -> bool:
    stripped = target.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    return stripped.startswith(IGNORED_SCHEMES)


def normalize_target(target: str) -> str:
    clean = target.strip()
    if "#" in clean:
        clean = clean.split("#", 1)[0]
    return clean.strip()


def iter_markdown_files(root: Path, excludes: Sequence[str]) -> Iterable[Path]:
    exclude_set = set(excludes)
    for path in root.rglob("*.md"):
        if any(part in exclude_set for part in path.parts):
            continue
        yield path


def find_broken_links(file_path: Path) -> List[Tuple[str, str]]:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    broken: List[Tuple[str, str]] = []

    for match in LINK_PATTERN.finditer(content):
        raw_target = match.group(1).strip()
        if should_skip_target(raw_target):
            continue

        target = normalize_target(raw_target)
        if not target:
            continue

        resolved = (file_path.parent / target).resolve()
        if not resolved.exists():
            broken.append((str(file_path), raw_target))

    return broken


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local markdown links")
    parser.add_argument("--root", default=".", help="Repository root to scan")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDES),
        help="Directory names to exclude from scanning",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    markdown_files = list(iter_markdown_files(root, args.exclude))

    all_broken: List[Tuple[str, str]] = []
    for file_path in markdown_files:
        all_broken.extend(find_broken_links(file_path))

    if all_broken:
        unique_broken = sorted(set(all_broken))
        print(f"BROKEN LINK COUNT: {len(unique_broken)}")
        for source, target in unique_broken:
            rel_source = Path(source).resolve().relative_to(root)
            print(f"{rel_source} -> {target}")
        return 1

    print(f"OK: no broken local markdown links found in {len(markdown_files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
