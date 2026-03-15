#!/usr/bin/env python3
"""Prune exact duplicates and low-risk same-class review clusters from a class-root dataset."""

from __future__ import annotations

import argparse
import csv
import random
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from scripts.colab_dataset_layout import IMAGE_EXTENSIONS
from scripts.prepare_grouped_runtime_dataset import (
    ImageRecord,
    ReviewPair,
    UnionFind,
    _classify_review_cluster,
)

DEFAULT_SEED = 42
VARIANT_SUFFIXES = ("mirror_vertical", "mirror", "lower", "height", "change")


@dataclass
class DuplicateCleanupAction:
    duplicate_group_index: int
    duplicate_count: int
    kept_relative_paths: str
    selected_relative_path: str
    deleted_relative_path: str
    delete_reason: str


def _load_rows_from_source(source_path: Path, *, suffix: str) -> List[Dict[str, str]]:
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(f"/{suffix}")]
            if not members:
                raise FileNotFoundError(f"No {suffix} found in zip: {source_path}")
            if len(members) > 1:
                raise RuntimeError(f"Zip contains multiple {suffix} files: {members}")
            payload = archive.read(members[0]).decode("utf-8")
        return list(csv.DictReader(payload.splitlines()))
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_exact_duplicate_rows(source_path: Path) -> List[Dict[str, str]]:
    return _load_rows_from_source(source_path, suffix="exact_duplicates.csv")


def _load_review_rows(source_path: Path) -> List[Dict[str, str]]:
    return _load_rows_from_source(source_path, suffix="same_class_review_candidates.csv")


def _load_manifest_rows(source_path: Path) -> List[Dict[str, str]]:
    return _load_rows_from_source(source_path, suffix="dataset_manifest.csv")


def _strip_known_variant_suffix(stem: str) -> str:
    lowered = str(stem)
    for suffix in sorted(VARIANT_SUFFIXES, key=len, reverse=True):
        for separator in ("_", "-", " "):
            token = f"{separator}{suffix}"
            if lowered.lower().endswith(token.lower()):
                return lowered[: -len(token)]
        if lowered.lower().endswith(suffix.lower()) and len(lowered) > len(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _variant_pattern(base_stem: str) -> re.Pattern[str]:
    suffix_group = "|".join(re.escape(item) for item in sorted(VARIANT_SUFFIXES, key=len, reverse=True))
    return re.compile(
        rf"^{re.escape(base_stem)}(?:[_\-\s]?(?:{suffix_group}))?$",
        re.IGNORECASE,
    )


def _find_variant_relpaths(dataset_root: Path, relative_path: str) -> List[str]:
    relative = Path(relative_path)
    parent_dir = (dataset_root / relative.parent)
    if not parent_dir.is_dir():
        return []
    base_stem = _strip_known_variant_suffix(relative.stem)
    pattern = _variant_pattern(base_stem)
    matches: List[str] = []
    for candidate in parent_dir.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if pattern.match(candidate.stem):
            matches.append(candidate.relative_to(dataset_root).as_posix())
    return sorted(set(matches))


def build_cleanup_plan(
    *,
    dataset_root: Path,
    exact_duplicates_source: Path,
    seed: int = DEFAULT_SEED,
) -> List[DuplicateCleanupAction]:
    rng = random.Random(int(seed))
    rows = _load_exact_duplicate_rows(Path(exact_duplicates_source))
    actions: List[DuplicateCleanupAction] = []

    for row_index, row in enumerate(rows, start=1):
        raw_paths = str(row.get("relative_paths", "")).strip()
        if not raw_paths:
            continue
        relative_paths = [item.strip() for item in raw_paths.split("|") if item.strip()]
        if len(relative_paths) < 2:
            continue
        delete_count = len(relative_paths) - 1
        selected = sorted(rng.sample(relative_paths, delete_count))
        kept = sorted(set(relative_paths) - set(selected))
        for selected_path in selected:
            matches = _find_variant_relpaths(Path(dataset_root), selected_path)
            if selected_path not in matches:
                matches = sorted(set(matches + [selected_path]))
            for matched in matches:
                reason = "selected_exact_duplicate" if matched == selected_path else "selected_variant_family"
                actions.append(
                    DuplicateCleanupAction(
                        duplicate_group_index=row_index,
                        duplicate_count=len(relative_paths),
                        kept_relative_paths="|".join(kept),
                        selected_relative_path=selected_path,
                        deleted_relative_path=matched,
                        delete_reason=reason,
                    )
                )
    deduped: Dict[str, DuplicateCleanupAction] = {}
    for action in actions:
        deduped.setdefault(action.deleted_relative_path, action)
    return sorted(deduped.values(), key=lambda item: (item.duplicate_group_index, item.deleted_relative_path))


def _record_from_manifest_row(row: Dict[str, str]) -> ImageRecord:
    synthetic_raw = str(row.get("synthetic_hint", "")).strip().lower()
    return ImageRecord(
        relative_path=str(row.get("relative_path", "")),
        absolute_path=str(row.get("absolute_path", "")),
        raw_class_name=str(row.get("raw_class_name", "")),
        normalized_class_name=str(row.get("normalized_class_name", "")),
        source_hint=str(row.get("source_hint", "")),
        synthetic_hint=synthetic_raw in {"1", "true", "yes"},
        readable=True,
        width=int(float(row.get("width", 0) or 0)),
        height=int(float(row.get("height", 0) or 0)),
        blur_score=float(row.get("blur_score", 0.0) or 0.0),
        brightness_mean=float(row.get("brightness_mean", 0.0) or 0.0),
        exact_hash=str(row.get("exact_hash", "")),
        phash_hex=str(row.get("phash_hex", "")),
        class_order_index=int(float(row.get("class_order_index", 0) or 0)),
        excluded_reason=str(row.get("excluded_reason", "")),
    )


def _review_pair_from_row(row: Dict[str, str]) -> ReviewPair:
    adjacency_raw = str(row.get("adjacency_distance", "")).strip()
    return ReviewPair(
        pair_type=str(row.get("pair_type", "same_class_review")),
        class_a=str(row.get("class_a", "")),
        class_b=str(row.get("class_b", "")),
        path_a=str(row.get("path_a", "")),
        path_b=str(row.get("path_b", "")),
        exact_match=str(row.get("exact_match", "")).strip().lower() in {"1", "true", "yes"},
        phash_distance=int(float(row.get("phash_distance", 0) or 0)),
        dino_cosine=float(row.get("dino_cosine", -1.0) or -1.0),
        bioclip_cosine=float(row.get("bioclip_cosine", -1.0) or -1.0),
        adjacency_distance=(None if adjacency_raw in {"", "None"} else int(float(adjacency_raw))),
        review_rank=int(float(row.get("review_rank", 0) or 0)),
        decision=str(row.get("decision", "review")),
        reason=str(row.get("reason", "")),
        cluster_id=str(row.get("cluster_id", "")),
        triage_resolution=str(row.get("triage_resolution", "")),
        triage_reason=str(row.get("triage_reason", "")),
    )


def build_review_cleanup_plan(
    *,
    dataset_root: Path,
    review_source: Path,
    dataset_manifest_source: Path,
    seed: int = DEFAULT_SEED,
    starting_group_index: int = 0,
) -> List[DuplicateCleanupAction]:
    rng = random.Random(int(seed))
    review_rows = _load_review_rows(Path(review_source))
    manifest_rows = _load_manifest_rows(Path(dataset_manifest_source))
    record_lookup = {
        row["relative_path"]: _record_from_manifest_row(row)
        for row in manifest_rows
        if str(row.get("relative_path", "")).strip()
    }
    review_pairs = [_review_pair_from_row(row) for row in review_rows]
    if not review_pairs:
        return []

    cluster_uf = UnionFind()
    for pair in review_pairs:
        cluster_uf.union(f"{pair.class_a}::{pair.path_a}", f"{pair.class_b}::{pair.path_b}")

    pairs_by_root: Dict[str, List[ReviewPair]] = defaultdict(list)
    paths_by_root: Dict[str, set[str]] = defaultdict(set)
    for pair in review_pairs:
        root = cluster_uf.find(f"{pair.class_a}::{pair.path_a}")
        pairs_by_root[root].append(pair)
        paths_by_root[root].add(pair.path_a)
        paths_by_root[root].add(pair.path_b)

    actions: List[DuplicateCleanupAction] = []
    group_index = int(starting_group_index)
    for root in sorted(pairs_by_root.keys()):
        pairs = pairs_by_root[root]
        relative_paths = sorted(paths_by_root[root])
        records = [record_lookup[path] for path in relative_paths if path in record_lookup]
        if len(records) < 2:
            continue
        resolution, reason = _classify_review_cluster(records=records, pairs=pairs)
        if resolution != "auto_resolve":
            continue
        group_index += 1
        delete_count = len(relative_paths) - 1
        selected = sorted(rng.sample(relative_paths, delete_count))
        kept = sorted(set(relative_paths) - set(selected))
        for selected_path in selected:
            matches = _find_variant_relpaths(Path(dataset_root), selected_path)
            if selected_path not in matches:
                matches = sorted(set(matches + [selected_path]))
            for matched in matches:
                delete_reason = "selected_review_cluster" if matched == selected_path else "selected_review_variant_family"
                actions.append(
                    DuplicateCleanupAction(
                        duplicate_group_index=group_index,
                        duplicate_count=len(relative_paths),
                        kept_relative_paths="|".join(kept),
                        selected_relative_path=selected_path,
                        deleted_relative_path=matched,
                        delete_reason=f"{delete_reason}:{reason}",
                    )
                )
    deduped: Dict[str, DuplicateCleanupAction] = {}
    for action in actions:
        deduped.setdefault(action.deleted_relative_path, action)
    return sorted(deduped.values(), key=lambda item: (item.duplicate_group_index, item.deleted_relative_path))


def build_combined_cleanup_plan(
    *,
    dataset_root: Path,
    exact_duplicates_source: Path,
    review_source: Path | None,
    dataset_manifest_source: Path | None,
    seed: int = DEFAULT_SEED,
) -> List[DuplicateCleanupAction]:
    exact_actions = build_cleanup_plan(
        dataset_root=dataset_root,
        exact_duplicates_source=exact_duplicates_source,
        seed=seed,
    )
    if review_source is None or dataset_manifest_source is None:
        return exact_actions
    review_actions = build_review_cleanup_plan(
        dataset_root=dataset_root,
        review_source=review_source,
        dataset_manifest_source=dataset_manifest_source,
        seed=seed,
        starting_group_index=max((action.duplicate_group_index for action in exact_actions), default=0),
    )
    combined: Dict[str, DuplicateCleanupAction] = {}
    for action in [*exact_actions, *review_actions]:
        combined.setdefault(action.deleted_relative_path, action)
    return sorted(combined.values(), key=lambda item: (item.duplicate_group_index, item.deleted_relative_path))


def write_cleanup_report(path: Path, actions: Sequence[DuplicateCleanupAction]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "duplicate_group_index",
        "duplicate_count",
        "kept_relative_paths",
        "selected_relative_path",
        "deleted_relative_path",
        "delete_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for action in actions:
            writer.writerow(
                {
                    "duplicate_group_index": action.duplicate_group_index,
                    "duplicate_count": action.duplicate_count,
                    "kept_relative_paths": action.kept_relative_paths,
                    "selected_relative_path": action.selected_relative_path,
                    "deleted_relative_path": action.deleted_relative_path,
                    "delete_reason": action.delete_reason,
                }
            )


def apply_cleanup_plan(*, dataset_root: Path, actions: Iterable[DuplicateCleanupAction]) -> int:
    deleted = 0
    for action in actions:
        target = Path(dataset_root) / action.deleted_relative_path
        if target.exists():
            target.unlink()
            deleted += 1
    return deleted


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Flat class-root dataset directory.")
    parser.add_argument(
        "--exact-duplicates-source",
        type=Path,
        required=True,
        help="Path to exact_duplicates.csv or a run zip that contains it.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic selection.")
    parser.add_argument(
        "--same-class-review-source",
        type=Path,
        default=None,
        help="Optional path to same_class_review_candidates.csv or a run zip that contains it.",
    )
    parser.add_argument(
        "--dataset-manifest-source",
        type=Path,
        default=None,
        help="Optional path to dataset_manifest.csv or a run zip that contains it. Required if --same-class-review-source is used.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("outputs") / "exact_duplicate_cleanup_plan.csv",
        help="Where to write the deletion plan CSV.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the planned files from --dataset-root. Without this flag, only the plan is written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.same_class_review_source and not args.dataset_manifest_source:
        parser.error("--dataset-manifest-source is required when --same-class-review-source is provided")

    actions = build_combined_cleanup_plan(
        dataset_root=args.dataset_root,
        exact_duplicates_source=args.exact_duplicates_source,
        review_source=args.same_class_review_source,
        dataset_manifest_source=args.dataset_manifest_source,
        seed=args.seed,
    )
    write_cleanup_report(args.report_csv, actions)
    print(f"[DEDUP] Planned deletions: {len(actions)}")
    print(f"[DEDUP] Report written to: {Path(args.report_csv).resolve()}")
    if args.apply:
        deleted = apply_cleanup_plan(dataset_root=args.dataset_root, actions=actions)
        print(f"[DEDUP] Deleted files: {deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
