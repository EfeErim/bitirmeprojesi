#!/usr/bin/env python3
"""Calibrate prototype-router reconciliation thresholds on a held-out manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_demo_checklist import parse_manifest_rows, resolve_image_path  # noqa: E402
from src.router.prototype_reconciler import nearest_target  # noqa: E402
from src.shared.json_utils import read_json, write_json  # noqa: E402


@dataclass(frozen=True)
class ScoredRow:
    image_id: str
    expected_target: str
    predicted_target: str | None
    similarity: float
    margin: float
    resolved_image: str
    status: str


def _target_is_adapter_eligible(target: str) -> bool:
    return "__" in target and not target.startswith(("unknown_crop", "non_plant"))


def _parse_float_grid(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def score_manifest(
    *,
    manifest_path: Path,
    prototype_bank_path: Path,
    repo_root: Path,
    limit: int | None = None,
) -> list[ScoredRow]:
    prototype_payload = read_json(prototype_bank_path, default={}, expect_type=dict)
    rows = [row for row in parse_manifest_rows(manifest_path) if _target_is_adapter_eligible(row.expected_target)]
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    scored: list[ScoredRow] = []
    for row in rows:
        image_path, asset_status = resolve_image_path(row.source, repo_root)
        if asset_status != "ok" or image_path is None:
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    predicted_target=None,
                    similarity=0.0,
                    margin=0.0,
                    resolved_image="" if image_path is None else str(image_path),
                    status=asset_status,
                )
            )
            continue
        try:
            match = nearest_target(image_path, prototype_payload)
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    predicted_target=match.target_id,
                    similarity=match.similarity,
                    margin=match.margin,
                    resolved_image=str(image_path),
                    status="ok",
                )
            )
        except Exception as exc:
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    predicted_target=None,
                    similarity=0.0,
                    margin=0.0,
                    resolved_image=str(image_path),
                    status=f"error:{exc}",
                )
            )
    return scored


def evaluate_thresholds(
    rows: list[ScoredRow],
    *,
    min_similarity: float,
    min_margin: float,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.status == "ok"]
    accepted = [
        row
        for row in eligible
        if row.predicted_target and row.similarity >= min_similarity and row.margin >= min_margin
    ]
    correct = [row for row in accepted if row.predicted_target == row.expected_target]
    wrong = [row for row in accepted if row.predicted_target != row.expected_target]
    total = len(eligible)
    coverage = len(accepted) / total if total else 0.0
    precision = len(correct) / len(accepted) if accepted else 0.0
    accuracy = len(correct) / total if total else 0.0
    return {
        "min_similarity": min_similarity,
        "min_margin": min_margin,
        "eligible": total,
        "accepted": len(accepted),
        "correct": len(correct),
        "wrong": len(wrong),
        "coverage": round(coverage, 6),
        "precision": round(precision, 6),
        "accuracy": round(accuracy, 6),
    }


def calibrate(
    rows: list[ScoredRow],
    *,
    similarity_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    min_precision: float,
    min_coverage: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for min_similarity in similarity_grid:
        for min_margin in margin_grid:
            result = evaluate_thresholds(rows, min_similarity=min_similarity, min_margin=min_margin)
            result["eligible_for_promotion"] = (
                result["precision"] >= min_precision and result["coverage"] >= min_coverage
            )
            candidates.append(result)

    candidates.sort(
        key=lambda item: (
            bool(item["eligible_for_promotion"]),
            float(item["precision"]),
            float(item["coverage"]),
            float(item["accuracy"]),
            -float(item["min_similarity"]),
            -float(item["min_margin"]),
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates else None
    return {
        "selected_policy": selected if selected and selected.get("eligible_for_promotion") else None,
        "best_candidate": selected,
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prototype-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/router_prototype_calibration.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int)
    parser.add_argument("--similarity-grid", default="0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--margin-grid", default="0.00,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--fail-on-no-policy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = score_manifest(
        manifest_path=args.manifest,
        prototype_bank_path=args.prototype_bank,
        repo_root=args.repo_root,
        limit=args.limit,
    )
    calibration = calibrate(
        rows,
        similarity_grid=_parse_float_grid(args.similarity_grid, (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)),
        margin_grid=_parse_float_grid(args.margin_grid, (0.00, 0.02, 0.04, 0.06, 0.08, 0.10)),
        min_precision=args.min_precision,
        min_coverage=args.min_coverage,
    )
    payload = {
        "schema_version": "router_prototype_calibration.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "prototype_bank": str(args.prototype_bank),
        "constraints": {
            "min_precision": args.min_precision,
            "min_coverage": args.min_coverage,
        },
        "summary": {
            "rows": len(rows),
            "ok_rows": sum(1 for row in rows if row.status == "ok"),
            "selected_for_runtime": calibration["selected_policy"] is not None,
        },
        **calibration,
        "rows": [row.__dict__ for row in rows],
    }
    output = write_json(args.output, payload, ensure_ascii=False, sort_keys=False)
    print(json.dumps({"output": str(output), **payload["summary"]}, indent=2, ensure_ascii=False))
    if args.fail_on_no_policy and not payload["summary"]["selected_for_runtime"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
