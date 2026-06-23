#!/usr/bin/env python3
"""Calibrate prototype-router reconciliation thresholds on a held-out manifest."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ScoredRow:
    image_id: str
    expected_target: str
    expected_behavior: str
    predicted_target: str | None
    similarity: float
    margin: float
    resolved_image: str
    status: str
    prototype_class_label: str | None = None
    prototype_level: str = "target"


def _target_is_supported_positive(target: str) -> bool:
    return "__" in target and not target.startswith(("unknown_crop", "non_plant")) and not target.endswith(
        "__unknown_part"
    )


def _target_is_negative(target: str, expected_behavior: str = "") -> bool:
    normalized = str(target or "").strip().lower()
    behavior = str(expected_behavior or "").strip().lower()
    return (
        normalized in {"unknown_crop", "non_plant"}
        or normalized.endswith("__unknown_part")
        or "unsupported" in behavior
        or "abstain" in behavior
    )


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
    rows = parse_manifest_rows(manifest_path)
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
                    expected_behavior=row.expected_behavior,
                    predicted_target=None,
                    similarity=0.0,
                    margin=0.0,
                    resolved_image="" if image_path is None else str(image_path),
                    status=asset_status,
                    prototype_class_label=None,
                    prototype_level="unavailable",
                )
            )
            continue
        try:
            match = nearest_target(image_path, prototype_payload)
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    expected_behavior=row.expected_behavior,
                    predicted_target=match.target_id,
                    similarity=match.similarity,
                    margin=match.margin,
                    resolved_image=str(image_path),
                    status="ok",
                    prototype_class_label=match.class_label,
                    prototype_level=match.prototype_level,
                )
            )
        except Exception as exc:
            scored.append(
                ScoredRow(
                    image_id=row.image_id,
                    expected_target=row.expected_target,
                    expected_behavior=row.expected_behavior,
                    predicted_target=None,
                    similarity=0.0,
                    margin=0.0,
                    resolved_image=str(image_path),
                    status=f"error:{exc}",
                    prototype_class_label=None,
                    prototype_level="error",
                )
            )
    return scored


def evaluate_thresholds(
    rows: list[ScoredRow],
    *,
    min_similarity: float,
    min_margin: float,
    min_negative_gap: float = 0.0,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.status == "ok"]
    supported = [row for row in eligible if _target_is_supported_positive(row.expected_target)]
    negatives = [row for row in eligible if _target_is_negative(row.expected_target, row.expected_behavior)]
    accepted = [
        row
        for row in eligible
        if row.predicted_target
        and row.similarity >= min_similarity
        and row.margin >= min_margin
        and row.margin >= min_negative_gap
    ]
    supported_accepted = [row for row in accepted if _target_is_supported_positive(row.expected_target)]
    correct = [row for row in supported_accepted if row.predicted_target == row.expected_target]
    wrong = [row for row in supported_accepted if row.predicted_target != row.expected_target]
    negative_false_accepts = [row for row in accepted if _target_is_negative(row.expected_target, row.expected_behavior)]
    non_plant_false_accepts = [
        row for row in negative_false_accepts if str(row.expected_target or "").strip().lower() == "non_plant"
    ]
    total = len(eligible)
    coverage = len(accepted) / total if total else 0.0
    precision = len(correct) / len(accepted) if accepted else 0.0
    accuracy = len(correct) / total if total else 0.0
    supported_coverage = len(supported_accepted) / len(supported) if supported else 0.0
    supported_precision = len(correct) / len(supported_accepted) if supported_accepted else 0.0
    negative_false_accept_rate = len(negative_false_accepts) / len(negatives) if negatives else 0.0
    return {
        "min_similarity": min_similarity,
        "min_margin": min_margin,
        "min_negative_gap": min_negative_gap,
        "promotion_mode": "prototype_override",
        "eligible": total,
        "accepted": len(accepted),
        "correct": len(correct),
        "wrong": len(wrong),
        "coverage": round(coverage, 6),
        "precision": round(precision, 6),
        "accuracy": round(accuracy, 6),
        "supported_rows": len(supported),
        "supported_accepted": len(supported_accepted),
        "supported_correct": len(correct),
        "supported_wrong": len(wrong),
        "supported_wrong_image_ids": [row.image_id for row in wrong[:25]],
        "supported_wrong_rows": [
            {
                "image_id": row.image_id,
                "expected_target": row.expected_target,
                "predicted_target": row.predicted_target,
                "prototype_class_label": row.prototype_class_label,
                "prototype_level": row.prototype_level,
                "similarity": row.similarity,
                "margin": row.margin,
            }
            for row in wrong[:25]
        ],
        "supported_wrong_truncated": len(wrong) > 25,
        "supported_coverage": round(supported_coverage, 6),
        "supported_precision": round(supported_precision, 6),
        "negative_rows": len(negatives),
        "negative_false_accept_count": len(negative_false_accepts),
        "negative_false_accept_rate": round(negative_false_accept_rate, 6),
        "non_plant_false_accept_count": len(non_plant_false_accepts),
    }


def evaluate_class_thresholds(
    rows: list[ScoredRow],
    *,
    target_id: str,
    class_label: str,
    min_similarity: float,
    min_margin: float,
    min_negative_gap: float = 0.0,
    include_negative_rows: bool = True,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.status == "ok"
        and (include_negative_rows or not _target_is_negative(row.expected_target, row.expected_behavior))
    ]
    target_rows = [row for row in eligible if row.expected_target == target_id]
    accepted = [
        row
        for row in eligible
        if row.predicted_target == target_id
        and row.prototype_class_label == class_label
        and row.similarity >= min_similarity
        and row.margin >= min_margin
        and row.margin >= min_negative_gap
    ]
    supported_accepted = [row for row in accepted if _target_is_supported_positive(row.expected_target)]
    correct = [row for row in supported_accepted if row.expected_target == target_id]
    wrong = [row for row in supported_accepted if row.expected_target != target_id]
    negative_false_accepts = [row for row in accepted if _target_is_negative(row.expected_target, row.expected_behavior)]
    non_plant_false_accepts = [
        row for row in negative_false_accepts if str(row.expected_target or "").strip().lower() == "non_plant"
    ]
    supported_precision = len(correct) / len(supported_accepted) if supported_accepted else 0.0
    target_coverage = len(correct) / len(target_rows) if target_rows else 0.0
    return {
        "min_similarity": min_similarity,
        "min_margin": min_margin,
        "min_negative_gap": min_negative_gap,
        "promotion_mode": "prototype_override",
        "target_id": target_id,
        "class_label": class_label,
        "eligible": len(eligible),
        "target_rows": len(target_rows),
        "accepted": len(accepted),
        "supported_accepted": len(supported_accepted),
        "supported_correct": len(correct),
        "supported_wrong": len(wrong),
        "supported_precision": round(supported_precision, 6),
        "target_coverage": round(target_coverage, 6),
        "negative_false_accept_count": len(negative_false_accepts),
        "non_plant_false_accept_count": len(non_plant_false_accepts),
        "supported_wrong_image_ids": [row.image_id for row in wrong[:25]],
        "supported_wrong_rows": [
            {
                "image_id": row.image_id,
                "expected_target": row.expected_target,
                "predicted_target": row.predicted_target,
                "prototype_class_label": row.prototype_class_label,
                "prototype_level": row.prototype_level,
                "similarity": row.similarity,
                "margin": row.margin,
            }
            for row in wrong[:25]
        ],
        "supported_wrong_truncated": len(wrong) > 25,
    }


def calibrate_class_policies(
    rows: list[ScoredRow],
    *,
    target_id: str,
    similarity_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    negative_gap_grid: tuple[float, ...],
    min_precision: float,
    max_supported_wrong: int | None,
    min_accepted: int,
    require_zero_non_plant_false_accepts: bool,
    max_negative_false_accepts: int | None,
    include_negative_rows: bool = True,
) -> dict[str, Any]:
    class_labels = sorted(
        {
            str(row.prototype_class_label or "").strip()
            for row in rows
            if row.status == "ok" and row.predicted_target == target_id and str(row.prototype_class_label or "").strip()
        }
    )
    policies: dict[str, dict[str, Any]] = {}
    for class_label in class_labels:
        candidates: list[dict[str, Any]] = []
        for min_similarity in similarity_grid:
            for min_margin in margin_grid:
                for min_negative_gap in negative_gap_grid:
                    result = evaluate_class_thresholds(
                        rows,
                        target_id=target_id,
                        class_label=class_label,
                        min_similarity=min_similarity,
                        min_margin=min_margin,
                        min_negative_gap=min_negative_gap,
                        include_negative_rows=include_negative_rows,
                    )
                    result["eligible_for_promotion"] = (
                        result["supported_accepted"] >= min_accepted
                        and result["supported_precision"] >= min_precision
                        and (
                            max_supported_wrong is None
                            or result["supported_wrong"] <= max_supported_wrong
                        )
                        and (
                            not require_zero_non_plant_false_accepts
                            or result["non_plant_false_accept_count"] == 0
                        )
                        and (
                            max_negative_false_accepts is None
                            or result["negative_false_accept_count"] <= max_negative_false_accepts
                        )
                    )
                    candidates.append(result)
        candidates.sort(
            key=lambda item: (
                bool(item["eligible_for_promotion"]),
                float(item["supported_correct"]),
                float(item["target_coverage"]),
                -float(item["negative_false_accept_count"]),
                float(item["supported_precision"]),
                -float(item["min_similarity"]),
                -float(item["min_margin"]),
                -float(item["min_negative_gap"]),
            ),
            reverse=True,
        )
        selected = candidates[0] if candidates and candidates[0].get("eligible_for_promotion") else None
        best_candidate = candidates[0] if candidates else None
        failure_reasons: list[str] = []
        if not selected and best_candidate:
            if int(best_candidate.get("supported_accepted") or 0) < int(min_accepted):
                failure_reasons.append("supported_accepted_below_class_min")
            if float(best_candidate.get("supported_precision") or 0.0) < float(min_precision):
                failure_reasons.append("supported_precision_below_class_target")
            if (
                max_supported_wrong is not None
                and int(best_candidate.get("supported_wrong") or 0) > int(max_supported_wrong)
            ):
                failure_reasons.append("supported_wrong_above_class_target")
            if (
                max_negative_false_accepts is not None
                and int(best_candidate.get("negative_false_accept_count") or 0) > int(max_negative_false_accepts)
            ):
                failure_reasons.append("negative_false_accepts_above_class_target")
            if require_zero_non_plant_false_accepts and int(best_candidate.get("non_plant_false_accept_count") or 0):
                failure_reasons.append("non_plant_false_accepts_present")
        policies[class_label] = {
            "status": "class_specific" if selected else "no_eligible_policy",
            "selected_policy": selected,
            "best_candidate": best_candidate,
            "failure_reasons": failure_reasons,
        }
    return policies


def calibrate(
    rows: list[ScoredRow],
    *,
    similarity_grid: tuple[float, ...],
    margin_grid: tuple[float, ...],
    negative_gap_grid: tuple[float, ...] = (0.0,),
    min_precision: float,
    min_coverage: float,
    require_zero_non_plant_false_accepts: bool = True,
    max_negative_false_accepts: int | None = 0,
    max_negative_false_accept_rate: float | None = 0.05,
    max_supported_wrong: int | None = None,
    include_target_policies: bool = True,
    target_policy_negative_mode: str = "all",
    target_min_precision: float | None = None,
    target_max_supported_wrong: int | None = None,
    include_class_policies: bool = True,
    target_class_min_accepted: int = 5,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for min_similarity in similarity_grid:
        for min_margin in margin_grid:
            for min_negative_gap in negative_gap_grid:
                result = evaluate_thresholds(
                    rows,
                    min_similarity=min_similarity,
                    min_margin=min_margin,
                    min_negative_gap=min_negative_gap,
                )
                result["eligible_for_promotion"] = (
                    result["supported_precision"] >= min_precision
                    and result["supported_coverage"] >= min_coverage
                    and (
                        not require_zero_non_plant_false_accepts
                        or result["non_plant_false_accept_count"] == 0
                    )
                    and (
                        max_negative_false_accepts is None
                        or result["negative_false_accept_count"] <= max_negative_false_accepts
                    )
                    and (
                        max_negative_false_accept_rate is None
                        or result["negative_false_accept_rate"] <= max_negative_false_accept_rate
                    )
                    and (
                        max_supported_wrong is None
                        or result["supported_wrong"] <= max_supported_wrong
                    )
                )
                candidates.append(result)

    candidates.sort(
        key=lambda item: (
            bool(item["eligible_for_promotion"]),
            float(item["supported_coverage"]),
            -float(item["negative_false_accept_count"]),
            float(item["supported_precision"]),
            float(item["accuracy"]),
            -float(item["min_similarity"]),
            -float(item["min_margin"]),
            -float(item["min_negative_gap"]),
        ),
        reverse=True,
    )
    selected = candidates[0] if candidates else None
    target_policies: dict[str, dict[str, Any]] = {}
    if include_target_policies:
        negative_rows = [row for row in rows if _target_is_negative(row.expected_target, row.expected_behavior)]
        for target in sorted({row.expected_target for row in rows if _target_is_supported_positive(row.expected_target)}):
            target_rows = [row for row in rows if row.expected_target == target]
            target_calibration_rows = target_rows if target_policy_negative_mode == "none" else [*target_rows, *negative_rows]
            target_result = calibrate(
                target_calibration_rows,
                similarity_grid=similarity_grid,
                margin_grid=margin_grid,
                negative_gap_grid=negative_gap_grid,
                min_precision=min_precision if target_min_precision is None else target_min_precision,
                min_coverage=min_coverage,
                require_zero_non_plant_false_accepts=require_zero_non_plant_false_accepts,
                max_negative_false_accepts=max_negative_false_accepts,
                max_negative_false_accept_rate=max_negative_false_accept_rate,
                max_supported_wrong=target_max_supported_wrong,
                include_target_policies=False,
                target_policy_negative_mode=target_policy_negative_mode,
                include_class_policies=False,
                target_class_min_accepted=target_class_min_accepted,
            )
            target_constraints = {
                "min_precision": min_precision if target_min_precision is None else target_min_precision,
                "min_coverage": min_coverage,
                "max_supported_wrong": target_max_supported_wrong,
            }
            best_candidate = target_result["best_candidate"] or {}
            failure_reasons: list[str] = []
            if not target_result["selected_policy"]:
                if float(best_candidate.get("supported_precision") or 0.0) < float(target_constraints["min_precision"]):
                    failure_reasons.append("supported_precision_below_target")
                if float(best_candidate.get("supported_coverage") or 0.0) < float(target_constraints["min_coverage"]):
                    failure_reasons.append("supported_coverage_below_target")
                if (
                    target_constraints["max_supported_wrong"] is not None
                    and int(best_candidate.get("supported_wrong") or 0) > int(target_constraints["max_supported_wrong"])
                ):
                    failure_reasons.append("supported_wrong_above_target")
                if (
                    max_negative_false_accepts is not None
                    and int(best_candidate.get("negative_false_accept_count") or 0) > int(max_negative_false_accepts)
                ):
                    failure_reasons.append("negative_false_accepts_above_target")
                if require_zero_non_plant_false_accepts and int(best_candidate.get("non_plant_false_accept_count") or 0):
                    failure_reasons.append("non_plant_false_accepts_present")
            target_policies[target] = {
                "status": "target_specific" if target_result["selected_policy"] else "no_eligible_policy",
                "selected_policy": target_result["selected_policy"],
                "best_candidate": target_result["best_candidate"],
                "class_policies": (
                    calibrate_class_policies(
                        rows,
                        target_id=target,
                        similarity_grid=similarity_grid,
                        margin_grid=margin_grid,
                        negative_gap_grid=negative_gap_grid,
                        min_precision=min_precision if target_min_precision is None else target_min_precision,
                        max_supported_wrong=target_max_supported_wrong,
                        min_accepted=target_class_min_accepted,
                        require_zero_non_plant_false_accepts=require_zero_non_plant_false_accepts,
                        max_negative_false_accepts=max_negative_false_accepts,
                        include_negative_rows=True,
                    )
                    if include_class_policies and not target_result["selected_policy"]
                    else {}
                ),
                "negative_mode": target_policy_negative_mode,
                "constraints": target_constraints,
                "failure_reasons": failure_reasons,
            }
    return {
        "selected_policy": selected if selected and selected.get("eligible_for_promotion") else None,
        "best_candidate": selected,
        "target_policies": target_policies,
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
    parser.add_argument("--negative-gap-grid", default="0.00,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--min-precision", type=float, default=0.985)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--allow-non-plant-false-accepts", action="store_true")
    parser.add_argument("--max-negative-false-accepts", type=int, default=0)
    parser.add_argument("--max-negative-false-accept-rate", type=float, default=0.05)
    parser.add_argument(
        "--target-min-precision",
        type=float,
        default=0.98,
        help=(
            "Precision floor for target-specific policies. The global policy still uses --min-precision; "
            "the default recovers near-miss targets without promoting noisy targets."
        ),
    )
    parser.add_argument(
        "--target-max-supported-wrong",
        type=int,
        default=1,
        help="Maximum wrong supported rows allowed for a target-specific policy.",
    )
    parser.add_argument(
        "--target-policy-negative-mode",
        choices=("all", "none"),
        default="all",
        help=(
            "Use all negative rows when selecting each target policy, or select target policies from target rows only. "
            "The global selected policy always keeps the full negative guard."
        ),
    )
    parser.add_argument(
        "--target-class-min-accepted",
        type=int,
        default=5,
        help=(
            "Minimum accepted supported rows before a class-specific target policy can be used. "
            "Class policies are evaluated against the full calibration set to preserve false-accept guards."
        ),
    )
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
        negative_gap_grid=_parse_float_grid(args.negative_gap_grid, (0.00, 0.02, 0.04, 0.06, 0.08, 0.10)),
        min_precision=args.min_precision,
        min_coverage=args.min_coverage,
        require_zero_non_plant_false_accepts=not args.allow_non_plant_false_accepts,
        max_negative_false_accepts=args.max_negative_false_accepts,
        max_negative_false_accept_rate=args.max_negative_false_accept_rate,
        target_min_precision=args.target_min_precision,
        target_max_supported_wrong=args.target_max_supported_wrong,
        target_policy_negative_mode=args.target_policy_negative_mode,
        target_class_min_accepted=args.target_class_min_accepted,
    )
    payload = {
        "schema_version": "router_prototype_calibration.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest) if args.manifest.is_file() else "",
        "prototype_bank": str(args.prototype_bank),
        "prototype_bank_sha256": _sha256_file(args.prototype_bank) if args.prototype_bank.is_file() else "",
        "constraints": {
            "min_precision": args.min_precision,
            "min_coverage": args.min_coverage,
            "require_zero_non_plant_false_accepts": not args.allow_non_plant_false_accepts,
            "max_negative_false_accepts": args.max_negative_false_accepts,
            "max_negative_false_accept_rate": args.max_negative_false_accept_rate,
            "target_min_precision": args.target_min_precision,
            "target_max_supported_wrong": args.target_max_supported_wrong,
            "target_policy_negative_mode": args.target_policy_negative_mode,
            "target_class_min_accepted": args.target_class_min_accepted,
            "promotion_mode": "prototype_override",
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
