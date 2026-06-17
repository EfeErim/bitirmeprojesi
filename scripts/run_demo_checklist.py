#!/usr/bin/env python3
"""Run the M2 demo checklist through the Notebook 8 helper path."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab_auto_router_adapter_prediction import run_auto_router_adapter_prediction  # noqa: E402
from scripts.colab_router_adapter_inference import run_inference as run_router_inference  # noqa: E402
from src.workflows.inference import InferenceWorkflow  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ABSTAIN_STATUSES = {
    "unknown_crop",
    "router_uncertain",
    "adapter_unavailable",
    "non_plant_rejected",
    "router_unavailable",
}
DEPENDENCY_MARKERS = (
    "gated repo",
    "401 client error",
    "access to model",
    "not authenticated",
    "cuda",
    "no module named",
)


@dataclass(frozen=True)
class ChecklistRow:
    image_id: str
    source: str
    expected_target: str
    expected_behavior: str
    notes: str


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_checklist_rows(checklist_path: Path) -> list[ChecklistRow]:
    rows: list[ChecklistRow] = []
    found_candidate_section = False
    in_table = False
    for line in checklist_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## M1 Candidate Checklist"):
            found_candidate_section = True
            continue
        if not found_candidate_section:
            continue
        if line.startswith("| image_id | source | expected_target |"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("## "):
            break
        if not line.startswith("| demo_"):
            continue
        cells = _split_markdown_row(line)
        if len(cells) < 12:
            raise ValueError(f"Checklist row has {len(cells)} cells, expected 12: {line}")
        rows.append(
            ChecklistRow(
                image_id=cells[0],
                source=cells[1],
                expected_target=cells[2],
                expected_behavior=cells[3],
                notes=cells[11],
            )
        )
    return rows


def parse_manifest_rows(manifest_path: Path) -> list[ChecklistRow]:
    rows: list[ChecklistRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "source", "expected_target", "expected_behavior", "notes"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing}")
        for record in reader:
            image_id = str(record.get("image_id") or "").strip()
            if not image_id:
                continue
            rows.append(
                ChecklistRow(
                    image_id=image_id,
                    source=str(record.get("source") or "").strip(),
                    expected_target=str(record.get("expected_target") or "").strip(),
                    expected_behavior=str(record.get("expected_behavior") or "").strip(),
                    notes=str(record.get("notes") or "").strip(),
                )
            )
    return rows


def _first_image(root: Path) -> Path | None:
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return root
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    return None


def _path_match_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).lower().replace("ı", "i"))
    return "".join(ch for ch in normalized if ch.isalnum() and not unicodedata.combining(ch))


def _resolve_existing_path(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    if candidate.exists():
        return candidate
    current = root
    for part in Path(relative_path).parts:
        direct = current / part
        if direct.exists():
            current = direct
            continue
        if not current.is_dir():
            return candidate
        part_key = _path_match_key(part)
        matches = [child for child in current.iterdir() if _path_match_key(child.name) == part_key]
        if not matches:
            return candidate
        current = sorted(matches, key=lambda path: path.name.lower())[0]
    return current


def resolve_image_path(source: str, repo_root: Path) -> tuple[Path | None, str]:
    source_kind, _, source_value = source.partition(":")
    if source_kind == "local_test_pool":
        image_path = _first_image(_resolve_existing_path(repo_root, source_value))
        return image_path, "ok" if image_path is not None else "asset_missing"
    if source_kind in {"staged_phone", "staged_external", "fallback_capture"}:
        image_path = repo_root / source_value
        return (image_path, "ok") if image_path.exists() else (image_path, "asset_missing")
    return None, "unsupported_source"


def _target_parts(expected_target: str) -> tuple[str | None, str | None]:
    if "__" not in expected_target:
        return None, None
    crop, part = expected_target.split("__", 1)
    if crop in {"unknown", "unknown_crop", "non_plant"}:
        return None, None
    if part in {"unknown", "unknown_part"}:
        return crop, None
    return crop, part


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _expected_class_from_source(source: str) -> str:
    if not source.startswith("local_test_pool:"):
        return ""
    return Path(source.partition(":")[2]).name


def classify_failure(result: dict[str, Any], *, asset_status: str) -> str:
    if asset_status != "ok":
        return "asset_missing"
    status = str(result.get("status") or "").lower()
    message = str(result.get("message") or "").lower()
    if status == "router_unavailable":
        if any(marker in message for marker in DEPENDENCY_MARKERS):
            return "dependency_access"
        return "notebook_runtime"
    if status in {"unknown_crop", "router_uncertain"}:
        return "router"
    if status == "adapter_unavailable":
        return "adapter_loading"
    if status == "non_plant_rejected":
        return "input_guard"
    if status == "success":
        return ""
    return "notebook_runtime"


def evaluate_pass(row: ChecklistRow, result: dict[str, Any], *, asset_status: str) -> str:
    if asset_status != "ok":
        return "fail"
    status = str(result.get("status") or "").lower()
    expected_target = row.expected_target.lower()
    expected_behavior = row.expected_behavior.lower()
    diagnosis = str(result.get("diagnosis") or "")

    if status == "router_unavailable":
        return "fail"
    if expected_target in {"unknown_crop", "non_plant"} or "unsupported" in expected_behavior:
        return "pass" if status in ABSTAIN_STATUSES and not diagnosis else "fail"
    if "unknown or unsafe" in expected_behavior or "review or low confidence" in expected_behavior:
        return "pass" if status == "success" or status in ABSTAIN_STATUSES else "fail"
    if "answer or review, no crash" in expected_behavior:
        return "pass" if status == "success" or status in ABSTAIN_STATUSES else "fail"
    if status != "success":
        return "fail"

    expected_class = _expected_class_from_source(row.source)
    if not expected_class:
        return "pass"
    return "pass" if _norm(expected_class) in _norm(diagnosis) or _norm(diagnosis) in _norm(expected_class) else "fail"


def _confidence_or_ood(result: dict[str, Any]) -> str:
    confidence = result.get("confidence")
    ood = result.get("ood_analysis")
    parts: list[str] = []
    if confidence is not None:
        try:
            parts.append(f"confidence={float(confidence):.4f}")
        except (TypeError, ValueError):
            parts.append(f"confidence={confidence}")
    if isinstance(ood, dict):
        parts.append(f"is_ood={bool(ood.get('is_ood', False))}")
        if ood.get("score_method"):
            parts.append(f"method={ood.get('score_method')}")
    return "; ".join(parts)


def _run_row(
    row: ChecklistRow,
    *,
    repo_root: Path,
    config_env: str,
    device: str,
    adapter_root: Path,
    mode: str,
) -> dict[str, Any]:
    image_path, asset_status = resolve_image_path(row.source, repo_root)
    if asset_status != "ok" or image_path is None:
        result: dict[str, Any] = {
            "status": "asset_missing",
            "crop": None,
            "part": None,
            "diagnosis": None,
            "confidence": 0.0,
            "message": f"Checklist source could not be resolved: {row.source}",
        }
    else:
        crop_hint, part_hint = _target_parts(row.expected_target)
        try:
            if mode == "asset-audit":
                result = {
                    "status": "asset_ready",
                    "crop": crop_hint,
                    "part": part_hint,
                    "diagnosis": None,
                    "confidence": 0.0,
                    "message": "Asset resolved; inference not run in asset-audit mode.",
                }
            elif mode == "adapter-smoke":
                if crop_hint is None or part_hint is None:
                    result = {
                        "status": "router_skipped_target_not_adapter_eligible",
                        "crop": crop_hint,
                        "part": part_hint,
                        "diagnosis": None,
                        "confidence": 0.0,
                        "message": "Expected target is not adapter-eligible for trusted-hint smoke.",
                    }
                else:
                    workflow = InferenceWorkflow(
                        environment=config_env,
                        device=device,
                        adapter_root=adapter_root,
                    )
                    result = workflow.predict(
                        image_path,
                        crop_hint=crop_hint,
                        part_hint=part_hint,
                        return_ood=True,
                        trust_crop_hint=True,
                    )
            else:
                router_result = run_router_inference(
                    image_path,
                    config_env=config_env,
                    device=device,
                )
                result = run_auto_router_adapter_prediction(
                    image_path,
                    router_result=router_result,
                    config_env=config_env,
                    device=device,
                    adapter_root=adapter_root,
                    return_ood=True,
                )
        except Exception as exc:  # Notebook execution surfaces dependency failures as runtime exceptions.
            result = {
                "status": "router_unavailable",
                "crop": None,
                "part": None,
                "diagnosis": None,
                "confidence": 0.0,
                "message": str(exc),
            }

    pass_fail = evaluate_pass(row, result, asset_status=asset_status)
    if mode == "asset-audit" and asset_status == "ok":
        pass_fail = "pass"
    failure_bucket = "" if pass_fail == "pass" else classify_failure(result, asset_status=asset_status)
    return {
        "image_id": row.image_id,
        "source": row.source,
        "resolved_image": "" if image_path is None else str(image_path),
        "expected_target": row.expected_target,
        "expected_behavior": row.expected_behavior,
        "actual_status": result.get("status"),
        "predicted_crop": result.get("crop"),
        "predicted_part": result.get("part"),
        "predicted_disease": result.get("diagnosis"),
        "confidence_or_ood": _confidence_or_ood(result),
        "pass_fail": pass_fail,
        "failure_bucket": failure_bucket,
        "notes": row.notes,
        "message": result.get("message", ""),
    }


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    answered = sum(1 for row in items if row.get("actual_status") == "success")
    passed = sum(1 for row in items if row.get("pass_fail") == "pass")
    failed = sum(1 for row in items if row.get("pass_fail") == "fail")
    abstained = sum(1 for row in items if str(row.get("actual_status") or "") in ABSTAIN_STATUSES)
    asset_ready = sum(1 for row in items if row.get("actual_status") == "asset_ready")
    buckets: dict[str, int] = {}
    targets: dict[str, dict[str, int]] = {}
    for row in items:
        bucket = str(row.get("failure_bucket") or "")
        if bucket:
            buckets[bucket] = buckets.get(bucket, 0) + 1
        target = str(row.get("expected_target") or "")
        target_summary = targets.setdefault(target, {"total": 0, "pass": 0, "fail": 0})
        target_summary["total"] += 1
        target_summary[str(row.get("pass_fail") or "fail")] += 1
    return {
        "total": len(items),
        "passed": passed,
        "failed": failed,
        "answered": answered,
        "abstained_or_reviewed": abstained,
        "asset_ready": asset_ready,
        "failure_buckets": dict(sorted(buckets.items())),
        "per_target": dict(sorted(targets.items())),
    }


def write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# M2 Demo Checklist Run",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- checklist: `{report['checklist']}`",
        f"- device: `{report['device']}`",
        f"- adapter_root: `{report['adapter_root']}`",
        f"- mode: `{report['mode']}`",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        f"- answered: {summary['answered']}",
        f"- abstained_or_reviewed: {summary['abstained_or_reviewed']}",
        f"- asset_ready: {summary['asset_ready']}",
        f"- failure_buckets: `{json.dumps(summary['failure_buckets'], sort_keys=True)}`",
        "",
        "## Rows",
        "",
        "| image_id | status | pass_fail | failure_bucket | predicted | message |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        predicted = " / ".join(
            str(row.get(key) or "") for key in ("predicted_crop", "predicted_part", "predicted_disease")
        )
        message = str(row.get("message") or "").replace("\n", " ")[:220]
        lines.append(
            "| {image_id} | {status} | {pass_fail} | {bucket} | {predicted} | {message} |".format(
                image_id=row["image_id"],
                status=row.get("actual_status") or "",
                pass_fail=row.get("pass_fail") or "",
                bucket=row.get("failure_bucket") or "",
                predicted=predicted,
                message=message,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", type=Path, default=Path("docs/demo_checklist.md"))
    parser.add_argument("--no-checklist", action="store_true", help="Run only rows from --extra-manifest files.")
    parser.add_argument(
        "--extra-manifest",
        action="append",
        default=[],
        type=Path,
        help="CSV manifest with image_id, source, expected_target, expected_behavior, notes columns.",
    )
    parser.add_argument("--output", type=Path, default=Path(".runtime_tmp/m2_demo_checklist_run.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path(".runtime_tmp/m2_demo_checklist_run.md"))
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--config-env", default="colab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-local", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("official", "adapter-smoke", "asset-audit"),
        default="official",
        help=(
            "official runs the Notebook 8 helper path; adapter-smoke skips router with expected crop/part "
            "and is not an official M2 pass; asset-audit only checks files."
        ),
    )
    parser.add_argument(
        "--trust-expected-target",
        action="store_true",
        help="Deprecated alias for --mode adapter-smoke.",
    )
    parser.add_argument("--stop-on-dependency-blocker", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    rows = [] if args.no_checklist else parse_checklist_rows(args.checklist)
    for manifest_path in args.extra_manifest:
        rows.extend(parse_manifest_rows(manifest_path))
    if args.only_local:
        rows = [row for row in rows if row.source.startswith("local_test_pool:")]
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    mode = "adapter-smoke" if args.trust_expected_target else str(args.mode)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        result = _run_row(
            row,
            repo_root=repo_root,
            config_env=str(args.config_env),
            device=str(args.device),
            adapter_root=args.adapter_root,
            mode=mode,
        )
        output_rows.append(result)
        if args.stop_on_dependency_blocker and result.get("failure_bucket") == "dependency_access":
            break

    report = {
        "schema_version": "v1_m2_demo_checklist_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checklist": str(args.checklist),
        "device": str(args.device),
        "adapter_root": str(args.adapter_root),
        "mode": mode,
        "trust_expected_target": mode == "adapter-smoke",
        "summary": summarize_results(output_rows),
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(report, args.markdown_output)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
