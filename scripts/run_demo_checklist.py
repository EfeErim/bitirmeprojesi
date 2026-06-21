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
    expected_crop: str = ""
    expected_part: str = ""
    expected_class: str = ""


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
        expected_crop, expected_part = _target_parts(cells[2])
        rows.append(
            ChecklistRow(
                image_id=cells[0],
                source=cells[1],
                expected_target=cells[2],
                expected_behavior=cells[3],
                notes=cells[11],
                expected_crop=expected_crop or "",
                expected_part=expected_part or "",
                expected_class=_expected_class_from_source(cells[1]),
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
            expected_target = str(record.get("expected_target") or "").strip()
            inferred_crop, inferred_part = _target_parts(expected_target)
            source = str(record.get("source") or "").strip()
            original_source = str(record.get("original_source") or "").strip()
            expected_class = (
                str(record.get("expected_class") or "").strip()
                or str(record.get("disease_class") or "").strip()
                or _expected_class_from_source(source)
                or _expected_class_from_reference(original_source)
            )
            rows.append(
                ChecklistRow(
                    image_id=image_id,
                    source=source,
                    expected_target=expected_target,
                    expected_behavior=str(record.get("expected_behavior") or "").strip(),
                    notes=str(record.get("notes") or "").strip(),
                    expected_crop=str(record.get("expected_crop") or inferred_crop or "").strip(),
                    expected_part=str(record.get("expected_part") or inferred_part or "").strip(),
                    expected_class=expected_class,
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
    source_path = Path(source.partition(":")[2])
    if source_path.suffix.lower() in IMAGE_SUFFIXES:
        return source_path.parent.name
    return source_path.name


def _expected_class_from_reference(source: str) -> str:
    if not source:
        return ""
    if source.startswith("local_test_pool:"):
        return _expected_class_from_source(source)
    try:
        path = Path(source)
    except TypeError:
        return ""
    return path.parent.name if path.parent != path else ""


def _class_matches(expected_class: str, diagnosis: Any) -> bool:
    expected_key = _norm(expected_class)
    diagnosis_key = _norm(diagnosis)
    if not expected_key or not diagnosis_key:
        return False
    return expected_key in diagnosis_key or diagnosis_key in expected_key


def _opposite_part_label(expected_part: str, diagnosis: Any) -> bool:
    expected = str(expected_part or "").strip().lower()
    diagnosis_key = _norm(diagnosis)
    if expected == "fruit":
        return "leaf" in diagnosis_key or "yaprak" in diagnosis_key
    if expected == "leaf":
        return "fruit" in diagnosis_key or "meyve" in diagnosis_key
    return False


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

    expected_class = row.expected_class or _expected_class_from_source(row.source)
    if not expected_class:
        return "pass"
    return "pass" if _class_matches(expected_class, diagnosis) else "fail"


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


def resolve_prototype_thresholds_from_calibration(
    calibration_report_path: Path | None,
    *,
    min_similarity: float | None,
    min_margin: float | None,
) -> tuple[float | None, float | None, float | None, dict[str, Any], dict[str, Any]]:
    if calibration_report_path is None:
        return min_similarity, min_margin, None, {"enabled": False}, {}
    payload = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_policy") if isinstance(payload, dict) else None
    target_policies = payload.get("target_policies") if isinstance(payload, dict) else {}
    if not isinstance(target_policies, dict):
        target_policies = {}
    min_negative_gap = None
    report: dict[str, Any] = {
        "enabled": True,
        "path": str(calibration_report_path),
        "policy_selected": isinstance(selected, dict),
        "target_policy_count": len(target_policies),
    }
    if isinstance(selected, dict):
        if min_similarity is None:
            min_similarity = float(selected.get("min_similarity"))
        if min_margin is None:
            min_margin = float(selected.get("min_margin"))
        if selected.get("min_negative_gap") is not None:
            min_negative_gap = float(selected.get("min_negative_gap"))
        report["selected_policy"] = {
            "min_similarity": selected.get("min_similarity"),
            "min_margin": selected.get("min_margin"),
            "min_negative_gap": selected.get("min_negative_gap"),
            "precision": selected.get("precision"),
            "coverage": selected.get("coverage"),
            "supported_precision": selected.get("supported_precision"),
            "supported_coverage": selected.get("supported_coverage"),
            "negative_false_accept_count": selected.get("negative_false_accept_count"),
            "non_plant_false_accept_count": selected.get("non_plant_false_accept_count"),
        }
    return min_similarity, min_margin, min_negative_gap, report, target_policies


def _run_row(
    row: ChecklistRow,
    *,
    repo_root: Path,
    config_env: str,
    device: str,
    adapter_root: Path,
    mode: str,
    enable_prototype_reconciler: bool = False,
    prototype_bank_path: Path | None = None,
    taxonomy_registry_path: Path | None = None,
    prototype_min_similarity: float | None = None,
    prototype_min_margin: float | None = None,
    prototype_min_negative_gap: float | None = None,
    prototype_target_policies: dict[str, Any] | None = None,
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
                    enable_prototype_reconciler=enable_prototype_reconciler,
                    prototype_bank_path=prototype_bank_path,
                    taxonomy_registry_path=taxonomy_registry_path,
                    prototype_min_similarity=prototype_min_similarity,
                    prototype_min_margin=prototype_min_margin,
                    prototype_min_negative_gap=prototype_min_negative_gap,
                    prototype_target_policies=prototype_target_policies,
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
    router_handoff = result.get("router_handoff") if isinstance(result.get("router_handoff"), dict) else {}
    reconciliation = (
        router_handoff.get("prototype_reconciliation")
        if isinstance(router_handoff.get("prototype_reconciliation"), dict)
        else {}
    )
    return {
        "image_id": row.image_id,
        "source": row.source,
        "resolved_image": "" if image_path is None else str(image_path),
        "expected_target": row.expected_target,
        "expected_crop": row.expected_crop,
        "expected_part": row.expected_part,
        "expected_class": row.expected_class,
        "expected_behavior": row.expected_behavior,
        "actual_status": result.get("status"),
        "predicted_crop": result.get("crop"),
        "predicted_part": result.get("part"),
        "vlm_crop": reconciliation.get("vlm_crop"),
        "vlm_part": reconciliation.get("vlm_part"),
        "prototype_crop": reconciliation.get("prototype_crop"),
        "prototype_part": reconciliation.get("prototype_part"),
        "prototype_target": reconciliation.get("prototype_target"),
        "reconciled_crop": reconciliation.get("reconciled_crop"),
        "reconciled_part": reconciliation.get("reconciled_part"),
        "taxonomy_relation": reconciliation.get("taxonomy_relation"),
        "prototype_similarity": reconciliation.get("prototype_similarity"),
        "prototype_margin": reconciliation.get("prototype_margin"),
        "prototype_min_similarity": reconciliation.get("min_similarity"),
        "prototype_min_margin": reconciliation.get("min_margin"),
        "prototype_min_negative_gap": reconciliation.get("prototype_min_negative_gap"),
        "reconcile_decision": reconciliation.get("reconcile_decision"),
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


def build_analysis_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    crop_counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    part_counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    class_counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    adapter_unavailable = {"wrong_router": 0, "missing_adapter": 0, "unknown": 0}
    reconciliation_counts: dict[str, int] = {}
    per_target: dict[str, dict[str, int]] = {}
    opposite_part_rows: list[str] = []
    answered_wrong_by_target: dict[str, int] = {}
    answered_wrong_by_expected_class: dict[str, int] = {}
    prototype_correct_but_abstained: list[str] = []
    negative_false_accepts: list[str] = []
    policy_thresholds_by_target: dict[str, dict[str, Any]] = {}

    for row in items:
        target = str(row.get("expected_target") or "")
        target_summary = per_target.setdefault(
            target,
            {
                "total": 0,
                "answered": 0,
                "abstained_or_reviewed": 0,
                "pass": 0,
                "fail": 0,
                "exact_class_correct": 0,
                "opposite_part_disease_labels": 0,
            },
        )
        target_summary["total"] += 1
        pass_fail = str(row.get("pass_fail") or "fail")
        if pass_fail in {"pass", "fail"}:
            target_summary[pass_fail] += 1
        status = str(row.get("actual_status") or "")
        reconcile_decision = str(row.get("reconcile_decision") or "")
        if reconcile_decision:
            reconciliation_counts[reconcile_decision] = reconciliation_counts.get(reconcile_decision, 0) + 1
        if status == "success":
            target_summary["answered"] += 1
        if status in ABSTAIN_STATUSES:
            target_summary["abstained_or_reviewed"] += 1

        expected_crop = str(row.get("expected_crop") or "").strip().lower()
        expected_part = str(row.get("expected_part") or "").strip().lower()
        predicted_crop = str(row.get("predicted_crop") or "").strip().lower()
        predicted_part = str(row.get("predicted_part") or "").strip().lower()
        expected_class = str(row.get("expected_class") or "").strip()
        predicted_disease = row.get("predicted_disease")

        if expected_crop:
            crop_counts["correct" if predicted_crop == expected_crop else "incorrect"] += 1
        else:
            crop_counts["not_applicable"] += 1
        if expected_part:
            part_counts["correct" if predicted_part == expected_part else "incorrect"] += 1
        else:
            part_counts["not_applicable"] += 1
        if expected_class and status == "success":
            class_correct = _class_matches(expected_class, predicted_disease)
            class_counts["correct" if class_correct else "incorrect"] += 1
            if class_correct:
                target_summary["exact_class_correct"] += 1
            else:
                answered_wrong_by_target[target] = answered_wrong_by_target.get(target, 0) + 1
                answered_wrong_by_expected_class[expected_class] = answered_wrong_by_expected_class.get(expected_class, 0) + 1
        else:
            class_counts["not_applicable"] += 1

        if _opposite_part_label(expected_part, predicted_disease):
            opposite_part_rows.append(str(row.get("image_id") or ""))
            target_summary["opposite_part_disease_labels"] += 1

        if status == "adapter_unavailable":
            if expected_crop and expected_part:
                if predicted_crop != expected_crop or predicted_part != expected_part:
                    adapter_unavailable["wrong_router"] += 1
                else:
                    adapter_unavailable["missing_adapter"] += 1
            else:
                adapter_unavailable["unknown"] += 1

        prototype_target = str(row.get("prototype_crop") or "")
        prototype_part = str(row.get("prototype_part") or "")
        if prototype_target and prototype_part:
            prototype_target = f"{prototype_target}__{prototype_part}"
        else:
            prototype_target = str(row.get("prototype_target") or "")
        if target and prototype_target == target and reconcile_decision == "abstain":
            prototype_correct_but_abstained.append(str(row.get("image_id") or ""))
        if target in {"unknown_crop", "non_plant"} or target.endswith("__unknown_part"):
            if status == "success" or reconcile_decision in {"accept_router", "use_prototype"}:
                negative_false_accepts.append(str(row.get("image_id") or ""))
        if target and row.get("prototype_similarity") is not None:
            policy_thresholds_by_target.setdefault(
                target,
                {
                    "min_similarity": row.get("prototype_min_similarity"),
                    "min_margin": row.get("prototype_min_margin"),
                    "min_negative_gap": row.get("prototype_min_negative_gap"),
                },
            )

    return {
        "schema_version": "v1_m2_demo_analysis_summary",
        "total": len(items),
        "router_crop_correctness": crop_counts,
        "router_part_correctness": part_counts,
        "normalized_disease_class_correctness": class_counts,
        "adapter_unavailable": adapter_unavailable,
        "prototype_reconciliation": dict(sorted(reconciliation_counts.items())),
        "answered_wrong_by_target": dict(sorted(answered_wrong_by_target.items())),
        "answered_wrong_by_expected_class": dict(sorted(answered_wrong_by_expected_class.items())),
        "prototype_correct_but_abstained": {
            "count": len(prototype_correct_but_abstained),
            "image_ids": prototype_correct_but_abstained[:100],
            "truncated": len(prototype_correct_but_abstained) > 100,
        },
        "negative_false_accepts": {
            "count": len(negative_false_accepts),
            "image_ids": negative_false_accepts[:100],
            "truncated": len(negative_false_accepts) > 100,
        },
        "policy_thresholds_by_target": dict(sorted(policy_thresholds_by_target.items())),
        "opposite_part_disease_labels": {
            "count": len(opposite_part_rows),
            "image_ids": opposite_part_rows[:100],
            "truncated": len(opposite_part_rows) > 100,
        },
        "per_target": dict(sorted(per_target.items())),
    }


def write_analysis_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# M2 Demo Analysis Summary",
        "",
        f"- total: {analysis['total']}",
        f"- router_crop_correctness: `{json.dumps(analysis['router_crop_correctness'], sort_keys=True)}`",
        f"- router_part_correctness: `{json.dumps(analysis['router_part_correctness'], sort_keys=True)}`",
        f"- normalized_disease_class_correctness: `{json.dumps(analysis['normalized_disease_class_correctness'], sort_keys=True)}`",
        f"- adapter_unavailable: `{json.dumps(analysis['adapter_unavailable'], sort_keys=True)}`",
        f"- prototype_reconciliation: `{json.dumps(analysis.get('prototype_reconciliation', {}), sort_keys=True)}`",
        f"- answered_wrong_by_target: `{json.dumps(analysis.get('answered_wrong_by_target', {}), sort_keys=True)}`",
        f"- prototype_correct_but_abstained: {analysis.get('prototype_correct_but_abstained', {}).get('count', 0)}",
        f"- negative_false_accepts: {analysis.get('negative_false_accepts', {}).get('count', 0)}",
        f"- opposite_part_disease_labels: {analysis['opposite_part_disease_labels']['count']}",
        "",
        "## Per Target",
        "",
        "| target | total | answered | abstained_or_reviewed | pass | fail | exact_class_correct | opposite_part_labels |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, values in analysis["per_target"].items():
        lines.append(
            "| {target} | {total} | {answered} | {abstained} | {passed} | {failed} | {exact} | {opposite} |".format(
                target=target,
                total=values["total"],
                answered=values["answered"],
                abstained=values["abstained_or_reviewed"],
                passed=values["pass"],
                failed=values["fail"],
                exact=values["exact_class_correct"],
                opposite=values["opposite_part_disease_labels"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--analysis-markdown-output", type=Path)
    parser.add_argument("--adapter-root", type=Path, default=Path("runs"))
    parser.add_argument("--config-env", default="colab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable-prototype-reconciler", action="store_true")
    parser.add_argument("--prototype-bank", type=Path)
    parser.add_argument("--taxonomy-registry", type=Path)
    parser.add_argument("--prototype-calibration-report", type=Path)
    parser.add_argument("--prototype-min-similarity", type=float)
    parser.add_argument("--prototype-min-margin", type=float)
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
    (
        prototype_min_similarity,
        prototype_min_margin,
        prototype_min_negative_gap,
        calibration_report,
        prototype_target_policies,
    ) = resolve_prototype_thresholds_from_calibration(
        args.prototype_calibration_report,
        min_similarity=args.prototype_min_similarity,
        min_margin=args.prototype_min_margin,
    )

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        result = _run_row(
            row,
            repo_root=repo_root,
            config_env=str(args.config_env),
            device=str(args.device),
            adapter_root=args.adapter_root,
            mode=mode,
            enable_prototype_reconciler=bool(args.enable_prototype_reconciler),
            prototype_bank_path=args.prototype_bank,
            taxonomy_registry_path=args.taxonomy_registry,
            prototype_min_similarity=prototype_min_similarity,
            prototype_min_margin=prototype_min_margin,
            prototype_min_negative_gap=prototype_min_negative_gap,
            prototype_target_policies=prototype_target_policies,
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
        "prototype_reconciler": {
            "enabled": bool(args.enable_prototype_reconciler),
            "prototype_bank": str(args.prototype_bank) if args.prototype_bank else "",
            "taxonomy_registry": str(args.taxonomy_registry) if args.taxonomy_registry else "",
            "prototype_calibration_report": calibration_report,
            "prototype_min_similarity": prototype_min_similarity,
            "prototype_min_margin": prototype_min_margin,
            "prototype_min_negative_gap": prototype_min_negative_gap,
            "prototype_target_policy_count": len(prototype_target_policies),
        },
        "trust_expected_target": mode == "adapter-smoke",
        "summary": summarize_results(output_rows),
        "rows": output_rows,
    }
    analysis = build_analysis_summary(output_rows)
    report["analysis_summary"] = analysis
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(report, args.markdown_output)
    analysis_output = args.analysis_output or (args.output.parent / "analysis_summary.json")
    analysis_markdown_output = args.analysis_markdown_output or (args.output.parent / "analysis_summary.md")
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis_output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    analysis_markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_analysis_markdown(analysis, analysis_markdown_output)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
