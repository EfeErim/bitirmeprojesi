#!/usr/bin/env python3
"""Part-aware SAM box ROI ablation helpers for Colab notebooks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

from PIL import Image

from scripts.colab_router_adapter_inference import run_inference as run_router_inference
from src.core.config_manager import get_config
from src.router.roi_helpers import bbox_area_ratio, extract_roi, sanitize_bbox
from src.shared.json_utils import deep_merge
from src.workflows.inference import InferenceWorkflow
from src.workflows.training import TrainingWorkflow

StatusPrinter = Callable[[str], None]
WorkflowFactory = Callable[..., Any]
RouterRunner = Callable[..., Dict[str, Any]]
TrainingWorkflowFactory = Callable[..., Any]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ADAPTER_ALLOWED_ROUTER_STATUSES = {"ok", "trusted_hint_skipped", "skipped"}

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "full_image_baseline": {
        "notebook": "10_ablation_full_image_baseline.ipynb",
        "output_dir": "docs/ablation_results/full_image_baseline",
        "phase": "inference_only",
        "input_policy": "full_image",
    },
    "primary_roi_inference": {
        "notebook": "11_ablation_primary_roi_inference.ipynb",
        "output_dir": "docs/ablation_results/primary_roi_inference",
        "phase": "inference_only",
        "input_policy": "router_primary_roi",
    },
    "hybrid_roi_fallback": {
        "notebook": "12_ablation_hybrid_roi_fallback.ipynb",
        "output_dir": "docs/ablation_results/hybrid_roi_fallback",
        "phase": "inference_only",
        "input_policy": "router_primary_roi_else_full_image",
    },
    "roi_trained_adapter": {
        "notebook": "13_ablation_roi_trained_adapter.ipynb",
        "output_dir": "docs/ablation_results/roi_trained_adapter",
        "phase": "training_research",
        "input_policy": "train_val_test_router_primary_roi",
    },
    "mixed_full_roi_training": {
        "notebook": "14_ablation_mixed_full_roi_training.ipynb",
        "output_dir": "docs/ablation_results/mixed_full_roi_training",
        "phase": "training_research",
        "input_policy": "train_full_image_plus_router_primary_roi",
    },
    "roi_quality_audit": {
        "notebook": "15_ablation_roi_quality_audit.ipynb",
        "output_dir": "docs/ablation_results/roi_quality_audit",
        "phase": "audit_only",
        "input_policy": "router_primary_roi_quality",
    },
    "dual_view_inference": {
        "notebook": "16_ablation_dual_view_inference.ipynb",
        "output_dir": "docs/ablation_results/dual_view_inference",
        "phase": "inference_only",
        "input_policy": "full_image_plus_router_primary_roi_score_fusion",
    },
    "dual_view_trained_adapter": {
        "notebook": "17_ablation_dual_view_trained_adapter.ipynb",
        "output_dir": "docs/ablation_results/dual_view_trained_adapter",
        "phase": "training_research",
        "input_policy": "paired_full_image_plus_router_primary_roi_after_dual_view_gate",
    },
}


def _emit_status(status_printer: Optional[StatusPrinter], message: str) -> None:
    if status_printer is not None:
        status_printer(str(message))


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _router_payload(router_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = router_result.get("router")
    return dict(payload) if isinstance(payload, dict) else {}


def _primary_detection(router_result: Dict[str, Any]) -> Dict[str, Any]:
    router_payload = _router_payload(router_result)
    primary = router_payload.get("primary_detection")
    if isinstance(primary, dict):
        return dict(primary)
    details = router_result.get("router_details")
    if isinstance(details, dict) and isinstance(details.get("primary_detection"), dict):
        return dict(details["primary_detection"])
    return {}


def resolve_router_handoff(router_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the adapter handoff fields from a router result."""
    primary = _primary_detection(router_result)
    crop = _normalize_text(router_result.get("crop") or primary.get("crop"))
    part = _normalize_text(router_result.get("part") or primary.get("part"))
    status = _normalize_text(router_result.get("status") or _router_payload(router_result).get("status"))
    return {
        "status": status,
        "crop": crop or None,
        "part": part or None,
        "router_confidence": float(router_result.get("router_confidence", primary.get("crop_confidence", 0.0)) or 0.0),
        "bbox": primary.get("bbox"),
        "primary_detection": primary,
        "adapter_allowed": (
            status in ADAPTER_ALLOWED_ROUTER_STATUSES
            and bool(crop)
            and crop != "unknown"
            and bool(part)
            and part != "unknown"
        ),
    }


def prepare_primary_roi(image: Image.Image, bbox: Any, *, pad_ratio: float = 0.08) -> tuple[Optional[Image.Image], Optional[list[float]], float]:
    """Return the padded primary ROI crop, sanitized bbox, and bbox area ratio."""
    width, height = image.size
    sanitized = sanitize_bbox(bbox, width, height)
    if sanitized is None:
        return None, None, 0.0
    return extract_roi(image, sanitized, pad_ratio=pad_ratio), sanitized, bbox_area_ratio(sanitized, width, height)


def classify_roi_quality(
    *,
    bbox: Optional[list[float]],
    area_ratio: float,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.90,
) -> str:
    """Classify whether a sanitized bbox is usable as an adapter ROI view."""
    if bbox is None:
        return "roi_missing"
    if float(area_ratio) < float(min_area_ratio):
        return "roi_too_small"
    if float(area_ratio) > float(max_area_ratio):
        return "roi_too_large"
    return "roi_ok"


def _prediction_to_row(
    *,
    ablation_name: str,
    image_path: Path,
    expected_label: Optional[str],
    input_view: str,
    status: str,
    crop: Optional[str],
    part: Optional[str],
    router_status: str,
    router_confidence: float,
    bbox: Optional[list[float]],
    area_ratio: float,
    prediction: Optional[Dict[str, Any]],
    latency_ms: float,
) -> Dict[str, Any]:
    prediction = dict(prediction or {})
    ood = prediction.get("ood_analysis")
    ood_payload = dict(ood) if isinstance(ood, dict) else {}
    return {
        "ablation_name": ablation_name,
        "image_path": str(image_path),
        "expected_label": expected_label,
        "crop": crop,
        "part": part,
        "router_status": router_status,
        "router_confidence": float(router_confidence),
        "bbox": bbox,
        "bbox_area_ratio": float(area_ratio),
        "input_view": input_view,
        "diagnosis": prediction.get("diagnosis"),
        "confidence": float(prediction.get("confidence", 0.0) or 0.0),
        "ood_is_ood": ood_payload.get("is_ood"),
        "ood_primary_score": ood_payload.get("primary_score"),
        "latency_ms": float(latency_ms),
        "status": status,
    }


def _run_prediction(
    workflow: Any,
    image_input: Any,
    *,
    crop: str,
    part: str,
    return_ood: bool,
) -> tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    payload = workflow.predict(
        image_input,
        crop_hint=crop,
        part_hint=part,
        return_ood=return_ood,
        trust_crop_hint=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return dict(payload), latency_ms


def _resolve_workflow(
    *,
    workflow_factory: WorkflowFactory,
    config_env: Optional[str],
    device: str,
    adapter_root: Optional[str | Path],
    status_printer: Optional[StatusPrinter],
) -> Any:
    return workflow_factory(
        environment=config_env,
        device=device,
        adapter_root=adapter_root,
        status_callback=status_printer,
    )


def run_ablation_image(
    image_path: str | Path,
    *,
    ablation_name: str,
    expected_label: Optional[str] = None,
    adapter_crop: Optional[str] = None,
    adapter_part: Optional[str] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    adapter_root: Optional[str | Path] = None,
    return_ood: bool = True,
    status_printer: Optional[StatusPrinter] = None,
    workflow_factory: WorkflowFactory = InferenceWorkflow,
    workflow: Optional[Any] = None,
    router_runner: RouterRunner = run_router_inference,
) -> list[Dict[str, Any]]:
    """Run one ablation condition for a single image and return flat report rows."""
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unsupported ablation_name={ablation_name!r}. Expected one of {sorted(ABLATION_CONFIGS)}")
    config = ABLATION_CONFIGS[ablation_name]
    if config["phase"] != "inference_only":
        raise ValueError(f"{ablation_name} is a training-phase ablation; use describe_training_ablation_plan().")

    image_ref = Path(image_path)
    adapter_crop_name = _normalize_text(adapter_crop) or None
    adapter_part_name = _normalize_text(adapter_part) or None
    workflow = workflow or _resolve_workflow(
        workflow_factory=workflow_factory,
        config_env=config_env,
        device=device,
        adapter_root=adapter_root,
        status_printer=status_printer,
    )

    if config["input_policy"] == "full_image":
        started = time.perf_counter()
        payload = workflow.predict(
            image_ref,
            crop_hint=adapter_crop_name,
            part_hint=adapter_part_name,
            return_ood=return_ood,
            trust_crop_hint=bool(adapter_crop_name),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        handoff = resolve_router_handoff(payload)
        crop = adapter_crop_name or payload.get("crop") or handoff["crop"]
        part = adapter_part_name or payload.get("part") or handoff["part"]
        return [
            _prediction_to_row(
                ablation_name=ablation_name,
                image_path=image_ref,
                expected_label=expected_label,
                input_view="full_image",
                status=str(payload.get("status", "unknown")),
                crop=crop,
                part=part,
                router_status=handoff["status"],
                router_confidence=handoff["router_confidence"],
                bbox=handoff["bbox"] if isinstance(handoff["bbox"], list) else None,
                area_ratio=0.0,
                prediction=payload,
                latency_ms=latency_ms,
            )
        ]

    router_result = router_runner(
        image_ref,
        config_env=config_env,
        device=device,
        status_printer=status_printer,
        include_diagnostics=True,
        include_adapter_target=True,
    )
    handoff = resolve_router_handoff(router_result)
    crop = str(adapter_crop_name or handoff["crop"] or "")
    part = str(adapter_part_name or handoff["part"] or "")
    adapter_allowed = bool(adapter_crop_name and adapter_part_name) or bool(handoff["adapter_allowed"])
    if not adapter_allowed:
        return [
            _prediction_to_row(
                ablation_name=ablation_name,
                image_path=image_ref,
                expected_label=expected_label,
                input_view=str(config["input_policy"]),
                status="adapter_skipped",
                crop=crop or handoff["crop"],
                part=part or handoff["part"],
                router_status=handoff["status"],
                router_confidence=handoff["router_confidence"],
                bbox=handoff["bbox"] if isinstance(handoff["bbox"], list) else None,
                area_ratio=0.0,
                prediction=None,
                latency_ms=0.0,
            )
        ]

    with Image.open(image_ref) as opened:
        image = opened.convert("RGB")
        roi_image, sanitized_bbox, area_ratio = prepare_primary_roi(image, handoff["bbox"])

    if roi_image is not None:
        prediction, latency_ms = _run_prediction(
            workflow,
            roi_image,
            crop=crop,
            part=part,
            return_ood=return_ood,
        )
        return [
            _prediction_to_row(
                ablation_name=ablation_name,
                image_path=image_ref,
                expected_label=expected_label,
                input_view="router_primary_roi",
                status=str(prediction.get("status", "success")),
                crop=crop,
                part=part,
                router_status=handoff["status"],
                router_confidence=handoff["router_confidence"],
                bbox=sanitized_bbox,
                area_ratio=area_ratio,
                prediction=prediction,
                latency_ms=latency_ms,
            )
        ]

    if config["input_policy"] == "router_primary_roi":
        return [
            _prediction_to_row(
                ablation_name=ablation_name,
                image_path=image_ref,
                expected_label=expected_label,
                input_view="router_primary_roi",
                status="roi_missing",
                crop=crop,
                part=part,
                router_status=handoff["status"],
                router_confidence=handoff["router_confidence"],
                bbox=None,
                area_ratio=0.0,
                prediction=None,
                latency_ms=0.0,
            )
        ]

    prediction, latency_ms = _run_prediction(
        workflow,
        image_ref,
        crop=crop,
        part=part,
        return_ood=return_ood,
    )
    return [
        _prediction_to_row(
            ablation_name=ablation_name,
            image_path=image_ref,
            expected_label=expected_label,
            input_view="fallback_full_image",
            status="fallback_full_image",
            crop=crop,
            part=part,
            router_status=handoff["status"],
            router_confidence=handoff["router_confidence"],
            bbox=None,
            area_ratio=0.0,
            prediction=prediction,
            latency_ms=latency_ms,
        )
    ]


def discover_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _infer_expected_label(path: Path, *, label_from_parent: bool) -> Optional[str]:
    return path.parent.name if label_from_parent else None


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    by_group: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    by_image: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows_list:
        by_group[f"{row.get('crop') or 'unknown'}::{row.get('part') or 'unknown'}"].append(row)
        by_image[str(row.get("image_path") or "")].append(row)

    comparable = [row for row in rows_list if row.get("expected_label") and row.get("diagnosis")]
    correct = [row for row in comparable if str(row["expected_label"]).strip().lower() == str(row["diagnosis"]).strip().lower()]
    labels = sorted({str(row["expected_label"]).strip().lower() for row in comparable})
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for row in comparable if str(row["expected_label"]).strip().lower() == label and str(row["diagnosis"]).strip().lower() == label)
        fp = sum(1 for row in comparable if str(row["expected_label"]).strip().lower() != label and str(row["diagnosis"]).strip().lower() == label)
        fn = sum(1 for row in comparable if str(row["expected_label"]).strip().lower() == label and str(row["diagnosis"]).strip().lower() != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_values.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)

    per_crop_part = {}
    for group, group_rows in sorted(by_group.items()):
        group_comparable = [row for row in group_rows if row.get("expected_label") and row.get("diagnosis")]
        group_correct = [
            row
            for row in group_comparable
            if str(row["expected_label"]).strip().lower() == str(row["diagnosis"]).strip().lower()
        ]
        per_crop_part[group] = {
            "sample_count": len(group_rows),
            "comparable_count": len(group_comparable),
            "accuracy": (len(group_correct) / len(group_comparable)) if group_comparable else None,
        }

    paired_groups = [
        group_rows
        for group_rows in by_image.values()
        if len([row for row in group_rows if row.get("diagnosis")]) >= 2
    ]
    disagreement_count = 0
    ood_flip_count = 0
    confidence_deltas: list[float] = []
    for group_rows in paired_groups:
        predicted = [row for row in group_rows if row.get("diagnosis")]
        diagnoses = {str(row.get("diagnosis") or "") for row in predicted}
        if len(diagnoses) > 1:
            disagreement_count += 1
        ood_values = {row.get("ood_is_ood") for row in predicted if row.get("ood_is_ood") is not None}
        if len(ood_values) > 1:
            ood_flip_count += 1
        confidences = [float(row.get("confidence", 0.0) or 0.0) for row in predicted]
        if confidences:
            confidence_deltas.append(max(confidences) - min(confidences))

    return {
        "sample_count": len(rows_list),
        "comparable_count": len(comparable),
        "accuracy": (len(correct) / len(comparable)) if comparable else None,
        "macro_f1": (sum(f1_values) / len(f1_values)) if f1_values else None,
        "roi_missing_rate": (
            sum(1 for row in rows_list if row.get("status") == "roi_missing") / len(rows_list)
        ) if rows_list else 0.0,
        "fallback_rate": (
            sum(1 for row in rows_list if row.get("input_view") == "fallback_full_image") / len(rows_list)
        ) if rows_list else 0.0,
        "ood_flip_rate": (ood_flip_count / len(paired_groups)) if paired_groups else None,
        "prediction_disagreement_rate": (disagreement_count / len(paired_groups)) if paired_groups else None,
        "confidence_delta": (sum(confidence_deltas) / len(confidence_deltas)) if confidence_deltas else None,
        "per_crop_part": per_crop_part,
    }


def write_report(rows: list[Dict[str, Any]], *, output_dir: str | Path, ablation_name: str) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    payload = {
        "ablation_name": ablation_name,
        "config": ABLATION_CONFIGS[ablation_name],
        "summary": summary,
        "rows": rows,
    }
    (output_root / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        fieldnames = list(rows[0].keys())
        with (output_root / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return payload


def _run_git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=check,
        text=True,
        capture_output=True,
    )


def _tokenized_git_remote_url(remote_url: str) -> Optional[str]:
    token = str(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        return None
    parsed = urlsplit(str(remote_url or "").strip())
    if parsed.scheme != "https" or "github.com" not in parsed.netloc:
        return None
    netloc = parsed.netloc.split("@", 1)[-1]
    return urlunsplit((parsed.scheme, f"x-access-token:{token}@{netloc}", parsed.path, parsed.query, parsed.fragment))


def _configure_tokenized_push_url(*, cwd: Path) -> None:
    remote = _run_git(["remote", "get-url", "origin"], cwd=cwd, check=False)
    if remote.returncode != 0:
        return
    push_url = _tokenized_git_remote_url(remote.stdout.strip())
    if push_url:
        _run_git(["remote", "set-url", "--push", "origin", push_url], cwd=cwd, check=False)


def commit_and_push_ablation_results(
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    message: str,
    push: bool = True,
) -> Dict[str, Any]:
    """Commit and optionally push one ablation output directory from a Colab checkout."""
    root = Path(repo_root)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    root = root.resolve()
    try:
        relative_output = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output_dir must be inside repo_root: output_dir={output_path} repo_root={root}") from exc

    _run_git(["config", "user.email", "aads-colab@example.local"], cwd=root, check=False)
    _run_git(["config", "user.name", "AADS Colab"], cwd=root, check=False)
    _run_git(["add", str(relative_output).replace("\\", "/")], cwd=root)
    status = _run_git(["status", "--porcelain", "--", str(relative_output).replace("\\", "/")], cwd=root)
    if not status.stdout.strip():
        if push:
            _configure_tokenized_push_url(cwd=root)
            push_result = _run_git(["push"], cwd=root, check=False)
            return {
                "status": "nothing_to_commit",
                "output_dir": str(relative_output),
                "push_returncode": int(push_result.returncode),
                "push_stdout": push_result.stdout,
                "push_stderr": push_result.stderr,
            }
        return {"status": "nothing_to_commit", "output_dir": str(relative_output)}

    commit = _run_git(["commit", "-m", message], cwd=root, check=False)
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed:\nSTDOUT:\n{commit.stdout}\nSTDERR:\n{commit.stderr}")

    payload: Dict[str, Any] = {
        "status": "committed",
        "output_dir": str(relative_output),
        "commit_stdout": commit.stdout,
        "commit_stderr": commit.stderr,
    }
    if push:
        _configure_tokenized_push_url(cwd=root)
        push_result = _run_git(["push"], cwd=root, check=False)
        payload.update(
            {
                "push_returncode": int(push_result.returncode),
                "push_stdout": push_result.stdout,
                "push_stderr": push_result.stderr,
            }
        )
        if push_result.returncode != 0:
            raise RuntimeError(f"git push failed:\nSTDOUT:\n{push_result.stdout}\nSTDERR:\n{push_result.stderr}")
        payload["status"] = "committed_and_pushed"
    return payload


def run_ablation_folder(
    image_dir: str | Path,
    *,
    ablation_name: str,
    output_dir: Optional[str | Path] = None,
    label_from_parent: bool = True,
    adapter_crop: Optional[str] = None,
    adapter_part: Optional[str] = None,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    adapter_root: Optional[str | Path] = None,
    return_ood: bool = True,
    status_printer: Optional[StatusPrinter] = print,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    images = discover_images(image_dir)
    _emit_status(status_printer, f"[ABLATION] {ablation_name} images={len(images)}")
    workflow = _resolve_workflow(
        workflow_factory=InferenceWorkflow,
        config_env=config_env,
        device=device,
        adapter_root=adapter_root,
        status_printer=status_printer,
    )
    for index, image_path in enumerate(images, start=1):
        _emit_status(status_printer, f"[ABLATION] {index}/{len(images)} {image_path.name}")
        rows.extend(
            run_ablation_image(
                image_path,
                ablation_name=ablation_name,
                expected_label=_infer_expected_label(image_path, label_from_parent=label_from_parent),
                adapter_crop=adapter_crop,
                adapter_part=adapter_part,
                config_env=config_env,
                device=device,
                adapter_root=adapter_root,
                return_ood=return_ood,
                status_printer=status_printer,
                workflow=workflow,
            )
        )
    resolved_output_dir = output_dir or ABLATION_CONFIGS[ablation_name]["output_dir"]
    return write_report(rows, output_dir=resolved_output_dir, ablation_name=ablation_name)


def audit_roi_quality_image(
    image_path: str | Path,
    *,
    expected_label: Optional[str] = None,
    expected_crop: str = "tomato",
    expected_part: str = "fruit",
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.90,
    status_printer: Optional[StatusPrinter] = None,
    router_runner: RouterRunner = run_router_inference,
) -> Dict[str, Any]:
    image_ref = Path(image_path)
    started = time.perf_counter()
    router_result = router_runner(
        image_ref,
        config_env=config_env,
        device=device,
        include_diagnostics=True,
        include_adapter_target=False,
        status_printer=status_printer,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    handoff = resolve_router_handoff(router_result)
    with Image.open(image_ref) as opened:
        image = opened.convert("RGB")
        _roi_image, sanitized_bbox, area_ratio = prepare_primary_roi(image, handoff["bbox"])
    quality_status = classify_roi_quality(
        bbox=sanitized_bbox,
        area_ratio=area_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    router_crop = str(handoff["crop"] or "")
    router_part = str(handoff["part"] or "")
    return {
        "ablation_name": "roi_quality_audit",
        "image_path": str(image_ref),
        "expected_label": expected_label,
        "expected_crop": expected_crop,
        "expected_part": expected_part,
        "router_crop": router_crop,
        "router_part": router_part,
        "router_status": handoff["status"],
        "router_confidence": handoff["router_confidence"],
        "crop_matches_expected": router_crop == str(expected_crop).strip().lower(),
        "part_matches_expected": router_part == str(expected_part).strip().lower(),
        "bbox": sanitized_bbox,
        "bbox_area_ratio": float(area_ratio),
        "roi_quality_status": quality_status,
        "latency_ms": float(latency_ms),
        "status": quality_status,
    }


def summarize_roi_quality_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    total = len(rows_list)
    by_status = defaultdict(int)
    by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    crop_mismatch = 0
    part_mismatch = 0
    area_values: list[float] = []
    for row in rows_list:
        status = str(row.get("roi_quality_status") or row.get("status") or "unknown")
        label = str(row.get("expected_label") or "unknown")
        by_status[status] += 1
        by_label[label][status] += 1
        if row.get("crop_matches_expected") is False:
            crop_mismatch += 1
        if row.get("part_matches_expected") is False:
            part_mismatch += 1
        if row.get("bbox") is not None:
            area_values.append(float(row.get("bbox_area_ratio", 0.0) or 0.0))
    return {
        "sample_count": total,
        "roi_ok_count": int(by_status.get("roi_ok", 0)),
        "roi_missing_count": int(by_status.get("roi_missing", 0)),
        "roi_low_quality_count": int(by_status.get("roi_too_small", 0) + by_status.get("roi_too_large", 0)),
        "roi_ok_rate": (float(by_status.get("roi_ok", 0)) / total) if total else 0.0,
        "roi_missing_rate": (float(by_status.get("roi_missing", 0)) / total) if total else 0.0,
        "crop_mismatch_rate": (float(crop_mismatch) / total) if total else 0.0,
        "part_mismatch_rate": (float(part_mismatch) / total) if total else 0.0,
        "mean_bbox_area_ratio": (sum(area_values) / len(area_values)) if area_values else None,
        "status_counts": dict(sorted(by_status.items())),
        "per_label_status": {
            label: dict(sorted(statuses.items()))
            for label, statuses in sorted(by_label.items())
        },
    }


def write_roi_quality_report(rows: list[Dict[str, Any]], *, output_dir: str | Path) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_roi_quality_rows(rows)
    payload = {
        "ablation_name": "roi_quality_audit",
        "config": ABLATION_CONFIGS["roi_quality_audit"],
        "summary": summary,
        "rows": rows,
    }
    (output_root / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (output_root / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return payload


def run_roi_quality_audit_folder(
    image_dir: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    label_from_parent: bool = True,
    expected_crop: str = "tomato",
    expected_part: str = "fruit",
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    status_printer: Optional[StatusPrinter] = print,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    images = discover_images(image_dir)
    _emit_status(status_printer, f"[ROI_AUDIT] images={len(images)}")
    for index, image_path in enumerate(images, start=1):
        _emit_status(status_printer, f"[ROI_AUDIT] {index}/{len(images)} {image_path.name}")
        rows.append(
            audit_roi_quality_image(
                image_path,
                expected_label=_infer_expected_label(image_path, label_from_parent=label_from_parent),
                expected_crop=expected_crop,
                expected_part=expected_part,
                config_env=config_env,
                device=device,
                status_printer=status_printer,
            )
        )
    return write_roi_quality_report(
        rows,
        output_dir=output_dir or ABLATION_CONFIGS["roi_quality_audit"]["output_dir"],
    )


def run_dual_view_inference_image(
    image_path: str | Path,
    *,
    expected_label: Optional[str] = None,
    adapter_crop: str = "tomato",
    adapter_part: str = "fruit",
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    adapter_root: Optional[str | Path] = None,
    return_ood: bool = True,
    roi_confidence_margin: float = 0.10,
    require_semantic_roi_match: bool = True,
    status_printer: Optional[StatusPrinter] = None,
    workflow_factory: WorkflowFactory = InferenceWorkflow,
    workflow: Optional[Any] = None,
    router_runner: RouterRunner = run_router_inference,
) -> Dict[str, Any]:
    image_ref = Path(image_path)
    crop = _normalize_text(adapter_crop)
    part = _normalize_text(adapter_part)
    workflow = workflow or _resolve_workflow(
        workflow_factory=workflow_factory,
        config_env=config_env,
        device=device,
        adapter_root=adapter_root,
        status_printer=status_printer,
    )
    router_result = router_runner(
        image_ref,
        config_env=config_env,
        device=device,
        include_diagnostics=True,
        include_adapter_target=False,
        status_printer=status_printer,
    )
    handoff = resolve_router_handoff(router_result)
    with Image.open(image_ref) as opened:
        image = opened.convert("RGB")
        roi_image, sanitized_bbox, area_ratio = prepare_primary_roi(image, handoff["bbox"])
    roi_quality_status = classify_roi_quality(bbox=sanitized_bbox, area_ratio=area_ratio)
    semantic_roi_match = handoff["crop"] == crop and handoff["part"] == part
    roi_eligible = roi_quality_status == "roi_ok" and roi_image is not None
    if require_semantic_roi_match:
        roi_eligible = roi_eligible and semantic_roi_match

    full_prediction, full_latency_ms = _run_prediction(
        workflow,
        image_ref,
        crop=crop,
        part=part,
        return_ood=return_ood,
    )
    roi_prediction: Optional[Dict[str, Any]] = None
    roi_latency_ms = 0.0
    if roi_eligible:
        roi_prediction, roi_latency_ms = _run_prediction(
            workflow,
            roi_image,
            crop=crop,
            part=part,
            return_ood=return_ood,
        )

    full_confidence = float(full_prediction.get("confidence", 0.0) or 0.0)
    roi_confidence = float((roi_prediction or {}).get("confidence", 0.0) or 0.0)
    selected_view = "full_image"
    selected_prediction = full_prediction
    status = str(full_prediction.get("status", "success"))
    if roi_prediction is not None and roi_confidence >= full_confidence + float(roi_confidence_margin):
        selected_view = "router_primary_roi"
        selected_prediction = roi_prediction
        status = str(roi_prediction.get("status", "success"))
    elif roi_prediction is None:
        status = "semantic_mismatch_fallback" if require_semantic_roi_match and not semantic_roi_match else "fallback_full_image"

    full_ood = dict(full_prediction.get("ood_analysis", {}) or {})
    roi_ood = dict((roi_prediction or {}).get("ood_analysis", {}) or {})
    selected_ood = dict(selected_prediction.get("ood_analysis", {}) or {})
    full_diagnosis = full_prediction.get("diagnosis")
    roi_diagnosis = (roi_prediction or {}).get("diagnosis")
    return {
        "ablation_name": "dual_view_inference",
        "image_path": str(image_ref),
        "expected_label": expected_label,
        "crop": crop,
        "part": part,
        "router_status": handoff["status"],
        "router_confidence": handoff["router_confidence"],
        "router_crop": handoff["crop"],
        "router_part": handoff["part"],
        "bbox": sanitized_bbox,
        "bbox_area_ratio": float(area_ratio),
        "roi_quality_status": roi_quality_status,
        "semantic_roi_match": semantic_roi_match,
        "require_semantic_roi_match": bool(require_semantic_roi_match),
        "roi_eligible": bool(roi_eligible),
        "input_view": selected_view,
        "selected_view": selected_view,
        "diagnosis": selected_prediction.get("diagnosis"),
        "confidence": float(selected_prediction.get("confidence", 0.0) or 0.0),
        "ood_is_ood": selected_ood.get("is_ood"),
        "ood_primary_score": selected_ood.get("primary_score"),
        "full_diagnosis": full_diagnosis,
        "full_confidence": full_confidence,
        "full_ood_is_ood": full_ood.get("is_ood"),
        "roi_diagnosis": roi_diagnosis,
        "roi_confidence": roi_confidence if roi_prediction is not None else None,
        "roi_ood_is_ood": roi_ood.get("is_ood") if roi_prediction is not None else None,
        "dual_view_disagreement": bool(roi_prediction is not None and full_diagnosis != roi_diagnosis),
        "confidence_delta_roi_minus_full": (roi_confidence - full_confidence) if roi_prediction is not None else None,
        "latency_ms": float(full_latency_ms + roi_latency_ms),
        "status": status,
    }


def run_dual_view_inference_folder(
    image_dir: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    label_from_parent: bool = True,
    adapter_crop: str = "tomato",
    adapter_part: str = "fruit",
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    adapter_root: Optional[str | Path] = None,
    return_ood: bool = True,
    roi_confidence_margin: float = 0.10,
    require_semantic_roi_match: bool = True,
    status_printer: Optional[StatusPrinter] = print,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    images = discover_images(image_dir)
    _emit_status(status_printer, f"[DUAL_VIEW] images={len(images)}")
    workflow = _resolve_workflow(
        workflow_factory=InferenceWorkflow,
        config_env=config_env,
        device=device,
        adapter_root=adapter_root,
        status_printer=status_printer,
    )
    for index, image_path in enumerate(images, start=1):
        _emit_status(status_printer, f"[DUAL_VIEW] {index}/{len(images)} {image_path.name}")
        rows.append(
            run_dual_view_inference_image(
                image_path,
                expected_label=_infer_expected_label(image_path, label_from_parent=label_from_parent),
                adapter_crop=adapter_crop,
                adapter_part=adapter_part,
                config_env=config_env,
                device=device,
                adapter_root=adapter_root,
                return_ood=return_ood,
                roi_confidence_margin=roi_confidence_margin,
                require_semantic_roi_match=require_semantic_roi_match,
                status_printer=status_printer,
                workflow=workflow,
            )
        )
    return write_report(
        rows,
        output_dir=output_dir or ABLATION_CONFIGS["dual_view_inference"]["output_dir"],
        ablation_name="dual_view_inference",
    )


def _copy_image_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def build_mixed_full_roi_dataset(
    source_dataset_root: str | Path,
    *,
    dataset_key: str = "tomato__fruit",
    output_root: str | Path,
    roi_splits: Iterable[str] = ("continual",),
    pad_ratio: float = 0.08,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    router_runner: RouterRunner = run_router_inference,
    status_printer: Optional[StatusPrinter] = print,
) -> Dict[str, Any]:
    """Create a research-only dataset view with full-image samples plus router ROI crops.

    The returned root follows the runtime dataset layout expected by TrainingWorkflow:
    ``<output_root>/<dataset_key>/continual|val|test|ood|oe``. In-distribution splits
    are copied as full images; splits listed in ``roi_splits`` also receive ROI crops
    when the router provides a valid primary bbox.
    """
    source_root = Path(source_dataset_root)
    source_dataset = source_root / dataset_key
    if not source_dataset.is_dir():
        raise FileNotFoundError(f"source dataset not found: {source_dataset}")

    target_root = Path(output_root)
    target_dataset = target_root / dataset_key
    if target_dataset.exists():
        shutil.rmtree(target_dataset)
    roi_split_names = {str(split).strip().lower() for split in roi_splits}
    manifest: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "source_dataset": str(source_dataset),
        "target_dataset": str(target_dataset),
        "roi_splits": sorted(roi_split_names),
        "pad_ratio": float(pad_ratio),
        "splits": {},
    }

    for split in ("continual", "val", "test"):
        source_split = source_dataset / split
        if not source_split.is_dir():
            continue
        split_stats = {
            "full_images": 0,
            "roi_images": 0,
            "roi_missing": 0,
            "classes": {},
        }
        for class_dir in sorted(path for path in source_split.iterdir() if path.is_dir()):
            class_stats = {"full_images": 0, "roi_images": 0, "roi_missing": 0}
            for image_path in discover_images(class_dir):
                full_name = f"full__{_safe_stem(image_path)}{image_path.suffix.lower()}"
                _copy_image_file(image_path, target_dataset / split / class_dir.name / full_name)
                split_stats["full_images"] += 1
                class_stats["full_images"] += 1

                if split.lower() not in roi_split_names:
                    continue
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    router_result = router_runner(
                        image_path,
                        config_env=config_env,
                        device=device,
                        include_adapter_target=False,
                        status_printer=status_printer,
                    )
                    handoff = resolve_router_handoff(router_result)
                    roi, _bbox, _area_ratio = prepare_primary_roi(image, handoff["bbox"], pad_ratio=pad_ratio)
                    if roi is None:
                        split_stats["roi_missing"] += 1
                        class_stats["roi_missing"] += 1
                        continue
                    roi_name = f"roi__{_safe_stem(image_path)}.jpg"
                    roi_path = target_dataset / split / class_dir.name / roi_name
                    roi_path.parent.mkdir(parents=True, exist_ok=True)
                    roi.save(roi_path, quality=95)
                    split_stats["roi_images"] += 1
                    class_stats["roi_images"] += 1
            split_stats["classes"][class_dir.name] = class_stats
        manifest["splits"][split] = split_stats

    for split in ("ood", "oe"):
        source_split = source_dataset / split
        if not source_split.is_dir():
            continue
        copied = 0
        for image_path in discover_images(source_split):
            relative = image_path.relative_to(source_split)
            _copy_image_file(image_path, target_dataset / split / relative)
            copied += 1
        manifest["splits"][split] = {"full_images": copied, "roi_images": 0, "roi_missing": 0}

    manifest_path = target_dataset / "mixed_full_roi_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_roi_only_dataset(
    source_dataset_root: str | Path,
    *,
    dataset_key: str = "tomato__fruit",
    output_root: str | Path,
    roi_splits: Iterable[str] = ("continual", "val", "test"),
    pad_ratio: float = 0.08,
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    router_runner: RouterRunner = run_router_inference,
    status_printer: Optional[StatusPrinter] = print,
) -> Dict[str, Any]:
    """Create a research-only dataset view containing only valid router ROI crops."""
    source_root = Path(source_dataset_root)
    source_dataset = source_root / dataset_key
    if not source_dataset.is_dir():
        raise FileNotFoundError(f"source dataset not found: {source_dataset}")

    target_root = Path(output_root)
    target_dataset = target_root / dataset_key
    if target_dataset.exists():
        shutil.rmtree(target_dataset)
    roi_split_names = {str(split).strip().lower() for split in roi_splits}
    manifest: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "source_dataset": str(source_dataset),
        "target_dataset": str(target_dataset),
        "roi_splits": sorted(roi_split_names),
        "pad_ratio": float(pad_ratio),
        "splits": {},
    }

    for split in ("continual", "val", "test"):
        source_split = source_dataset / split
        if not source_split.is_dir():
            continue
        split_stats = {
            "full_images": 0,
            "roi_images": 0,
            "roi_missing": 0,
            "classes": {},
        }
        for class_dir in sorted(path for path in source_split.iterdir() if path.is_dir()):
            class_stats = {"full_images": 0, "roi_images": 0, "roi_missing": 0}
            for image_path in discover_images(class_dir):
                if split.lower() not in roi_split_names:
                    continue
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    router_result = router_runner(
                        image_path,
                        config_env=config_env,
                        device=device,
                        include_adapter_target=False,
                        status_printer=status_printer,
                    )
                    handoff = resolve_router_handoff(router_result)
                    roi, _bbox, _area_ratio = prepare_primary_roi(image, handoff["bbox"], pad_ratio=pad_ratio)
                    if roi is None:
                        split_stats["roi_missing"] += 1
                        class_stats["roi_missing"] += 1
                        continue
                    roi_name = f"roi__{_safe_stem(image_path)}.jpg"
                    roi_path = target_dataset / split / class_dir.name / roi_name
                    roi_path.parent.mkdir(parents=True, exist_ok=True)
                    roi.save(roi_path, quality=95)
                    split_stats["roi_images"] += 1
                    class_stats["roi_images"] += 1
            split_stats["classes"][class_dir.name] = class_stats
        manifest["splits"][split] = split_stats

    for split in ("ood", "oe"):
        source_split = source_dataset / split
        if not source_split.is_dir():
            continue
        copied = 0
        for image_path in discover_images(source_split):
            relative = image_path.relative_to(source_split)
            _copy_image_file(image_path, target_dataset / split / relative)
            copied += 1
        manifest["splits"][split] = {"full_images": copied, "roi_images": 0, "roi_missing": 0}

    manifest_path = target_dataset / "roi_only_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _training_config_for_ablation(
    *,
    environment: Optional[str],
    dataset_root: Path,
    dataset_key: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = get_config(environment=environment)
    config = deep_merge(config, overrides or {})
    continual = config.setdefault("training", {}).setdefault("continual", {})
    ood_cfg = continual.setdefault("ood", {})
    data_cfg = continual.setdefault("data", {})
    dataset_dir = dataset_root / dataset_key
    ood_dir = dataset_dir / "ood"
    oe_dir = dataset_dir / "oe"
    if ood_dir.is_dir():
        ood_cfg["ood_root"] = str(ood_dir)
    if oe_dir.is_dir():
        ood_cfg["oe_root"] = str(oe_dir)
        ood_cfg["oe_enabled"] = True
    else:
        ood_cfg["oe_enabled"] = False
        ood_cfg["oe_root"] = ""
    data_cfg["allow_under_min_training"] = True
    return config


def run_mixed_full_roi_training_ablation(
    *,
    source_dataset_root: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path,
    dataset_key: str = "tomato__fruit",
    crop_name: str = "tomato",
    part_name: str = "fruit",
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    training_config_overrides: Optional[Dict[str, Any]] = None,
    return_ood: bool = True,
    status_printer: Optional[StatusPrinter] = print,
    training_workflow_factory: TrainingWorkflowFactory = TrainingWorkflow,
) -> Dict[str, Any]:
    """Run Notebook 14 end to end: mixed dataset, training, and three inference reports."""
    docs_output_dir = Path(output_dir)
    work_root = Path(runtime_root)
    dataset_root = work_root / "mixed_dataset"
    training_output_root = work_root / "training_outputs"
    manifest = build_mixed_full_roi_dataset(
        source_dataset_root,
        dataset_key=dataset_key,
        output_root=dataset_root,
        config_env=config_env,
        device=device,
        status_printer=status_printer,
    )

    config_overrides = dict(training_config_overrides or {})
    continual_overrides: Dict[str, Any] = {}
    if batch_size is not None:
        continual_overrides["batch_size"] = int(batch_size)
    if learning_rate is not None:
        continual_overrides["learning_rate"] = float(learning_rate)
    if continual_overrides:
        config_overrides = deep_merge(config_overrides, {"training": {"continual": continual_overrides}})
    config = _training_config_for_ablation(
        environment=config_env,
        dataset_root=dataset_root,
        dataset_key=dataset_key,
        overrides=config_overrides,
    )
    training = training_workflow_factory(config=config, environment=None, device=device)
    _emit_status(status_printer, "[ABLATION14] training mixed full+ROI adapter")
    result = training.run(
        crop_name=crop_name,
        part_name=part_name,
        data_dir=dataset_root,
        output_dir=training_output_root,
        num_epochs=num_epochs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        run_id=f"{dataset_key}_mixed_full_roi",
    )
    result_payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    adapter_dir = Path(str(result_payload.get("adapter_dir") or getattr(result, "adapter_dir")))

    image_dir = Path(source_dataset_root) / dataset_key / "test"
    inference_reports: Dict[str, Any] = {}
    for report_name, ablation_name in {
        "full_image": "full_image_baseline",
        "primary_roi": "primary_roi_inference",
        "hybrid": "hybrid_roi_fallback",
    }.items():
        inference_reports[report_name] = run_ablation_folder(
            image_dir,
            ablation_name=ablation_name,
            output_dir=docs_output_dir / report_name,
            label_from_parent=True,
            adapter_crop=crop_name,
            adapter_part=part_name,
            config_env=config_env,
            device=device,
            adapter_root=adapter_dir,
            return_ood=return_ood,
            status_printer=status_printer,
        )

    summary = {
        "ablation_name": "mixed_full_roi_training",
        "dataset_key": dataset_key,
        "crop_name": crop_name,
        "part_name": part_name,
        "runtime_root": str(work_root),
        "mixed_dataset_manifest": manifest,
        "training_result": result_payload,
        "adapter_dir": str(adapter_dir),
        "inference_summaries": {
            name: dict(report.get("summary", {}))
            for name, report in inference_reports.items()
        },
    }
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    (docs_output_dir / "mixed_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def run_roi_trained_adapter_ablation(
    *,
    source_dataset_root: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path,
    dataset_key: str = "tomato__fruit",
    crop_name: str = "tomato",
    part_name: str = "fruit",
    config_env: Optional[str] = "colab",
    device: str = "cuda",
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    training_config_overrides: Optional[Dict[str, Any]] = None,
    return_ood: bool = True,
    status_printer: Optional[StatusPrinter] = print,
    training_workflow_factory: TrainingWorkflowFactory = TrainingWorkflow,
) -> Dict[str, Any]:
    """Run Notebook 13 end to end: ROI-only dataset, training, and ROI-only report."""
    docs_output_dir = Path(output_dir)
    work_root = Path(runtime_root)
    dataset_root = work_root / "roi_only_dataset"
    training_output_root = work_root / "training_outputs"
    manifest = build_roi_only_dataset(
        source_dataset_root,
        dataset_key=dataset_key,
        output_root=dataset_root,
        config_env=config_env,
        device=device,
        status_printer=status_printer,
    )

    config_overrides = dict(training_config_overrides or {})
    continual_overrides: Dict[str, Any] = {}
    if batch_size is not None:
        continual_overrides["batch_size"] = int(batch_size)
    if learning_rate is not None:
        continual_overrides["learning_rate"] = float(learning_rate)
    if continual_overrides:
        config_overrides = deep_merge(config_overrides, {"training": {"continual": continual_overrides}})
    config = _training_config_for_ablation(
        environment=config_env,
        dataset_root=dataset_root,
        dataset_key=dataset_key,
        overrides=config_overrides,
    )
    training = training_workflow_factory(config=config, environment=None, device=device)
    _emit_status(status_printer, "[ABLATION13] training ROI-only adapter")
    result = training.run(
        crop_name=crop_name,
        part_name=part_name,
        data_dir=dataset_root,
        output_dir=training_output_root,
        num_epochs=num_epochs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        run_id=f"{dataset_key}_roi_only",
    )
    result_payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    adapter_dir = Path(str(result_payload.get("adapter_dir") or getattr(result, "adapter_dir")))

    image_dir = Path(source_dataset_root) / dataset_key / "test"
    inference_report = run_ablation_folder(
        image_dir,
        ablation_name="primary_roi_inference",
        output_dir=docs_output_dir / "primary_roi",
        label_from_parent=True,
        adapter_crop=crop_name,
        adapter_part=part_name,
        config_env=config_env,
        device=device,
        adapter_root=adapter_dir,
        return_ood=return_ood,
        status_printer=status_printer,
    )

    summary = {
        "ablation_name": "roi_trained_adapter",
        "dataset_key": dataset_key,
        "crop_name": crop_name,
        "part_name": part_name,
        "runtime_root": str(work_root),
        "roi_only_dataset_manifest": manifest,
        "training_result": result_payload,
        "adapter_dir": str(adapter_dir),
        "inference_summaries": {
            "primary_roi": dict(inference_report.get("summary", {})),
        },
    }
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    (docs_output_dir / "roi_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def describe_training_ablation_plan(ablation_name: str) -> Dict[str, Any]:
    """Return the standardized second-phase training ablation contract."""
    if ablation_name not in {"roi_trained_adapter", "mixed_full_roi_training", "dual_view_trained_adapter"}:
        raise ValueError("Training ablation plan is only defined for notebooks 13, 14, and 17.")
    config = ABLATION_CONFIGS[ablation_name]
    if ablation_name == "dual_view_trained_adapter":
        return {
            "ablation_name": ablation_name,
            "notebook": config["notebook"],
            "output_dir": config["output_dir"],
            "phase": config["phase"],
            "input_policy": config["input_policy"],
            "status": "planned_after_dual_view_gate",
            "implementation_note": (
                "Run Notebook 15 and Notebook 16 first. Only start paired dual-view training if the dual-view "
                "inference report beats the full-image baseline on accuracy or macro-F1; otherwise the evidence "
                "says the problem is ROI quality or fusion policy, not adapter training."
            ),
            "readiness_note": (
                "A paired dual-view adapter would need fresh OOD calibration and production_readiness.json because "
                "its inference contract differs from single-view adapters."
            ),
        }
    return {
        "ablation_name": ablation_name,
        "notebook": config["notebook"],
        "output_dir": config["output_dir"],
        "phase": config["phase"],
        "input_policy": config["input_policy"],
        "status": "planned_second_phase",
        "implementation_note": (
            "Run inference-only notebooks 11 and 12 first. If ROI improves or reduces failures, "
            "add a research-only training dataset view that applies the same ROI policy to train, val, test, and OOD."
        ),
        "readiness_note": "Any ROI-trained adapter must recalibrate OOD and write a fresh production_readiness.json.",
    }


def _load_report_summary(report_path: Path) -> Optional[Dict[str, Any]]:
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    return dict(summary) if isinstance(summary, dict) else None


def evaluate_dual_view_training_gate(
    *,
    repo_root: str | Path,
    output_dir: str | Path | None = None,
    baseline_report: str | Path | None = None,
    dual_view_report: str | Path | None = None,
    min_accuracy_delta: float = 0.0,
    min_macro_f1_delta: float = 0.0,
) -> Dict[str, Any]:
    """Decide whether Notebook 17 should proceed to expensive paired dual-view training."""
    root = Path(repo_root)
    config = ABLATION_CONFIGS["dual_view_trained_adapter"]
    output_root = Path(output_dir) if output_dir is not None else root / config["output_dir"]
    baseline_path = (
        Path(baseline_report)
        if baseline_report is not None
        else root / ABLATION_CONFIGS["full_image_baseline"]["output_dir"] / "report.json"
    )
    dual_view_path = (
        Path(dual_view_report)
        if dual_view_report is not None
        else root / ABLATION_CONFIGS["dual_view_inference"]["output_dir"] / "report.json"
    )

    baseline_summary = _load_report_summary(baseline_path)
    dual_view_summary = _load_report_summary(dual_view_path)
    missing_reports = [
        name
        for name, summary in (
            ("full_image_baseline", baseline_summary),
            ("dual_view_inference", dual_view_summary),
        )
        if summary is None
    ]

    status = "blocked_until_dual_view_inference_results"
    gate_passed = False
    deltas: Dict[str, Optional[float]] = {"accuracy": None, "macro_f1": None}
    next_action = (
        "Run Notebooks 10 and 16, pull their committed reports, then rerun Notebook 17 to decide whether "
        "paired dual-view training is justified."
    )

    if not missing_reports:
        baseline_accuracy = baseline_summary.get("accuracy")
        dual_accuracy = dual_view_summary.get("accuracy")
        baseline_macro_f1 = baseline_summary.get("macro_f1")
        dual_macro_f1 = dual_view_summary.get("macro_f1")
        if baseline_accuracy is not None and dual_accuracy is not None:
            deltas["accuracy"] = float(dual_accuracy) - float(baseline_accuracy)
        if baseline_macro_f1 is not None and dual_macro_f1 is not None:
            deltas["macro_f1"] = float(dual_macro_f1) - float(baseline_macro_f1)

        accuracy_passed = deltas["accuracy"] is not None and deltas["accuracy"] >= float(min_accuracy_delta)
        macro_f1_passed = deltas["macro_f1"] is not None and deltas["macro_f1"] >= float(min_macro_f1_delta)
        gate_passed = bool(accuracy_passed or macro_f1_passed)
        status = "ready_for_paired_dual_view_training" if gate_passed else "skipped_by_gate"
        next_action = (
            "Implement a paired full-image plus ROI training loader/trainer and recalibrate OOD for Notebook 17."
            if gate_passed
            else "Do not train a paired dual-view adapter yet; inspect ROI quality and fusion errors before adding "
            "another training condition."
        )

    payload = {
        "ablation_name": "dual_view_trained_adapter",
        "config": config,
        "status": status,
        "gate_passed": gate_passed,
        "missing_reports": missing_reports,
        "thresholds": {
            "min_accuracy_delta": float(min_accuracy_delta),
            "min_macro_f1_delta": float(min_macro_f1_delta),
        },
        "reports": {
            "baseline_report": str(baseline_path),
            "dual_view_report": str(dual_view_path),
        },
        "baseline_summary": baseline_summary,
        "dual_view_summary": dual_view_summary,
        "deltas": deltas,
        "next_action": next_action,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dual_view_training_gate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run part-aware SAM box ROI ablation on an image folder.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--ablation-name", choices=sorted(ABLATION_CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-env", default="colab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--no-label-from-parent", action="store_true")
    args = parser.parse_args()

    if args.ablation_name == "roi_quality_audit":
        payload = run_roi_quality_audit_folder(
            args.image_dir,
            output_dir=args.output_dir,
            label_from_parent=not args.no_label_from_parent,
            config_env=args.config_env,
            device=args.device,
        )
        print(json.dumps(payload["summary"], indent=2))
        return 0

    if args.ablation_name == "dual_view_inference":
        payload = run_dual_view_inference_folder(
            args.image_dir,
            output_dir=args.output_dir,
            label_from_parent=not args.no_label_from_parent,
            config_env=args.config_env,
            device=args.device,
            adapter_root=args.adapter_root,
        )
        print(json.dumps(payload["summary"], indent=2))
        return 0

    if ABLATION_CONFIGS[args.ablation_name]["phase"] != "inference_only":
        print(json.dumps(describe_training_ablation_plan(args.ablation_name), indent=2))
        return 0

    payload = run_ablation_folder(
        args.image_dir,
        ablation_name=args.ablation_name,
        output_dir=args.output_dir,
        label_from_parent=not args.no_label_from_parent,
        config_env=args.config_env,
        device=args.device,
        adapter_root=args.adapter_root,
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
