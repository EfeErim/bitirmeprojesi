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
    if ablation_name not in {"roi_trained_adapter", "mixed_full_roi_training"}:
        raise ValueError("Training ablation plan is only defined for notebooks 13 and 14.")
    config = ABLATION_CONFIGS[ablation_name]
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
