#!/usr/bin/env python3
"""Run a multi-parameter calibration sweep for the router handoff surface."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_router_surface import (
    discover_eval_samples,
    sample_from_analysis,
    summarize_predictions,
)
from src.core.config_manager import get_config
from src.router.policy_taxonomy_utils import resolve_requested_profile
from src.router.router_pipeline import RouterPipeline
from src.router.vlm_stages import build_pipeline_surface_config

JsonDict = Dict[str, Any]

SUPPORTED_PARAMETERS: Dict[str, str] = {
    "router_min_confidence": "inference.router_min_confidence",
    "router_min_margin": "inference.router_min_margin",
    "vlm_confidence_threshold": "router.vlm.confidence_threshold",
    "part_open_set_min_confidence": "router.vlm.policy_graph.part_resolution.part_open_set_min_confidence",
    "part_open_set_margin": "router.vlm.policy_graph.part_resolution.part_open_set_margin",
    "global_crop_context_weight": "router.vlm.policy_graph.crop_evidence.global_crop_context_weight",
    "sam3_mask_threshold": "router.vlm.policy_graph.roi_filter.sam3_mask_threshold",
    "sam3_prompt_limit": "router.vlm.sam3_prompt_limit",
    "crop_num_prompts": "router.vlm.policy_graph.crop_evidence.crop_num_prompts",
    "part_num_prompts": "router.vlm.policy_graph.part_evidence.part_num_prompts",
    "max_rois_for_classification": "router.vlm.policy_graph.roi_filter.max_rois_for_classification",
}

INTEGER_PARAMETERS = {
    "router.vlm.sam3_prompt_limit",
    "router.vlm.policy_graph.crop_evidence.crop_num_prompts",
    "router.vlm.policy_graph.part_evidence.part_num_prompts",
    "router.vlm.policy_graph.roi_filter.max_rois_for_classification",
}

PRESET_SWEEPS: Dict[str, Dict[str, List[Any]]] = {
    "handoff": {
        "inference.router_min_confidence": [0.55, 0.65, 0.75],
        "inference.router_min_margin": [0.00, 0.10, 0.15],
    },
    "quick": {
        "inference.router_min_confidence": [0.55, 0.65, 0.75],
        "inference.router_min_margin": [0.00, 0.10, 0.15],
        "router.vlm.confidence_threshold": [0.20, 0.25, 0.35],
        "router.vlm.policy_graph.crop_evidence.global_crop_context_weight": [0.45, 0.65, 0.80],
    },
    "docs": {
        "inference.router_min_confidence": [0.55, 0.65, 0.75],
        "inference.router_min_margin": [0.00, 0.10, 0.15],
        "router.vlm.confidence_threshold": [0.20, 0.25, 0.35],
        "router.vlm.policy_graph.part_resolution.part_open_set_min_confidence": [0.30, 0.40, 0.50],
        "router.vlm.policy_graph.part_resolution.part_open_set_margin": [0.05, 0.10, 0.15],
        "router.vlm.policy_graph.crop_evidence.global_crop_context_weight": [0.45, 0.65, 0.80],
        "router.vlm.policy_graph.roi_filter.sam3_mask_threshold": [0.50, 0.60, 0.70],
        "router.vlm.sam3_prompt_limit": [4, 6],
        "router.vlm.policy_graph.crop_evidence.crop_num_prompts": [2, 4],
        "router.vlm.policy_graph.part_evidence.part_num_prompts": [2, 4],
        "router.vlm.policy_graph.roi_filter.max_rois_for_classification": [0, 16],
    },
}


def _canonical_parameter_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if name in SUPPORTED_PARAMETERS:
        return SUPPORTED_PARAMETERS[name]
    if name in SUPPORTED_PARAMETERS.values():
        return name
    supported = ", ".join(sorted(SUPPORTED_PARAMETERS))
    raise ValueError(f"Unsupported sweep parameter '{name}'. Supported aliases: {supported}")


def _coerce_parameter_value(parameter: str, raw_value: Any) -> Any:
    if parameter in INTEGER_PARAMETERS:
        return int(raw_value)
    return float(raw_value)


def _get_nested_value(payload: JsonDict, dotted_path: str, default: Any = None) -> Any:
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _set_nested_value(payload: JsonDict, dotted_path: str, value: Any) -> JsonDict:
    cloned = copy.deepcopy(payload)
    cursor: Any = cloned
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = copy.deepcopy(value)
    return cloned


def parse_sweep_spec(raw_spec: str) -> tuple[str, List[Any]]:
    """Parse PARAM=v1,v2 syntax used by the CLI."""
    if "=" not in str(raw_spec):
        raise ValueError(f"Sweep spec must use PARAM=v1,v2 syntax, got {raw_spec!r}")
    raw_name, raw_values = str(raw_spec).split("=", 1)
    parameter = _canonical_parameter_name(raw_name)
    values = [
        _coerce_parameter_value(parameter, item.strip())
        for item in raw_values.split(",")
        if item.strip()
    ]
    if not values:
        raise ValueError(f"Sweep spec for {parameter} did not include any values.")
    return parameter, values


def _dedupe_values(values: Iterable[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def resolve_sweep_grid(
    base_config: JsonDict,
    *,
    preset: str = "quick",
    sweep_specs: Sequence[str] | None = None,
    include_current: bool = True,
) -> Dict[str, List[Any]]:
    if preset not in PRESET_SWEEPS and preset != "none":
        raise ValueError(f"Unknown preset '{preset}'. Choose one of: none, {', '.join(sorted(PRESET_SWEEPS))}")

    grid: Dict[str, List[Any]] = {}
    if preset != "none":
        grid.update(copy.deepcopy(PRESET_SWEEPS[preset]))

    for raw_spec in list(sweep_specs or []):
        parameter, values = parse_sweep_spec(raw_spec)
        grid[parameter] = values

    if include_current:
        for parameter, values in list(grid.items()):
            current = _get_nested_value(base_config, parameter)
            if current is not None:
                current = _coerce_parameter_value(parameter, current)
                grid[parameter] = _dedupe_values([current, *values])

    return {parameter: _dedupe_values(values) for parameter, values in grid.items()}


def _variant_count(grid: Dict[str, List[Any]]) -> int:
    total = 1
    for values in grid.values():
        total *= max(1, len(values))
    return total


def iter_sweep_overrides(grid: Dict[str, List[Any]]) -> Iterable[JsonDict]:
    parameters = list(grid.keys())
    value_lists = [grid[parameter] for parameter in parameters]
    for combo in itertools.product(*value_lists):
        yield {parameter: value for parameter, value in zip(parameters, combo)}


def apply_overrides(base_config: JsonDict, overrides: JsonDict) -> JsonDict:
    config = copy.deepcopy(base_config)
    for parameter, value in overrides.items():
        config = _set_nested_value(config, parameter, value)
        if parameter.startswith("router.vlm."):
            profile = _get_nested_value(config, "router.vlm.profile")
            if isinstance(profile, str) and profile.strip():
                relative_path = parameter.removeprefix("router.vlm.")
                profile_root = f"router.vlm.profiles.{profile.strip()}"
                if isinstance(_get_nested_value(config, profile_root), dict):
                    config = _set_nested_value(config, f"{profile_root}.{relative_path}", value)
    return config


def _apply_config_to_loaded_router(router: RouterPipeline, config: JsonDict) -> None:
    surface = build_pipeline_surface_config(config)
    router.config = copy.deepcopy(config)
    router.vlm_config = surface.vlm_config
    router._base_vlm_config = surface.base_vlm_config
    router.set_runtime_profile(resolve_requested_profile(router.vlm_config), suppress_warning=True)


def _variant_id(overrides: JsonDict) -> str:
    body = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


def evaluate_variant(
    router: RouterPipeline,
    dataset: Sequence[JsonDict],
    *,
    config: JsonDict,
    overrides: JsonDict,
) -> JsonDict:
    _apply_config_to_loaded_router(router, config)

    samples: List[JsonDict] = []
    started_variant = time.perf_counter()
    for item in dataset:
        image = Image.open(item["image_path"]).convert("RGB")
        started = time.perf_counter()
        analysis = router.analyze_image_result(image)
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples.append(
            sample_from_analysis(
                item=item,
                analysis=analysis,
                latency_ms=latency_ms,
                config=config,
            )
        )

    metrics = summarize_predictions(samples)
    metrics["variant_wall_time_ms"] = round((time.perf_counter() - started_variant) * 1000.0, 4)
    return {
        "variant_id": "baseline" if not overrides else _variant_id(overrides),
        "overrides": copy.deepcopy(overrides),
        "metrics": metrics,
        "samples": samples,
    }


def _eligibility_reasons(
    metrics: JsonDict,
    baseline_metrics: JsonDict,
    *,
    target_negative_false_accept_rate: float,
    max_crop_accuracy_drop: float,
    max_part_precision_drop: float,
    max_part_recall_drop: float,
    max_wrong_part_rejection_drop: float,
    max_p95_latency_regression: float,
) -> List[str]:
    reasons: List[str] = []
    if float(metrics.get("negative_false_accept_rate", 0.0)) > float(target_negative_false_accept_rate):
        reasons.append("negative_false_accept_rate_above_target")
    min_crop_accuracy = max(0.0, float(baseline_metrics.get("crop_accuracy", 0.0)) - float(max_crop_accuracy_drop))
    if float(metrics.get("crop_accuracy", 0.0)) < min_crop_accuracy:
        reasons.append("crop_accuracy_drop")
    min_part_precision = max(
        0.0,
        float(baseline_metrics.get("part_non_unknown_precision", 0.0)) - float(max_part_precision_drop),
    )
    if float(metrics.get("part_non_unknown_precision", 0.0)) < min_part_precision:
        reasons.append("part_precision_drop")
    min_part_recall = max(0.0, float(baseline_metrics.get("part_recall", 0.0)) - float(max_part_recall_drop))
    if float(metrics.get("part_recall", 0.0)) < min_part_recall:
        reasons.append("part_recall_drop")
    min_wrong_part_rejection = max(
        0.0,
        float(baseline_metrics.get("wrong_part_rejection_rate", 0.0)) - float(max_wrong_part_rejection_drop),
    )
    if float(metrics.get("wrong_part_rejection_rate", 0.0)) < min_wrong_part_rejection:
        reasons.append("wrong_part_rejection_drop")
    baseline_p95_latency = float(baseline_metrics.get("p95_latency_ms", 0.0) or 0.0)
    p95_latency = float(metrics.get("p95_latency_ms", 0.0) or 0.0)
    if baseline_p95_latency > 0.0 and p95_latency > baseline_p95_latency * (1.0 + float(max_p95_latency_regression)):
        reasons.append("p95_latency_regression")
    return reasons


def annotate_and_rank_variants(
    variants: Sequence[JsonDict],
    *,
    baseline: JsonDict,
    target_negative_false_accept_rate: float = 0.05,
    max_crop_accuracy_drop: float = 0.02,
    max_part_precision_drop: float = 0.02,
    max_part_recall_drop: float = 0.02,
    max_wrong_part_rejection_drop: float = 0.02,
    max_p95_latency_regression: float = 0.25,
) -> List[JsonDict]:
    baseline_metrics = baseline.get("metrics", {})
    annotated: List[JsonDict] = []
    for variant in variants:
        row = copy.deepcopy(variant)
        metrics = row.get("metrics", {})
        reasons = _eligibility_reasons(
            metrics,
            baseline_metrics,
            target_negative_false_accept_rate=target_negative_false_accept_rate,
            max_crop_accuracy_drop=max_crop_accuracy_drop,
            max_part_precision_drop=max_part_precision_drop,
            max_part_recall_drop=max_part_recall_drop,
            max_wrong_part_rejection_drop=max_wrong_part_rejection_drop,
            max_p95_latency_regression=max_p95_latency_regression,
        )
        row["eligible"] = not reasons
        row["eligibility_reasons"] = reasons
        annotated.append(row)

    def _rank_key(row: JsonDict) -> tuple[Any, ...]:
        metrics = row.get("metrics", {})
        return (
            not bool(row.get("eligible", False)),
            float(metrics.get("negative_false_accept_rate", 0.0)),
            int(metrics.get("unsupported_part_emissions", 0)),
            -float(metrics.get("wrong_part_rejection_rate", 0.0)),
            -float(metrics.get("crop_accuracy", 0.0)),
            -float(metrics.get("part_non_unknown_precision", 0.0)),
            -float(metrics.get("part_recall", 0.0)),
            float(metrics.get("abstention_rate", 0.0)),
            float(metrics.get("p95_latency_ms", 0.0)),
            float(metrics.get("mean_latency_ms", 0.0)),
            row.get("variant_id", ""),
        )

    return sorted(annotated, key=_rank_key)


def _strip_samples(variant: JsonDict) -> JsonDict:
    slim = copy.deepcopy(variant)
    slim.pop("samples", None)
    return slim


def calibrate_router_surface(
    root: Path,
    *,
    config_env: str | None = "colab",
    device: str = "cuda",
    preset: str = "quick",
    sweep_specs: Sequence[str] | None = None,
    include_current: bool = True,
    max_variants: int = 128,
    target_negative_false_accept_rate: float = 0.05,
    max_crop_accuracy_drop: float = 0.02,
    max_part_precision_drop: float = 0.02,
    max_part_recall_drop: float = 0.02,
    max_wrong_part_rejection_drop: float = 0.02,
    max_p95_latency_regression: float = 0.25,
    include_samples: bool = False,
) -> JsonDict:
    dataset = discover_eval_samples(root)
    if not dataset:
        raise RuntimeError(f"No router eval images found under {root}")

    base_config = get_config(environment=config_env)
    grid = resolve_sweep_grid(
        base_config,
        preset=preset,
        sweep_specs=sweep_specs,
        include_current=include_current,
    )
    total_variants = _variant_count(grid)
    if total_variants > int(max_variants):
        raise RuntimeError(
            f"Sweep expands to {total_variants} variants, above --max-variants={max_variants}. "
            "Use a smaller preset, fewer --sweep values, or raise --max-variants intentionally."
        )

    router = RouterPipeline(config=base_config, device=device)
    router.load_models()
    if not router.is_ready():
        raise RuntimeError(
            "Router models failed to become ready for inference. "
            "Check router.vlm.enabled, model availability, and router dependency installation."
        )

    baseline = evaluate_variant(router, dataset, config=base_config, overrides={})
    variants: List[JsonDict] = []
    seen_variants = {json.dumps({}, sort_keys=True)}
    seen_configs = {json.dumps(base_config, sort_keys=True, default=str)}
    for overrides in iter_sweep_overrides(grid):
        key = json.dumps(overrides, sort_keys=True)
        if key in seen_variants:
            continue
        seen_variants.add(key)
        variant_config = apply_overrides(base_config, overrides)
        config_key = json.dumps(variant_config, sort_keys=True, default=str)
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        variants.append(evaluate_variant(router, dataset, config=variant_config, overrides=overrides))

    ranked = annotate_and_rank_variants(
        [baseline, *variants],
        baseline=baseline,
        target_negative_false_accept_rate=target_negative_false_accept_rate,
        max_crop_accuracy_drop=max_crop_accuracy_drop,
        max_part_precision_drop=max_part_precision_drop,
        max_part_recall_drop=max_part_recall_drop,
        max_wrong_part_rejection_drop=max_wrong_part_rejection_drop,
        max_p95_latency_regression=max_p95_latency_regression,
    )
    recommended = ranked[0] if ranked else baseline
    variant_times = [float(row.get("metrics", {}).get("variant_wall_time_ms", 0.0)) for row in variants]

    result = {
        "dataset_root": str(root),
        "sample_count": len(dataset),
        "config_env": config_env,
        "device": device,
        "preset": preset,
        "sweep_grid": grid,
        "variant_count": len(variants),
        "baseline": baseline if include_samples else _strip_samples(baseline),
        "recommended": recommended if include_samples else _strip_samples(recommended),
        "variants": ranked if include_samples else [_strip_samples(row) for row in ranked],
        "runtime_summary": {
            "mean_variant_wall_time_ms": 0.0 if not variant_times else round(statistics.fmean(variant_times), 4),
            "max_variant_wall_time_ms": 0.0 if not variant_times else round(max(variant_times), 4),
        },
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Eval root: data/router_eval/{id,negatives,ambiguous,wrong_part}/...",
    )
    parser.add_argument("--config-env", default="colab", help="Config environment override (default: colab)")
    parser.add_argument("--device", default="cuda", help="Torch device preference")
    parser.add_argument(
        "--preset",
        choices=["none", *sorted(PRESET_SWEEPS.keys())],
        default="quick",
        help="Built-in sweep grid. Use 'none' with explicit --sweep entries.",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help=(
            "Override or add one grid dimension as PARAM=v1,v2. "
            "Aliases include router_min_confidence, router_min_margin, "
            "vlm_confidence_threshold, global_crop_context_weight, sam3_mask_threshold."
        ),
    )
    parser.add_argument(
        "--exclude-current",
        action="store_true",
        help="Do not automatically include the current config value for every swept parameter.",
    )
    parser.add_argument("--max-variants", type=int, default=128, help="Refuse sweeps larger than this count.")
    parser.add_argument(
        "--target-negative-far",
        type=float,
        default=0.05,
        help="Maximum negative false-accept rate for an eligible recommendation.",
    )
    parser.add_argument("--max-crop-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-part-precision-drop", type=float, default=0.02)
    parser.add_argument("--max-part-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-wrong-part-rejection-drop", type=float, default=0.02)
    parser.add_argument(
        "--max-p95-latency-regression",
        type=float,
        default=0.25,
        help="Maximum allowed p95 latency increase vs baseline as a fraction (default: 0.25).",
    )
    parser.add_argument("--include-samples", action="store_true", help="Include per-sample rows for every variant.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = calibrate_router_surface(
        args.root,
        config_env=args.config_env,
        device=args.device,
        preset=args.preset,
        sweep_specs=args.sweep,
        include_current=not args.exclude_current,
        max_variants=args.max_variants,
        target_negative_false_accept_rate=args.target_negative_far,
        max_crop_accuracy_drop=args.max_crop_accuracy_drop,
        max_part_precision_drop=args.max_part_precision_drop,
        max_part_recall_drop=args.max_part_recall_drop,
        max_wrong_part_rejection_drop=args.max_wrong_part_rejection_drop,
        max_p95_latency_regression=args.max_p95_latency_regression,
        include_samples=args.include_samples,
    )
    body = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
