#!/usr/bin/env python3
"""Validate Phase 5 benchmark results against guardrail thresholds."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def _index_results(results_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = results_payload.get('results', [])
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        scenario = str(row.get('name', '')).strip()
        if scenario:
            indexed[scenario] = row
    return indexed


def _check_scenario(
    scenario: str,
    result: Dict[str, Any],
    limits: Dict[str, float],
) -> List[str]:
    failures: List[str] = []

    avg_wall = float(result.get('avg_wall_ms', 0.0))
    p95_wall = float(result.get('p95_wall_ms', 0.0))
    avg_processing = float(result.get('avg_processing_ms', 0.0))
    stage_timings = result.get('avg_stage_timings_ms', {}) or {}
    roi_classification = float(stage_timings.get('roi_classification', 0.0))

    if 'avg_wall_ms_max' in limits and avg_wall > float(limits['avg_wall_ms_max']):
        failures.append(
            f"{scenario}: avg_wall_ms={avg_wall:.4f} exceeds limit {float(limits['avg_wall_ms_max']):.4f}"
        )
    if 'p95_wall_ms_max' in limits and p95_wall > float(limits['p95_wall_ms_max']):
        failures.append(
            f"{scenario}: p95_wall_ms={p95_wall:.4f} exceeds limit {float(limits['p95_wall_ms_max']):.4f}"
        )
    if 'avg_processing_ms_max' in limits and avg_processing > float(limits['avg_processing_ms_max']):
        failures.append(
            f"{scenario}: avg_processing_ms={avg_processing:.4f} exceeds limit {float(limits['avg_processing_ms_max']):.4f}"
        )
    if 'roi_classification_ms_max' in limits and roi_classification > float(limits['roi_classification_ms_max']):
        failures.append(
            f"{scenario}: roi_classification={roi_classification:.4f} exceeds limit {float(limits['roi_classification_ms_max']):.4f}"
        )

    return failures


def evaluate_guardrails(results_path: Path, guardrails_path: Path) -> Tuple[bool, List[str]]:
    benchmark = _load_json(results_path)
    guardrails = _load_json(guardrails_path)

    results_by_scenario = _index_results(benchmark)
    limits = guardrails.get('limits', {})
    if not isinstance(limits, dict):
        return False, ["Guardrail limits are missing or invalid"]

    errors: List[str] = []
    for scenario, scenario_limits in limits.items():
        if scenario not in results_by_scenario:
            errors.append(f"Missing benchmark scenario in results: {scenario}")
            continue
        if not isinstance(scenario_limits, dict):
            errors.append(f"Invalid guardrail limits for scenario: {scenario}")
            continue

        scenario_errors = _check_scenario(scenario, results_by_scenario[scenario], scenario_limits)
        errors.extend(scenario_errors)

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 5 benchmark metrics against guardrail thresholds")
    parser.add_argument(
        '--results',
        default='logs/phase5_router_benchmark.json',
        help='Path to benchmark results JSON',
    )
    parser.add_argument(
        '--guardrails',
        default='config/perf_guardrails_phase5.json',
        help='Path to guardrail threshold JSON',
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    guardrails_path = Path(args.guardrails)

    try:
        ok, errors = evaluate_guardrails(results_path, guardrails_path)
    except Exception as exc:
        print(f"❌ Guardrail check failed with exception: {exc}")
        return 2

    if not ok:
        print("❌ Phase 5 performance regression detected:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("✅ Phase 5 performance guardrails passed")
    print(f"Results: {results_path}")
    print(f"Guardrails: {guardrails_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
