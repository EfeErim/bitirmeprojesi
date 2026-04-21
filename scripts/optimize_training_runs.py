#!/usr/bin/env python3
"""Analyze a comparable training cohort.

Bayesian proposal generation and execution are disabled on this surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.index_training_runs import build_run_registry
from src.training.services.optimization import (
    DEFAULT_PARETO_OBJECTIVES,
    build_pareto_frontiers,
    resolve_single_cohort,
)

JsonDict = Dict[str, Any]


def _read_trials_jsonl(path: Path) -> List[JsonDict]:
    trials: List[JsonDict] = []
    if not path.exists():
        return trials
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).strip()
        if not text:
            continue
        trials.append(json.loads(text))
    return trials


def _summarize_selected_cohort(
    *,
    cohort_key: str,
    pareto_payload: JsonDict,
) -> JsonDict:
    pareto_cohort = next(
        cohort for cohort in pareto_payload.get("cohorts", []) if cohort.get("cohort_key") == cohort_key
    )
    return {
        "cohort_key": cohort_key,
        "comparability": pareto_cohort.get("comparability", {}),
        "pareto_frontier": {
            "frontier_count": pareto_cohort.get("frontier_count", 0),
            "frontier_run_ids": pareto_cohort.get("frontier_run_ids", []),
        },
        "bayesian_recommendations": {
            "enabled": False,
            "eligible_run_count": 0,
            "search_strategy": "disabled",
            "best_observed_run_id": "",
            "best_observed_score": None,
            "proposal_count": 0,
            "proposals": [],
        },
    }


def run_optimizer(args: argparse.Namespace) -> JsonDict:
    if bool(getattr(args, "execute", False)):
        raise ValueError("Bayesian optimization execution is disabled for this repository.")

    registry_result = build_run_registry(runs_root=args.runs_root, output_root=args.index_root)
    trials = _read_trials_jsonl(Path(registry_result["trials_jsonl"]))
    cohort_key, cohort_trials = resolve_single_cohort(
        trials,
        cohort_key=args.cohort_key,
        dataset_lineage_key=args.dataset_lineage_key,
        dataset_key=args.dataset_key,
        crop_name=args.crop_name,
        part_name=args.part_name,
        backbone_model_name=args.backbone_model_name,
        engine=args.engine,
    )
    pareto_payload = build_pareto_frontiers(
        cohort_trials,
        objective_names=args.objectives or DEFAULT_PARETO_OBJECTIVES,
    )
    summary = _summarize_selected_cohort(
        cohort_key=cohort_key,
        pareto_payload=pareto_payload,
    )
    result: JsonDict = {
        "runs_root": str(args.runs_root),
        "index_root": str(args.index_root),
        "registry_paths": {
            "trials_jsonl": str(registry_result["trials_jsonl"]),
            "latest_registry_json": str(registry_result["latest_registry_json"]),
            "pareto_inputs_json": str(registry_result["pareto_inputs_json"]),
            "pareto_frontiers_json": str(registry_result["pareto_frontiers_json"]),
        },
        "selected_cohort": summary,
        "bayesian_optimization_enabled": False,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--index-root", type=Path, default=Path("runs") / "_index")
    parser.add_argument("--cohort-key")
    parser.add_argument("--dataset-lineage-key")
    parser.add_argument("--dataset-key")
    parser.add_argument("--crop-name")
    parser.add_argument("--part-name")
    parser.add_argument("--backbone-model-name")
    parser.add_argument("--engine", default="continual_sd_lora")
    parser.add_argument("--objective", dest="objectives", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="Disabled; retained only to report a clear error.")
    parser.add_argument("--config-env", default="colab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-output-root", type=Path, default=Path("runs"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--validation-every-n-epochs", type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = run_optimizer(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
