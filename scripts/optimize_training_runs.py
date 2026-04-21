#!/usr/bin/env python3
"""Analyze a comparable training cohort and optionally launch Bayesian proposals."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.index_training_runs import build_run_registry
from src.core.config_manager import get_config
from src.shared.json_utils import deep_merge, read_json, write_json
from src.training.services.optimization import (
    DEFAULT_PARETO_OBJECTIVES,
    build_bayesian_recommendations,
    build_pareto_frontiers,
    resolve_single_cohort,
)

JsonDict = Dict[str, Any]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _load_search_space(path: Path | None) -> JsonDict | None:
    if path is None:
        return None
    payload = read_json(path, default={}, expect_type=(dict, list))
    if isinstance(payload, list):
        return {"parameters": payload}
    return dict(payload)


def _build_campaign_id(*, crop_name: str, part_name: str) -> str:
    return f"bo_{crop_name}_{part_name}_{_utc_stamp()}".lower()


def _build_optimizer_run_id(*, crop_name: str, part_name: str, proposal_rank: int) -> str:
    return f"{crop_name}_{part_name}_bo_{_utc_stamp()}_{proposal_rank:02d}".lower()


def _campaign_root(index_root: Path) -> Path:
    return index_root / "optimization_campaigns"


def _summarize_selected_cohort(
    *,
    cohort_key: str,
    pareto_payload: JsonDict,
    recommendation_payload: JsonDict,
) -> JsonDict:
    pareto_cohort = next(
        cohort for cohort in pareto_payload.get("cohorts", []) if cohort.get("cohort_key") == cohort_key
    )
    recommendation_cohort = next(
        cohort for cohort in recommendation_payload.get("cohorts", []) if cohort.get("cohort_key") == cohort_key
    )
    return {
        "cohort_key": cohort_key,
        "comparability": recommendation_cohort.get("comparability", {}),
        "pareto_frontier": {
            "frontier_count": pareto_cohort.get("frontier_count", 0),
            "frontier_run_ids": pareto_cohort.get("frontier_run_ids", []),
        },
        "bayesian_recommendations": {
            "eligible_run_count": recommendation_cohort.get("eligible_run_count", 0),
            "search_strategy": recommendation_cohort.get("search_strategy"),
            "best_observed_run_id": recommendation_cohort.get("best_observed_run_id"),
            "best_observed_score": recommendation_cohort.get("best_observed_score"),
            "proposal_count": len(recommendation_cohort.get("proposals", [])),
            "proposals": recommendation_cohort.get("proposals", []),
        },
    }


def run_optimizer(args: argparse.Namespace) -> JsonDict:
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
    search_space_payload = _load_search_space(args.search_space)
    pareto_payload = build_pareto_frontiers(
        cohort_trials,
        objective_names=args.objectives or DEFAULT_PARETO_OBJECTIVES,
    )
    recommendation_payload = build_bayesian_recommendations(
        cohort_trials,
        objective_names=args.objectives or DEFAULT_PARETO_OBJECTIVES,
        proposal_count=args.proposal_count,
        candidate_pool_size=args.candidate_pool_size,
        search_space_payload=search_space_payload,
        random_seed=args.random_seed,
    )
    summary = _summarize_selected_cohort(
        cohort_key=cohort_key,
        pareto_payload=pareto_payload,
        recommendation_payload=recommendation_payload,
    )
    result: JsonDict = {
        "runs_root": str(args.runs_root),
        "index_root": str(args.index_root),
        "registry_paths": {
            "trials_jsonl": str(registry_result["trials_jsonl"]),
            "latest_registry_json": str(registry_result["latest_registry_json"]),
            "pareto_inputs_json": str(registry_result["pareto_inputs_json"]),
            "pareto_frontiers_json": str(registry_result["pareto_frontiers_json"]),
            "bayesian_recommendations_json": str(registry_result["bayesian_recommendations_json"]),
        },
        "selected_cohort": summary,
    }
    if not args.execute:
        return result

    from src.workflows.training import TrainingWorkflow

    comparability = dict(summary.get("comparability", {}))
    crop_name = str(comparability.get("crop_name", "") or "")
    part_name = str(comparability.get("part_name", "") or "unspecified")
    if not crop_name:
        raise ValueError("Selected cohort does not expose a crop_name for execution.")
    proposals = list(summary.get("bayesian_recommendations", {}).get("proposals", []))
    if not proposals:
        raise ValueError("No Bayesian proposals are available to execute for the selected cohort.")
    eligible_run_count = int(summary.get("bayesian_recommendations", {}).get("eligible_run_count", 0) or 0)
    search_strategy = str(summary.get("bayesian_recommendations", {}).get("search_strategy", "") or "")
    if eligible_run_count <= 0 and not bool(getattr(args, "allow_bootstrap_execute", False)):
        raise ValueError(
            "Execution is blocked because the selected cohort has zero eligible Bayesian evidence "
            "(all observed runs were excluded). Re-run without --execute, improve readiness evidence, "
            "or pass --allow-bootstrap-execute to explicitly execute random-bootstrap proposals."
        )

    base_config = get_config(environment=args.config_env)
    campaign_id = _build_campaign_id(crop_name=crop_name, part_name=part_name)
    campaign_payload: JsonDict = {
        "campaign_id": campaign_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_key": cohort_key,
        "comparability": comparability,
        "search_strategy": summary.get("bayesian_recommendations", {}).get("search_strategy"),
        "eligible_run_count": eligible_run_count,
        "bootstrap_execution_allowed": bool(getattr(args, "allow_bootstrap_execute", False)),
        "bootstrap_execution_used": bool(search_strategy == "random_bootstrap"),
        "objective_names": args.objectives or list(DEFAULT_PARETO_OBJECTIVES),
        "proposal_count": len(proposals),
        "proposals": proposals,
        "executed_runs": [],
    }
    campaign_path = _campaign_root(args.index_root) / f"{campaign_id}.json"
    write_json(campaign_path, campaign_payload, ensure_ascii=False)

    executed_runs: List[JsonDict] = []
    for proposal in proposals:
        config_override = dict(proposal.get("config_override", {}))
        merged_config = deep_merge(base_config, config_override)
        run_id = _build_optimizer_run_id(
            crop_name=crop_name,
            part_name=part_name,
            proposal_rank=int(proposal.get("rank", len(executed_runs) + 1)),
        )
        output_dir = args.run_output_root / run_id
        workflow = TrainingWorkflow(
            config=merged_config,
            environment=args.config_env,
            device=args.device,
        )
        workflow_result = workflow.run(
            crop_name=crop_name,
            data_dir=args.data_dir,
            output_dir=output_dir,
            num_workers=args.num_workers,
            validation_every_n_epochs=args.validation_every_n_epochs,
            run_id=run_id,
        )
        executed_runs.append(
            {
                "run_id": run_id,
                "proposal_rank": proposal.get("rank"),
                "output_dir": str(output_dir),
                "parameters": proposal.get("parameters", {}),
                "result": workflow_result.to_dict(),
            }
        )

    campaign_payload["executed_runs"] = executed_runs
    write_json(campaign_path, campaign_payload, ensure_ascii=False)
    final_registry = build_run_registry(runs_root=args.runs_root, output_root=args.index_root)
    result["campaign_json"] = str(campaign_path)
    result["executed_runs"] = executed_runs
    result["registry_paths"].update(
        {
            "trials_jsonl": str(final_registry["trials_jsonl"]),
            "latest_registry_json": str(final_registry["latest_registry_json"]),
            "pareto_inputs_json": str(final_registry["pareto_inputs_json"]),
            "pareto_frontiers_json": str(final_registry["pareto_frontiers_json"]),
            "bayesian_recommendations_json": str(final_registry["bayesian_recommendations_json"]),
        }
    )
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
    parser.add_argument("--proposal-count", type=int, default=3)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--search-space", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-bootstrap-execute",
        action="store_true",
        help=(
            "Allow --execute when a cohort has zero eligible Bayesian evidence and recommendations "
            "fall back to random bootstrap."
        ),
    )
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
    if args.execute and args.data_dir is None:
        parser.error("--data-dir is required when --execute is used.")
    result = run_optimizer(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
