#!/usr/bin/env python3
"""Build a local training-run registry for Pareto preparation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.json_utils import read_json, write_json
from src.training.services.optimization import build_pareto_frontiers
from src.training.services.traceability import (
    build_experiment_manifest,
    build_optimization_record,
    load_authoritative_artifacts_from_root,
)

JsonDict = Dict[str, Any]

REGISTRY_SCHEMA = "v1_training_run_registry"
TRIAL_SCHEMA = "v1_training_registry_trial"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def discover_artifact_roots(runs_root: str | Path) -> List[Path]:
    root = Path(runs_root)
    candidates: set[Path] = set()
    if not root.exists():
        return []
    for summary_path in root.rglob("training/summary.json"):
        if "_index" in summary_path.parts:
            continue
        candidates.add(summary_path.parent.parent)
    for optimization_path in root.rglob("training/optimization_record.json"):
        if "_index" in optimization_path.parts:
            continue
        candidates.add(optimization_path.parent.parent)
    for manifest_path in root.rglob("training/experiment_manifest.json"):
        if "_index" in manifest_path.parts:
            continue
        candidates.add(manifest_path.parent.parent)
    for overview_path in root.rglob("guided/01_run_overview.json"):
        if "_index" in overview_path.parts:
            continue
        candidates.add(overview_path.parent.parent)
    return sorted(candidates, key=lambda item: str(item).lower())


def _find_run_dir(artifact_root: Path, runs_root: Path) -> Path:
    current = artifact_root.resolve()
    runs_root = runs_root.resolve()
    while current != current.parent:
        if current.parent == runs_root:
            return current
        current = current.parent
    return artifact_root


def _artifact_root_priority(artifact_root: Path) -> int:
    normalized = artifact_root.as_posix().lower()
    if "/outputs/colab_notebook_training/artifacts" in normalized:
        return 0
    if normalized.endswith("/checkpoint_state/artifacts"):
        return 1
    if normalized.endswith("/telemetry/artifacts"):
        return 2
    if normalized.endswith("/training_metrics"):
        return 3
    return 4


def _read_json_if_exists(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return read_json(path, default=default)


def _collect_summary_payload(artifact_root: Path, run_dir: Path) -> JsonDict:
    summary = _read_json_if_exists(artifact_root / "training" / "summary.json", default={})
    if isinstance(summary, dict) and summary:
        return dict(summary)

    overview = _read_json_if_exists(artifact_root / "guided" / "01_run_overview.json", default={})
    telemetry_summary = _read_json_if_exists(run_dir / "telemetry" / "summary.json", default={})
    checkpoint_summary = _read_json_if_exists(run_dir / "checkpoint_state" / "summary.json", default={})
    merged: JsonDict = {}
    for payload in (overview, telemetry_summary.get("run_overview", {}) if isinstance(telemetry_summary, dict) else {}, checkpoint_summary):
        if isinstance(payload, dict):
            merged.update({key: value for key, value in payload.items() if value not in (None, "", [], {})})
    return merged


def _load_traceability_record(artifact_root: Path) -> tuple[JsonDict, JsonDict, str]:
    manifest_path = artifact_root / "training" / "experiment_manifest.json"
    optimization_path = artifact_root / "training" / "optimization_record.json"
    if manifest_path.exists() and optimization_path.exists():
        manifest = read_json(manifest_path, default={}, expect_type=dict)
        optimization = read_json(optimization_path, default={}, expect_type=dict)
        return dict(manifest), dict(optimization), "canonical"
    return {}, {}, "backfilled"


def _build_backfilled_records(artifact_root: Path, run_dir: Path) -> tuple[JsonDict, JsonDict]:
    summary = _collect_summary_payload(artifact_root, run_dir)
    run_context = _read_json_if_exists(artifact_root / "training" / "run_context.json", default={})
    production_readiness = _read_json_if_exists(artifact_root / "production_readiness.json", default={})
    classification_split = ""
    if isinstance(production_readiness, dict):
        classification_evidence = production_readiness.get("classification_evidence", {})
        if isinstance(classification_evidence, dict):
            classification_split = str(classification_evidence.get("split_name", "") or "")
    authoritative_artifacts = load_authoritative_artifacts_from_root(
        artifact_root,
        classification_split=classification_split,
    )
    manifest = build_experiment_manifest(
        summary_payload=summary,
        run_context_payload=run_context,
        artifact_root=artifact_root,
        explicit_surface=(str(summary.get("surface", "") or "").strip() or None),
        created_at=str(summary.get("created_at", "") or run_context.get("created_at", "") or ""),
        record_quality="backfilled",
    )
    optimization = build_optimization_record(
        summary_payload=summary,
        run_context_payload=run_context,
        production_readiness_payload=production_readiness,
        authoritative_artifacts=authoritative_artifacts,
        artifact_root=artifact_root,
        explicit_surface=(str(summary.get("surface", "") or "").strip() or None),
        created_at=str(summary.get("created_at", "") or run_context.get("created_at", "") or ""),
        record_quality="backfilled",
    )
    return manifest, optimization


def _build_trial_record(
    *,
    artifact_root: Path,
    run_dir: Path,
    experiment_manifest: JsonDict,
    optimization_record: JsonDict,
) -> JsonDict:
    record = dict(optimization_record)
    record["schema_version"] = TRIAL_SCHEMA
    record["registry_source"] = {
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "artifact_root_priority": _artifact_root_priority(artifact_root),
    }
    record["experiment_manifest"] = experiment_manifest
    return record


def _trial_group_key(trial: JsonDict, run_dir: Path) -> Tuple[str, str]:
    run_id = str(trial.get("run_id", "") or "")
    return (str(run_dir.name), run_id or str(trial.get("registry_source", {}).get("artifact_root", "")))


def _choose_preferred_trial(left: JsonDict, right: JsonDict) -> JsonDict:
    left_quality = str(left.get("record_quality", "") or "backfilled")
    right_quality = str(right.get("record_quality", "") or "backfilled")
    left_priority = int(left.get("registry_source", {}).get("artifact_root_priority", 99))
    right_priority = int(right.get("registry_source", {}).get("artifact_root_priority", 99))
    if left_quality != right_quality:
        return left if left_quality == "canonical" else right
    if left_priority != right_priority:
        return left if left_priority < right_priority else right
    return left


def collect_trial_records(runs_root: str | Path) -> List[JsonDict]:
    root = Path(runs_root)
    selected: Dict[Tuple[str, str], JsonDict] = {}
    for artifact_root in discover_artifact_roots(root):
        run_dir = _find_run_dir(artifact_root, root)
        manifest, optimization, quality = _load_traceability_record(artifact_root)
        if not manifest or not optimization:
            manifest, optimization = _build_backfilled_records(artifact_root, run_dir)
            quality = "backfilled"
        trial = _build_trial_record(
            artifact_root=artifact_root,
            run_dir=run_dir,
            experiment_manifest=manifest,
            optimization_record=optimization,
        )
        trial["record_quality"] = quality
        key = _trial_group_key(trial, run_dir)
        if key in selected:
            selected[key] = _choose_preferred_trial(selected[key], trial)
        else:
            selected[key] = trial
    return sorted(
        selected.values(),
        key=lambda item: (
            str(item.get("created_at", "") or ""),
            str(item.get("run_id", "") or ""),
        ),
    )


def build_pareto_inputs(trials: Iterable[JsonDict]) -> JsonDict:
    grouped: Dict[str, List[JsonDict]] = defaultdict(list)
    incomplete: List[JsonDict] = []
    for trial in trials:
        cohort_key = str(trial.get("comparability", {}).get("cohort_key", "") or "")
        if cohort_key:
            grouped[cohort_key].append(trial)
        else:
            incomplete.append(trial)
    cohorts = []
    for cohort_key, cohort_trials in sorted(grouped.items(), key=lambda item: item[0]):
        first = cohort_trials[0]
        cohorts.append(
            {
                "cohort_key": cohort_key,
                "comparability": dict(first.get("comparability", {})),
                "run_count": len(cohort_trials),
                "runs": [
                    {
                        "run_id": trial.get("run_id"),
                        "run_label": trial.get("run_label"),
                        "created_at": trial.get("created_at"),
                        "record_quality": trial.get("record_quality"),
                        "status": dict(trial.get("status", {})),
                        "objectives": dict(trial.get("objectives", {})),
                        "objective_directions": dict(trial.get("objective_directions", {})),
                        "parameters": dict(trial.get("parameters", {})),
                        "artifact_root": trial.get("registry_source", {}).get("artifact_root"),
                    }
                    for trial in sorted(cohort_trials, key=lambda item: str(item.get("created_at", "") or ""))
                ],
            }
        )
    return {
        "schema_version": REGISTRY_SCHEMA,
        "generated_at": _utc_now_iso(),
        "optimization_profile": "accuracy_plus_ood",
        "cohort_count": len(cohorts),
        "cohorts": cohorts,
        "incomplete_trials": [
            {
                "run_id": trial.get("run_id"),
                "run_label": trial.get("run_label"),
                "record_quality": trial.get("record_quality"),
                "artifact_root": trial.get("registry_source", {}).get("artifact_root"),
            }
            for trial in incomplete
        ],
    }


def write_registry(
    *,
    trials: Iterable[JsonDict],
    output_root: str | Path,
) -> JsonDict:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    materialized_trials = list(trials)
    trials_jsonl = root / "trials.jsonl"
    with trials_jsonl.open("w", encoding="utf-8") as handle:
        for trial in materialized_trials:
            handle.write(json.dumps(trial, ensure_ascii=False))
            handle.write("\n")

    pareto_inputs = build_pareto_inputs(materialized_trials)
    pareto_inputs_path = write_json(root / "pareto_inputs.json", pareto_inputs, ensure_ascii=False)
    pareto_frontiers = build_pareto_frontiers(materialized_trials)
    pareto_frontiers_path = write_json(root / "pareto_frontiers.json", pareto_frontiers, ensure_ascii=False)
    stale_bayesian_recommendations_path = root / "bayesian_recommendations.json"
    if stale_bayesian_recommendations_path.exists():
        stale_bayesian_recommendations_path.unlink()
    latest_registry_payload = {
        "schema_version": REGISTRY_SCHEMA,
        "generated_at": pareto_inputs["generated_at"],
        "optimization_profile": "accuracy_plus_ood",
        "bayesian_optimization_enabled": False,
        "trial_count": len(materialized_trials),
        "canonical_count": sum(1 for trial in materialized_trials if trial.get("record_quality") == "canonical"),
        "backfilled_count": sum(1 for trial in materialized_trials if trial.get("record_quality") == "backfilled"),
        "cohort_count": int(pareto_inputs.get("cohort_count", 0)),
        "paths": {
            "trials_jsonl": str(trials_jsonl),
            "pareto_inputs_json": str(pareto_inputs_path),
            "pareto_frontiers_json": str(pareto_frontiers_path),
        },
        "cohorts": [
            {
                "cohort_key": cohort["cohort_key"],
                "run_count": cohort["run_count"],
                "comparability": cohort["comparability"],
            }
            for cohort in pareto_inputs.get("cohorts", [])
        ],
    }
    latest_registry_path = write_json(root / "latest_registry.json", latest_registry_payload, ensure_ascii=False)
    return {
        "trials_jsonl": trials_jsonl,
        "pareto_inputs_json": pareto_inputs_path,
        "pareto_frontiers_json": pareto_frontiers_path,
        "latest_registry_json": latest_registry_path,
        "latest_registry": latest_registry_payload,
    }


def build_run_registry(*, runs_root: str | Path, output_root: str | Path | None = None) -> JsonDict:
    resolved_runs_root = Path(runs_root)
    resolved_output_root = Path(output_root) if output_root is not None else (resolved_runs_root / "_index")
    trials = collect_trial_records(resolved_runs_root)
    return write_registry(trials=trials, output_root=resolved_output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    result = build_run_registry(runs_root=args.runs_root, output_root=args.output_root)
    print(json.dumps(result["latest_registry"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
