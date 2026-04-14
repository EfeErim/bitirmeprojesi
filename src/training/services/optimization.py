"""Registry-driven Pareto frontier and Bayesian proposal helpers."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

JsonDict = Dict[str, Any]

PARETO_FRONTIERS_SCHEMA = "v1_training_pareto_frontiers"
BAYESIAN_RECOMMENDATIONS_SCHEMA = "v1_training_bayesian_recommendations"
SEARCH_SPACE_SCHEMA = "v1_training_search_space"
OPTIMIZATION_PROFILE = "accuracy_plus_ood"
DEFAULT_PARETO_OBJECTIVES = (
    "classification.macro_f1",
    "ood.ood_auroc",
    "ood.ood_false_positive_rate",
)
DEFAULT_OBJECTIVE_WEIGHTS: JsonDict = {
    "classification.macro_f1": 0.5,
    "ood.ood_auroc": 0.3,
    "ood.ood_false_positive_rate": 0.2,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _copy_json(value: Any) -> Any:
    return copy.deepcopy(value)


def _objective_direction(
    trial: Mapping[str, Any],
    objective_name: str,
    fallback: str = "maximize",
) -> str:
    directions = dict(trial.get("objective_directions", {})) if isinstance(trial.get("objective_directions"), Mapping) else {}
    direction = str(directions.get(objective_name, fallback) or fallback).strip().lower()
    if direction not in {"maximize", "minimize"}:
        return fallback
    return direction


def _normalize_weights(
    objective_names: Sequence[str],
    objective_weights: Mapping[str, Any] | None = None,
) -> JsonDict:
    raw = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    if isinstance(objective_weights, Mapping):
        raw.update(dict(objective_weights))
    filtered = {
        str(name): max(0.0, float(raw.get(name, 1.0)))
        for name in objective_names
    }
    total = sum(filtered.values())
    if total <= 0.0:
        equal_weight = 1.0 / max(1, len(filtered))
        return {name: equal_weight for name in filtered}
    return {name: value / total for name, value in filtered.items()}


def _build_default_search_space_payload() -> JsonDict:
    return {
        "schema_version": SEARCH_SPACE_SCHEMA,
        "parameters": [
            {"name": "training.learning_rate", "type": "float", "low": 5e-5, "high": 3e-4, "scale": "log"},
            {"name": "training.weight_decay", "type": "float", "low": 1e-4, "high": 5e-2, "scale": "log"},
            {"name": "training.num_epochs", "type": "int", "low": 8, "high": 24, "step": 2},
            {"name": "training.batch_size", "type": "categorical", "values": [4, 8, 12, 16]},
            {"name": "training.adapter.lora_r", "type": "categorical", "values": [8, 16, 24, 32]},
            {"name": "training.adapter.lora_alpha", "type": "categorical", "values": [8, 16, 24, 32]},
            {"name": "training.adapter.lora_dropout", "type": "float", "low": 0.0, "high": 0.25},
            {"name": "training.fusion.dropout", "type": "float", "low": 0.0, "high": 0.25},
            {"name": "training.ood.threshold_factor", "type": "float", "low": 1.5, "high": 4.5},
            {"name": "training.optimization.logitnorm_tau", "type": "float", "low": 0.5, "high": 2.0},
            {"name": "training.data.randaugment_magnitude", "type": "int", "low": 3, "high": 12, "step": 1},
        ],
    }


DEFAULT_SEARCH_SPACE = _build_default_search_space_payload()


@dataclass(frozen=True)
class SearchDimension:
    name: str
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[int] = None
    scale: str = "linear"
    values: Tuple[Any, ...] = ()

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "float":
            if self.low is None or self.high is None:
                raise ValueError(f"Search dimension {self.name} is missing bounds.")
            if self.scale == "log":
                lower = math.log(self.low)
                upper = math.log(self.high)
                return float(math.exp(rng.uniform(lower, upper)))
            return float(rng.uniform(self.low, self.high))
        if self.kind == "int":
            if self.low is None or self.high is None:
                raise ValueError(f"Search dimension {self.name} is missing bounds.")
            step = max(1, int(self.step or 1))
            candidates = list(range(int(self.low), int(self.high) + 1, step))
            if not candidates:
                raise ValueError(f"Search dimension {self.name} has no integer candidates.")
            return int(rng.choice(candidates))
        if self.kind == "categorical":
            if not self.values:
                raise ValueError(f"Search dimension {self.name} is missing categorical values.")
            return _copy_json(rng.choice(self.values))
        raise ValueError(f"Unsupported search dimension type: {self.kind}")

    def encode(self, value: Any) -> Optional[List[float]]:
        if self.kind == "float":
            raw = _coerce_float(value)
            if raw is None or self.low is None or self.high is None:
                return None
            if self.high <= self.low:
                return [0.0]
            if self.scale == "log":
                if raw <= 0 or self.low <= 0 or self.high <= 0:
                    return None
                lower = math.log(self.low)
                upper = math.log(self.high)
                encoded = (math.log(raw) - lower) / max(1e-12, upper - lower)
            else:
                encoded = (raw - self.low) / max(1e-12, self.high - self.low)
            return [float(min(1.0, max(0.0, encoded)))]
        if self.kind == "int":
            raw = _coerce_float(value)
            if raw is None or self.low is None or self.high is None:
                return None
            if self.high <= self.low:
                return [0.0]
            encoded = (raw - self.low) / max(1e-12, self.high - self.low)
            return [float(min(1.0, max(0.0, encoded)))]
        if self.kind == "categorical":
            if not self.values:
                return None
            encoded = [0.0] * len(self.values)
            try:
                index = next(idx for idx, item in enumerate(self.values) if item == value)
            except StopIteration:
                return None
            encoded[index] = 1.0
            return encoded
        return None


def resolve_search_space(
    search_space_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> List[SearchDimension]:
    if search_space_payload is None:
        payload = dict(DEFAULT_SEARCH_SPACE)
    elif isinstance(search_space_payload, Mapping):
        payload = dict(search_space_payload)
    else:
        payload = {"parameters": list(search_space_payload)}

    raw_parameters = payload.get("parameters", [])
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("Search space must define a non-empty 'parameters' list.")

    dimensions: List[SearchDimension] = []
    for raw_dimension in raw_parameters:
        if not isinstance(raw_dimension, Mapping):
            raise ValueError("Each search-space parameter must be an object.")
        name = str(raw_dimension.get("name", "") or "").strip()
        kind = str(raw_dimension.get("type", "") or "").strip().lower()
        if not name or not kind:
            raise ValueError("Each search-space parameter requires 'name' and 'type'.")
        if kind == "categorical":
            values = tuple(_copy_json(list(raw_dimension.get("values", []))))
            if not values:
                raise ValueError(f"Categorical search-space parameter {name} must define values.")
            dimensions.append(SearchDimension(name=name, kind=kind, values=values))
            continue
        low = _coerce_float(raw_dimension.get("low"))
        high = _coerce_float(raw_dimension.get("high"))
        if low is None or high is None:
            raise ValueError(f"Search-space parameter {name} must define numeric low/high bounds.")
        dimensions.append(
            SearchDimension(
                name=name,
                kind=kind,
                low=low,
                high=high,
                step=int(raw_dimension.get("step", 1)) if kind == "int" else None,
                scale=str(raw_dimension.get("scale", "linear") or "linear").strip().lower(),
            )
        )
    return dimensions


def build_search_space_payload(dimensions: Sequence[SearchDimension]) -> JsonDict:
    parameters: List[JsonDict] = []
    for dimension in dimensions:
        payload: JsonDict = {"name": dimension.name, "type": dimension.kind}
        if dimension.kind == "categorical":
            payload["values"] = [_copy_json(value) for value in dimension.values]
        else:
            payload["low"] = dimension.low
            payload["high"] = dimension.high
            if dimension.kind == "int":
                payload["step"] = int(dimension.step or 1)
            if dimension.scale == "log":
                payload["scale"] = "log"
        parameters.append(payload)
    return {"schema_version": SEARCH_SPACE_SCHEMA, "parameters": parameters}


def group_trials_by_cohort(trials: Iterable[Mapping[str, Any]]) -> Dict[str, List[JsonDict]]:
    grouped: Dict[str, List[JsonDict]] = {}
    for trial in trials:
        payload = dict(trial)
        cohort_key = str(payload.get("comparability", {}).get("cohort_key", "") or "")
        if not cohort_key:
            continue
        grouped.setdefault(cohort_key, []).append(payload)
    return {
        cohort_key: sorted(
            cohort_trials,
            key=lambda item: (
                str(item.get("created_at", "") or ""),
                str(item.get("run_id", "") or ""),
            ),
        )
        for cohort_key, cohort_trials in grouped.items()
    }


def select_trials_for_cohort(
    trials: Iterable[Mapping[str, Any]],
    *,
    cohort_key: str | None = None,
    dataset_lineage_key: str | None = None,
    dataset_key: str | None = None,
    crop_name: str | None = None,
    part_name: str | None = None,
    backbone_model_name: str | None = None,
    engine: str | None = None,
) -> List[JsonDict]:
    selected: List[JsonDict] = []
    for trial in trials:
        payload = dict(trial)
        comparability = dict(payload.get("comparability", {})) if isinstance(payload.get("comparability"), Mapping) else {}
        if cohort_key and str(comparability.get("cohort_key", "") or "") != str(cohort_key):
            continue
        if dataset_lineage_key and str(comparability.get("dataset_lineage_key", "") or "") != str(dataset_lineage_key):
            continue
        if dataset_key and str(payload.get("dataset_key", "") or "") != str(dataset_key):
            continue
        if crop_name and _canonical_text(comparability.get("crop_name")) != _canonical_text(crop_name):
            continue
        if part_name and _canonical_text(comparability.get("part_name")) != _canonical_text(part_name):
            continue
        if backbone_model_name and str(comparability.get("backbone_model_name", "") or "") != str(backbone_model_name):
            continue
        if engine and _canonical_text(comparability.get("engine")) != _canonical_text(engine):
            continue
        selected.append(payload)
    return sorted(
        selected,
        key=lambda item: (
            str(item.get("created_at", "") or ""),
            str(item.get("run_id", "") or ""),
        ),
    )


def resolve_single_cohort(
    trials: Iterable[Mapping[str, Any]],
    **filters: Any,
) -> tuple[str, List[JsonDict]]:
    selected = select_trials_for_cohort(trials, **filters)
    grouped = group_trials_by_cohort(selected)
    if not grouped:
        raise ValueError("No comparable cohort matched the requested filters.")
    if len(grouped) != 1:
        raise ValueError(
            "Filters matched multiple comparable cohorts. Narrow the request with cohort_key or dataset_lineage_key."
        )
    cohort_key = next(iter(grouped))
    return cohort_key, grouped[cohort_key]


def build_training_config_override(flat_parameters: Mapping[str, Any]) -> JsonDict:
    override: JsonDict = {}
    for raw_key, raw_value in dict(flat_parameters).items():
        key = str(raw_key or "").strip()
        if not key.startswith("training."):
            continue
        path = ["training", "continual", *key.split(".")[1:]]
        current = override
        for part in path[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current[path[-1]] = _copy_json(raw_value)
    return override


def _collect_trial_objective_values(
    trial: Mapping[str, Any],
    objective_names: Sequence[str],
    *,
    require_readiness_passed: bool,
) -> tuple[Optional[JsonDict], Optional[str]]:
    status = dict(trial.get("status", {})) if isinstance(trial.get("status"), Mapping) else {}
    if require_readiness_passed and status.get("readiness_passed") is not True:
        return None, "readiness_not_passed"
    objectives_payload = dict(trial.get("objectives", {})) if isinstance(trial.get("objectives"), Mapping) else {}
    resolved: JsonDict = {}
    missing: List[str] = []
    for name in objective_names:
        value = _coerce_float(objectives_payload.get(name))
        if value is None:
            missing.append(name)
        else:
            resolved[name] = value
    if missing:
        return None, "missing_objectives:" + ",".join(missing)
    return resolved, None


def _dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    directions: Mapping[str, str],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for objective_name, direction in directions.items():
        left_value = float(left[objective_name])
        right_value = float(right[objective_name])
        if direction == "minimize":
            left_value = -left_value
            right_value = -right_value
        if left_value < right_value:
            better_or_equal = False
            break
        if left_value > right_value:
            strictly_better = True
    return better_or_equal and strictly_better


def build_pareto_frontiers(
    trials: Iterable[Mapping[str, Any]],
    *,
    objective_names: Sequence[str] = DEFAULT_PARETO_OBJECTIVES,
    require_readiness_passed: bool = True,
) -> JsonDict:
    grouped = group_trials_by_cohort(trials)
    cohorts: List[JsonDict] = []
    for cohort_key, cohort_trials in sorted(grouped.items(), key=lambda item: item[0]):
        directions = {
            objective_name: _objective_direction(cohort_trials[0], objective_name, "maximize")
            for objective_name in objective_names
        }
        eligible: List[Tuple[JsonDict, JsonDict]] = []
        excluded: List[JsonDict] = []
        for trial in cohort_trials:
            values, reason = _collect_trial_objective_values(
                trial,
                objective_names,
                require_readiness_passed=require_readiness_passed,
            )
            if values is None:
                excluded.append(
                    {
                        "run_id": trial.get("run_id"),
                        "run_label": trial.get("run_label"),
                        "reason": reason,
                    }
                )
                continue
            eligible.append((dict(trial), values))

        frontier_runs: List[JsonDict] = []
        for index, (trial, values) in enumerate(eligible):
            dominated = False
            for other_index, (_other_trial, other_values) in enumerate(eligible):
                if index == other_index:
                    continue
                if _dominates(other_values, values, directions):
                    dominated = True
                    break
            if dominated:
                continue
            frontier_runs.append(
                {
                    "run_id": trial.get("run_id"),
                    "run_label": trial.get("run_label"),
                    "created_at": trial.get("created_at"),
                    "record_quality": trial.get("record_quality"),
                    "status": _copy_json(trial.get("status", {})),
                    "objectives": _copy_json(trial.get("objectives", {})),
                    "parameters": _copy_json(trial.get("parameters", {})),
                    "artifact_root": trial.get("registry_source", {}).get("artifact_root"),
                }
            )

        comparability = (
            dict(cohort_trials[0].get("comparability", {}))
            if cohort_trials and isinstance(cohort_trials[0].get("comparability"), Mapping)
            else {}
        )
        cohorts.append(
            {
                "cohort_key": cohort_key,
                "comparability": comparability,
                "objective_names": list(objective_names),
                "objective_directions": directions,
                "eligible_run_count": len(eligible),
                "excluded_run_count": len(excluded),
                "frontier_count": len(frontier_runs),
                "frontier_run_ids": [run.get("run_id") for run in frontier_runs],
                "frontier_runs": frontier_runs,
                "excluded_runs": excluded,
            }
        )

    return {
        "schema_version": PARETO_FRONTIERS_SCHEMA,
        "generated_at": _utc_now_iso(),
        "optimization_profile": OPTIMIZATION_PROFILE,
        "objective_names": list(objective_names),
        "cohort_count": len(cohorts),
        "cohorts": cohorts,
    }


def _encode_parameters(parameters: Mapping[str, Any], dimensions: Sequence[SearchDimension]) -> Optional[np.ndarray]:
    encoded: List[float] = []
    for dimension in dimensions:
        if dimension.name not in parameters:
            return None
        component = dimension.encode(parameters.get(dimension.name))
        if component is None:
            return None
        encoded.extend(component)
    return np.asarray(encoded, dtype=float)


def _parameter_signature(parameters: Mapping[str, Any]) -> str:
    normalized = {str(key): _copy_json(value) for key, value in sorted(parameters.items(), key=lambda item: item[0])}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _normalize_objective_matrix(
    observed: Sequence[Tuple[JsonDict, JsonDict]],
    objective_names: Sequence[str],
    directions: Mapping[str, str],
    weights: Mapping[str, float],
) -> Tuple[List[float], List[JsonDict]]:
    ranges: JsonDict = {}
    for objective_name in objective_names:
        values = [float(objectives[objective_name]) for _trial, objectives in observed]
        ranges[objective_name] = {"min": min(values), "max": max(values)}

    normalized_rows: List[JsonDict] = []
    scores: List[float] = []
    for _trial, objectives in observed:
        normalized: JsonDict = {}
        score = 0.0
        for objective_name in objective_names:
            stats = ranges[objective_name]
            value = float(objectives[objective_name])
            low = float(stats["min"])
            high = float(stats["max"])
            if math.isclose(high, low, rel_tol=1e-12, abs_tol=1e-12):
                scaled = 1.0
            elif directions[objective_name] == "minimize":
                scaled = (high - value) / max(1e-12, high - low)
            else:
                scaled = (value - low) / max(1e-12, high - low)
            normalized[objective_name] = float(min(1.0, max(0.0, scaled)))
            score += weights[objective_name] * normalized[objective_name]
        normalized_rows.append(normalized)
        scores.append(float(score))
    return scores, normalized_rows


def _expected_improvement(mu: float, sigma: float, best_value: float, xi: float = 0.01) -> float:
    if sigma <= 1e-12:
        return 0.0
    improvement = mu - best_value - xi
    z_score = improvement / sigma
    cdf = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z_score * z_score) / math.sqrt(2.0 * math.pi)
    return float(improvement * cdf + sigma * pdf)


def _fit_gaussian_process(X: np.ndarray, y: np.ndarray) -> Any:
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except Exception:
        return None
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        random_state=42,
    )
    model.fit(X, y)
    return model


def _sample_candidate_pool(
    dimensions: Sequence[SearchDimension],
    *,
    rng: random.Random,
    observed_signatures: set[str],
    proposal_count: int,
    candidate_pool_size: int,
) -> List[JsonDict]:
    required = max(candidate_pool_size, proposal_count * 16)
    sampled: List[JsonDict] = []
    seen = set(observed_signatures)
    attempts = 0
    max_attempts = max(128, required * 30)
    while len(sampled) < required and attempts < max_attempts:
        attempts += 1
        candidate = {dimension.name: dimension.sample(rng) for dimension in dimensions}
        signature = _parameter_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        sampled.append(candidate)
    return sampled


def build_bayesian_recommendations(
    trials: Iterable[Mapping[str, Any]],
    *,
    objective_names: Sequence[str] = DEFAULT_PARETO_OBJECTIVES,
    objective_weights: Mapping[str, Any] | None = None,
    search_space_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    proposal_count: int = 3,
    candidate_pool_size: int = 256,
    random_seed: int = 42,
    require_readiness_passed: bool = True,
) -> JsonDict:
    grouped = group_trials_by_cohort(trials)
    dimensions = resolve_search_space(search_space_payload)
    search_space = build_search_space_payload(dimensions)
    weights = _normalize_weights(objective_names, objective_weights)
    cohorts: List[JsonDict] = []
    for cohort_key, cohort_trials in sorted(grouped.items(), key=lambda item: item[0]):
        directions = {
            objective_name: _objective_direction(cohort_trials[0], objective_name, "maximize")
            for objective_name in objective_names
        }
        observed_trials: List[Tuple[JsonDict, JsonDict]] = []
        excluded: List[JsonDict] = []
        for trial in cohort_trials:
            objectives, reason = _collect_trial_objective_values(
                trial,
                objective_names,
                require_readiness_passed=require_readiness_passed,
            )
            if objectives is None:
                excluded.append(
                    {"run_id": trial.get("run_id"), "run_label": trial.get("run_label"), "reason": reason}
                )
                continue
            encoded = _encode_parameters(dict(trial.get("parameters", {})), dimensions)
            if encoded is None:
                excluded.append(
                    {
                        "run_id": trial.get("run_id"),
                        "run_label": trial.get("run_label"),
                        "reason": "missing_search_space_parameters",
                    }
                )
                continue
            observed_trials.append((dict(trial), objectives))

        strategy = "random_bootstrap"
        best_run_id = ""
        best_score = None
        proposals: List[JsonDict] = []
        if dimensions:
            scores, normalized_rows = (
                _normalize_objective_matrix(observed_trials, objective_names, directions, weights)
                if observed_trials
                else ([], [])
            )
            observed_parameters = [dict(trial.get("parameters", {})) for trial, _objectives in observed_trials]
            observed_signatures = {_parameter_signature(parameters) for parameters in observed_parameters}
            rng = random.Random(random_seed + sum(ord(ch) for ch in cohort_key))
            candidates = _sample_candidate_pool(
                dimensions,
                rng=rng,
                observed_signatures=observed_signatures,
                proposal_count=proposal_count,
                candidate_pool_size=candidate_pool_size,
            )
            encoded_candidates = [
                _encode_parameters(candidate, dimensions)
                for candidate in candidates
            ]
            encoded_candidates = [encoded for encoded in encoded_candidates if encoded is not None]
            if observed_trials and scores:
                best_index = max(range(len(scores)), key=lambda idx: scores[idx])
                best_run_id = str(observed_trials[best_index][0].get("run_id", "") or "")
                best_score = float(scores[best_index])
            X = (
                np.vstack([_encode_parameters(parameters, dimensions) for parameters in observed_parameters])
                if observed_parameters
                else np.empty((0, 0))
            )
            y = np.asarray(scores, dtype=float) if scores else np.empty((0,), dtype=float)
            model = None
            if len(observed_trials) >= 3 and X.size > 0 and y.size > 0 and encoded_candidates:
                model = _fit_gaussian_process(X, y)
            ranked_candidates: List[Tuple[JsonDict, float, Optional[float], float]] = []
            if model is not None and encoded_candidates:
                strategy = "gaussian_process_expected_improvement"
                stacked_candidates = np.vstack(encoded_candidates)
                means, stds = model.predict(stacked_candidates, return_std=True)
                for candidate, encoded, mean, std in zip(candidates, encoded_candidates, means, stds):
                    acquisition = _expected_improvement(float(mean), float(std), float(best_score or 0.0))
                    exploration = (
                        float(min(np.linalg.norm(X - encoded, axis=1)))
                        if X.size > 0
                        else 0.0
                    )
                    ranked_candidates.append((candidate, float(acquisition), float(mean), exploration))
                ranked_candidates.sort(key=lambda item: (item[1], item[2] if item[2] is not None else -1.0), reverse=True)
            else:
                for candidate, encoded in zip(candidates, encoded_candidates):
                    exploration = (
                        float(min(np.linalg.norm(X - encoded, axis=1)))
                        if X.size > 0
                        else 0.0
                    )
                    ranked_candidates.append((candidate, exploration, None, exploration))
                ranked_candidates.sort(key=lambda item: item[1], reverse=True)

            for rank, (candidate, acquisition, prediction, exploration) in enumerate(
                ranked_candidates[: max(0, proposal_count)],
                start=1,
            ):
                proposals.append(
                    {
                        "rank": rank,
                        "parameters": candidate,
                        "config_override": build_training_config_override(candidate),
                        "acquisition_score": float(acquisition),
                        "predicted_score": float(prediction) if prediction is not None else None,
                        "exploration_distance": float(exploration),
                        "signature": _parameter_signature(candidate),
                    }
                )

        comparability = (
            dict(cohort_trials[0].get("comparability", {}))
            if cohort_trials and isinstance(cohort_trials[0].get("comparability"), Mapping)
            else {}
        )
        cohorts.append(
            {
                "cohort_key": cohort_key,
                "comparability": comparability,
                "objective_names": list(objective_names),
                "objective_directions": directions,
                "objective_weights": weights,
                "observed_run_count": len(cohort_trials),
                "eligible_run_count": len(observed_trials),
                "excluded_run_count": len(excluded),
                "search_strategy": strategy,
                "best_observed_run_id": best_run_id,
                "best_observed_score": best_score,
                "normalized_observed_objectives": normalized_rows if observed_trials else [],
                "proposals": proposals,
                "excluded_runs": excluded,
            }
        )

    return {
        "schema_version": BAYESIAN_RECOMMENDATIONS_SCHEMA,
        "generated_at": _utc_now_iso(),
        "optimization_profile": OPTIMIZATION_PROFILE,
        "objective_names": list(objective_names),
        "objective_weights": weights,
        "proposal_count": int(max(0, proposal_count)),
        "search_space": search_space,
        "cohort_count": len(cohorts),
        "cohorts": cohorts,
    }


__all__ = [
    "BAYESIAN_RECOMMENDATIONS_SCHEMA",
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "DEFAULT_PARETO_OBJECTIVES",
    "DEFAULT_SEARCH_SPACE",
    "OPTIMIZATION_PROFILE",
    "PARETO_FRONTIERS_SCHEMA",
    "SEARCH_SPACE_SCHEMA",
    "build_bayesian_recommendations",
    "build_pareto_frontiers",
    "build_search_space_payload",
    "build_training_config_override",
    "group_trials_by_cohort",
    "resolve_search_space",
    "resolve_single_cohort",
    "select_trials_for_cohort",
]
