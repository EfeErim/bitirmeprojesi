from typing import Any, Callable


def _clamp_unit_interval(value: Any, *, default: float) -> float:
    try:
        resolved = float(value)
    except Exception:
        resolved = float(default)
    return max(0.0, min(1.0, resolved))


def _coerce_non_negative_float(value: Any, *, default: float) -> float:
    try:
        resolved = float(value)
    except Exception:
        resolved = float(default)
    return max(0.0, resolved)


def resolve_effective_confidence_threshold(
    confidence_threshold: float,
    policy_value_fn: Callable[[str, str, Any], Any],
) -> float:
    """Resolve profile/policy-adjusted confidence threshold with optional clamps."""
    base_threshold = _clamp_unit_interval(confidence_threshold, default=0.0)

    multiplier_raw = policy_value_fn('execution', 'confidence_threshold_multiplier', 1.0)
    multiplier = _coerce_non_negative_float(multiplier_raw, default=1.0)

    adjusted = base_threshold * multiplier

    min_raw = policy_value_fn('execution', 'confidence_threshold_min', 0.0)
    max_raw = policy_value_fn('execution', 'confidence_threshold_max', 1.0)
    min_threshold = _clamp_unit_interval(min_raw, default=0.0)
    max_threshold = max(min_threshold, _clamp_unit_interval(max_raw, default=1.0))
    return max(min_threshold, min(max_threshold, adjusted))


def passes_open_set_gate(crop_label: str, crop_confidence: float, min_confidence: float) -> bool:
    """Evaluate open-set acceptance for a classified detection."""
    if str(crop_label).strip().lower() == 'unknown':
        return False
    return float(crop_confidence) >= float(min_confidence)
