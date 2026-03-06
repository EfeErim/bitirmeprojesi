from typing import Any, Callable


def resolve_effective_confidence_threshold(
    confidence_threshold: float,
    policy_value_fn: Callable[[str, str, Any], Any],
) -> float:
    """Resolve profile/policy-adjusted confidence threshold with optional clamps."""
    base_threshold = max(0.0, min(1.0, float(confidence_threshold)))

    multiplier_raw = policy_value_fn('execution', 'confidence_threshold_multiplier', 1.0)
    try:
        multiplier = float(multiplier_raw)
    except Exception:
        multiplier = 1.0
    multiplier = max(0.0, multiplier)

    adjusted = base_threshold * multiplier

    min_raw = policy_value_fn('execution', 'confidence_threshold_min', 0.0)
    max_raw = policy_value_fn('execution', 'confidence_threshold_max', 1.0)
    try:
        min_threshold = float(min_raw)
    except Exception:
        min_threshold = 0.0
    try:
        max_threshold = float(max_raw)
    except Exception:
        max_threshold = 1.0

    min_threshold = max(0.0, min(1.0, min_threshold))
    max_threshold = max(min_threshold, min(1.0, max_threshold))
    return max(min_threshold, min(max_threshold, adjusted))


def passes_open_set_gate(crop_label: str, crop_confidence: float, min_confidence: float) -> bool:
    """Evaluate open-set acceptance for a classified detection."""
    if str(crop_label).strip().lower() == 'unknown':
        return False
    return float(crop_confidence) >= float(min_confidence)
