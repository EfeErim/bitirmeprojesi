from src.router.confidence_utils import (
    resolve_effective_confidence_threshold,
    passes_open_set_gate,
)


def _policy_value_from(mapping):
    def _policy_value(stage, key, default):
        return mapping.get((stage, key), default)

    return _policy_value


def test_resolve_effective_confidence_threshold_applies_multiplier_and_clamps():
    policy_value = _policy_value_from(
        {
            ('execution', 'confidence_threshold_multiplier'): 1.5,
            ('execution', 'confidence_threshold_min'): 0.3,
            ('execution', 'confidence_threshold_max'): 0.8,
        }
    )

    value = resolve_effective_confidence_threshold(0.6, policy_value)
    assert value == 0.8


def test_resolve_effective_confidence_threshold_handles_invalid_policy_values():
    policy_value = _policy_value_from(
        {
            ('execution', 'confidence_threshold_multiplier'): 'bad',
            ('execution', 'confidence_threshold_min'): 'bad',
            ('execution', 'confidence_threshold_max'): 'bad',
        }
    )

    value = resolve_effective_confidence_threshold(0.55, policy_value)
    assert value == 0.55


def test_passes_open_set_gate_rejects_unknown_and_low_confidence():
    assert passes_open_set_gate('unknown', 0.99, 0.1) is False
    assert passes_open_set_gate('tomato', 0.49, 0.5) is False
    assert passes_open_set_gate('tomato', 0.5, 0.5) is True
