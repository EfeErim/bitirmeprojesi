import numpy as np

from src.ood.dynamic_thresholds import DynamicOODThreshold


def test_dynamic_threshold_default_initialization():
    threshold_computer = DynamicOODThreshold()
    assert threshold_computer.min_val_samples_per_class == 30
    assert threshold_computer.threshold_factor == 2.0
    assert threshold_computer.use_confidence_intervals is True


def test_dynamic_threshold_confidence_interval_and_fallback_surfaces():
    threshold_computer = DynamicOODThreshold(
        threshold_factor=2.576,
        min_val_samples_per_class=50,
        fallback_threshold=30.0,
        confidence_level=0.99,
    )
    data = np.random.normal(10.0, 2.0, 50)
    lower, upper = threshold_computer._compute_confidence_interval(data, 0.95)
    assert lower <= float(np.mean(data)) <= upper

    threshold_zero = threshold_computer._handle_insufficient_samples(class_idx=0, sample_count=0)
    threshold_five = threshold_computer._handle_insufficient_samples(class_idx=0, sample_count=5)
    assert threshold_zero >= threshold_computer.min_fallback_threshold
    assert threshold_zero <= threshold_computer.max_fallback_threshold
    assert threshold_five >= threshold_computer.min_fallback_threshold
    assert threshold_five <= threshold_computer.max_fallback_threshold
