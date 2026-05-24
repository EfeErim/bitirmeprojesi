from scripts import calibrate_router_surface as calibrator


def test_parse_sweep_spec_accepts_alias_and_coerces_int_values():
    parameter, values = calibrator.parse_sweep_spec("sam3_prompt_limit=4,6")

    assert parameter == "router.vlm.sam3_prompt_limit"
    assert values == [4, 6]


def test_resolve_sweep_grid_includes_current_config_value():
    base_config = {
        "inference": {"router_min_confidence": 0.65},
        "router": {"vlm": {"confidence_threshold": 0.25}},
    }

    grid = calibrator.resolve_sweep_grid(
        base_config,
        preset="none",
        sweep_specs=["router_min_confidence=0.55,0.75"],
    )

    assert grid == {"inference.router_min_confidence": [0.65, 0.55, 0.75]}


def test_apply_overrides_mirrors_vlm_values_into_active_profile():
    base_config = {
        "router": {
            "vlm": {
                "profile": "balanced",
                "sam3_prompt_limit": 6,
                "profiles": {"balanced": {"sam3_prompt_limit": 6}},
            }
        }
    }

    config = calibrator.apply_overrides(base_config, {"router.vlm.sam3_prompt_limit": 4})

    assert config["router"]["vlm"]["sam3_prompt_limit"] == 4
    assert config["router"]["vlm"]["profiles"]["balanced"]["sam3_prompt_limit"] == 4
    assert base_config["router"]["vlm"]["profiles"]["balanced"]["sam3_prompt_limit"] == 6


def test_rank_variants_prefers_eligible_low_false_accept_config():
    baseline = {
        "metrics": {
            "crop_accuracy": 1.0,
            "part_non_unknown_precision": 1.0,
            "negative_false_accept_rate": 0.5,
        }
    }
    variants = [
        {
            "variant_id": "bad",
            "metrics": {
                "negative_false_accept_rate": 0.2,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 1.0,
                "crop_accuracy": 1.0,
                "part_non_unknown_precision": 1.0,
                "part_recall": 1.0,
                "abstention_rate": 0.1,
                "mean_latency_ms": 10.0,
            },
        },
        {
            "variant_id": "good",
            "metrics": {
                "negative_false_accept_rate": 0.0,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 1.0,
                "crop_accuracy": 1.0,
                "part_non_unknown_precision": 1.0,
                "part_recall": 1.0,
                "abstention_rate": 0.2,
                "mean_latency_ms": 12.0,
            },
        },
    ]

    ranked = calibrator.annotate_and_rank_variants(
        variants,
        baseline=baseline,
        target_negative_false_accept_rate=0.05,
    )

    assert ranked[0]["variant_id"] == "good"
    assert ranked[0]["eligible"] is True
    assert ranked[1]["eligible"] is False


def test_rank_variants_rejects_safety_metric_regressions():
    baseline = {
        "metrics": {
            "crop_accuracy": 0.90,
            "part_non_unknown_precision": 0.90,
            "part_recall": 0.80,
            "wrong_part_rejection_rate": 1.0,
            "p95_latency_ms": 100.0,
        }
    }
    variants = [
        {
            "variant_id": "unsafe_accuracy_gain",
            "metrics": {
                "negative_false_accept_rate": 0.0,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 0.70,
                "crop_accuracy": 0.95,
                "part_non_unknown_precision": 0.90,
                "part_recall": 0.70,
                "abstention_rate": 0.1,
                "mean_latency_ms": 100.0,
                "p95_latency_ms": 140.0,
            },
        },
        {
            "variant_id": "safe",
            "metrics": {
                "negative_false_accept_rate": 0.0,
                "unsupported_part_emissions": 0,
                "wrong_part_rejection_rate": 0.99,
                "crop_accuracy": 0.90,
                "part_non_unknown_precision": 0.90,
                "part_recall": 0.79,
                "abstention_rate": 0.2,
                "mean_latency_ms": 90.0,
                "p95_latency_ms": 110.0,
            },
        },
    ]

    ranked = calibrator.annotate_and_rank_variants(variants, baseline=baseline)

    assert ranked[0]["variant_id"] == "safe"
    assert ranked[0]["eligible"] is True
    assert ranked[1]["variant_id"] == "unsafe_accuracy_gain"
    assert ranked[1]["eligible"] is False
    assert ranked[1]["eligibility_reasons"] == [
        "part_recall_drop",
        "wrong_part_rejection_drop",
        "p95_latency_regression",
    ]
