from src.router.runtime_settings import build_sam3_runtime_settings, resolve_sam3_stage_order


def _policy_value_from(mapping):
    def _policy_value(stage, key, default):
        return mapping.get((stage, key), default)

    return _policy_value


def test_resolve_sam3_stage_order_filters_invalid_and_duplicates():
    policy_value = _policy_value_from(
        {
            ('execution', 'sam3_stage_order'): [
                'roi_filter',
                'invalid_stage',
                'roi_classification',
                'roi_filter',
                'postprocess',
            ]
        }
    )

    order = resolve_sam3_stage_order(policy_value)
    assert order == ['roi_filter', 'roi_classification', 'postprocess']


def test_resolve_sam3_stage_order_uses_default_on_non_list():
    policy_value = _policy_value_from({('execution', 'sam3_stage_order'): 'not-a-list'})

    order = resolve_sam3_stage_order(policy_value)
    assert order == ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']


def test_build_sam3_runtime_settings_applies_threshold_floor_and_parsing():
    policy_value = _policy_value_from(
        {
            ('crop_evidence', 'classification_min_confidence'): 0.1,
            ('crop_evidence', 'global_crop_context_weight'): 0.7,
            ('roi_filter', 'max_rois_for_classification'): '3',
            ('focus_mode', 'focus_parts'): ['leaf', 'fruit'],
            ('part_resolution', 'generic_part_labels'): ['whole', 'plant'],
        }
    )

    settings = build_sam3_runtime_settings(
        policy_value_fn=policy_value,
        vlm_config={'ensemble_config': {'crop_num_prompts': '2', 'part_num_prompts': 4}},
        effective_threshold=0.65,
    )

    assert settings['classification_min_confidence'] == 0.65
    assert settings['max_rois_for_classification'] == 3
    assert settings['crop_num_prompts'] == 2
    assert settings['part_num_prompts'] == 4
    assert settings['global_crop_context_enabled'] is True
    assert settings['global_crop_context_weight'] == 0.7
    assert settings['focus_parts'] == ['leaf', 'fruit']
    assert settings['generic_part_labels'] == ['whole', 'plant']


def test_build_sam3_runtime_settings_uses_conservative_leaf_defaults():
    settings = build_sam3_runtime_settings(
        policy_value_fn=_policy_value_from({}),
        vlm_config={'ensemble_config': {}},
        effective_threshold=0.25,
    )

    assert settings['preferred_part_labels'] == []
    assert settings['leaf_override_target_labels'] == ['whole plant', 'whole', 'plant', 'entire plant']
    assert settings['leaf_override_min_margin'] == 0.04
    assert settings['leaf_visual_force_without_leaf_score'] is False
    assert settings['leaf_visual_min_margin'] == 0.05
    assert settings['leaf_part_rebalance_threshold'] == 0.52
    assert settings['leaf_part_rebalance_min_confidence'] == 0.18
    assert settings['leaf_part_rebalance_support_ratio'] == 0.75
