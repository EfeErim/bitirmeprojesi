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
    assert settings['focus_parts'] == ['leaf', 'fruit']
    assert settings['generic_part_labels'] == ['whole', 'plant']
