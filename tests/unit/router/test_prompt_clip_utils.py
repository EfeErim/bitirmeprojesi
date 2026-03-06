import torch

from src.router.prompt_clip_utils import (
    aggregate_prompt_logits,
    build_prompt_batch,
    build_prompt_ensemble,
    crop_prompt_aliases,
    get_clip_logit_scale,
    get_prompt_templates_for_type,
    open_set_unknown_prompts,
)


def test_crop_prompt_aliases_contains_known_crops():
    aliases = crop_prompt_aliases()
    assert 'tomato' in aliases
    assert 'Solanum lycopersicum' in aliases['tomato']


def test_build_prompt_ensemble_uses_custom_templates_when_configured():
    prompts = build_prompt_ensemble(
        label='tomato',
        label_type='crop',
        vlm_config={'prompt_templates': {'crop': ['custom {term}']}},
    )
    assert prompts == ['custom tomato', 'custom Solanum lycopersicum']


def test_build_prompt_batch_falls_back_to_label_when_ensemble_empty():
    prompt_texts, prompt_to_class = build_prompt_batch(
        labels=[''],
        label_type='crop',
        vlm_config={},
    )
    assert prompt_texts == ['']
    assert prompt_to_class == [0]


def test_get_prompt_templates_for_type_uses_default_and_fallback():
    cfg = {'prompt_templates': {'default': ['x {term}']}}
    assert get_prompt_templates_for_type(cfg, 'crop') == ['x {term}']

    assert get_prompt_templates_for_type({'prompt_templates': {'crop': []}}, 'crop') == ['{term}']


def test_open_set_unknown_prompts_differs_by_label_type():
    part_prompts = open_set_unknown_prompts('part')
    crop_prompts = open_set_unknown_prompts('crop')
    assert len(part_prompts) == 4
    assert len(crop_prompts) == 5


def test_aggregate_prompt_logits_max_pools_by_class():
    logits = torch.tensor([[0.2, 0.7, 0.4, 0.5]], dtype=torch.float32)
    prompt_to_class = [0, 1, 0, 1]
    aggregated = aggregate_prompt_logits(logits, prompt_to_class, num_classes=2)
    assert aggregated.shape == (1, 2)
    assert torch.allclose(aggregated, torch.tensor([[0.4, 0.7]], dtype=torch.float32))


def test_get_clip_logit_scale_handles_tensor_and_missing_attr():
    class _Model:
        pass

    m = _Model()
    m.logit_scale = torch.tensor(0.0)
    assert get_clip_logit_scale(m) == 1.0

    assert get_clip_logit_scale(object()) == 1.0
