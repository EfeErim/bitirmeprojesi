from typing import Any, Dict, List, Optional, Tuple

import torch

TURKEY_PRIORITY_CROP_ALIASES: Dict[str, List[str]] = {
    'wheat': ['wheat', 'bread wheat', 'durum wheat', 'Triticum aestivum', 'Triticum durum'],
    'sugar beet': ['sugar beet', 'Beta vulgaris'],
    'tomato': ['tomato', 'Solanum lycopersicum'],
    'barley': ['barley', 'Hordeum vulgare'],
    'maize': ['maize', 'corn', 'Zea mays'],
    'potato': ['potato', 'Solanum tuberosum'],
    'grape': ['grape', 'grapevine', 'Vitis vinifera'],
    'strawberry': ['strawberry', 'Fragaria × ananassa'],
    'apricot': ['apricot', 'Prunus armeniaca'],
    'hazelnut': ['hazelnut', 'hazel', 'Corylus avellana'],
    'olive': ['olive', 'olive tree', 'Olea europaea'],
    'sunflower': ['sunflower', 'Helianthus annuus'],
    'cotton': ['cotton', 'Gossypium'],
}

TURKEY_PRIORITY_CROP_PARTS: Dict[str, List[str]] = {
    'wheat': ['leaf', 'stem', 'root', 'grain', 'ear', 'spike', 'kernel', 'seedling', 'whole plant'],
    'sugar beet': ['leaf', 'root', 'crown', 'stem', 'seed', 'seedling', 'whole plant'],
    'tomato': ['leaf', 'fruit', 'flower', 'stem', 'seed', 'bud', 'shoot', 'seedling', 'whole plant'],
    'barley': ['leaf', 'stem', 'root', 'grain', 'ear', 'spike', 'kernel', 'seedling', 'whole plant'],
    'maize': ['leaf', 'stem', 'root', 'grain', 'ear', 'cob', 'kernel', 'tassel', 'husk', 'seedling', 'whole plant'],
    'potato': ['leaf', 'tuber', 'flower', 'stem', 'root', 'shoot', 'seedling', 'whole plant'],
    'grape': ['leaf', 'fruit', 'berry', 'vine', 'stem', 'tendril', 'shoot', 'whole plant'],
    'strawberry': ['leaf', 'fruit', 'berry', 'flower', 'crown', 'root', 'shoot', 'seedling', 'whole plant'],
    'apricot': ['leaf', 'fruit', 'flower', 'branch', 'stem', 'bark', 'trunk', 'seed', 'bud', 'shoot', 'whole plant'],
    'hazelnut': [
        'leaf', 'nut', 'kernel', 'shell', 'husk', 'involucre', 'cupule',
        'cluster', 'fruit cluster', 'nut cluster', 'hazelnut cluster',
        'flower', 'male catkin', 'female flower', 'catkin', 'branch', 'shoot',
        'stem', 'bark', 'trunk', 'root', 'bud', 'whole plant',
    ],
    'olive': ['leaf', 'fruit', 'flower', 'branch', 'stem', 'bark', 'trunk', 'seed', 'whole plant'],
    'sunflower': ['leaf', 'flower', 'head', 'stem', 'seed', 'root', 'bud', 'seedling', 'whole plant'],
    'cotton': ['leaf', 'flower', 'boll', 'stem', 'root', 'seed', 'bud', 'branch', 'whole plant'],
}


def _crop_key(label: str) -> str:
    return str(label or '').strip().lower()


def default_prompt_templates_for_type(label_type: str) -> List[str]:
    """Return built-in prompt templates for a semantic label type."""
    if label_type == 'part':
        return [
            "a photo of a plant {term}",
            "a close-up photo of a plant {term}",
            "a macro photo of a plant {term}",
            "a {term} with damage",
            "a {term} with disease",
            "a diseased plant {term}",
        ]
    if label_type == 'crop':
        return [
            "a photo of {term}",
            "a close-up photo of {term}",
            "a macro photo of {term}",
            "an image of {term}",
            "a {term} crop",
            "a {term} plant",
            "a {term} with disease",
            "a diseased {term}",
        ]
    return ["{term}"]


def _crop_part_prompt_variants(term: str, part: str) -> List[str]:
    return [
        f"a photo of a {term} {part}",
        f"a close-up photo of a {term} {part}",
        f"a diseased {term} {part}",
    ]


def get_prompt_templates_for_type(vlm_config: Dict[str, Any], label_type: str) -> List[str]:
    """Get dynamic prompt templates from config with sane default fallback."""
    templates_cfg = vlm_config.get('prompt_templates', {}) if isinstance(vlm_config, dict) else {}
    if not isinstance(templates_cfg, dict):
        templates_cfg = {}

    label_templates = templates_cfg.get(label_type, templates_cfg.get('default', []))
    if not isinstance(label_templates, list):
        label_templates = []

    cleaned_templates = [str(template).strip() for template in label_templates if str(template).strip()]
    return cleaned_templates if cleaned_templates else default_prompt_templates_for_type(label_type)


def crop_prompt_aliases() -> Dict[str, List[str]]:
    """Return crop prompt aliases including scientific names where available."""
    aliases = dict(TURKEY_PRIORITY_CROP_ALIASES)
    return aliases


def build_prompt_ensemble(label: str, label_type: str, vlm_config: Dict[str, Any]) -> List[str]:
    """Build multiple prompt variants for a single semantic class label."""
    label_text = str(label).strip()
    if not label_text:
        return []

    prompt_templates_cfg = vlm_config.get('prompt_templates', {}) if isinstance(vlm_config, dict) else {}
    custom_templates = prompt_templates_cfg.get(label_type, []) if isinstance(prompt_templates_cfg, dict) else []

    if label_type == 'part':
        templates = custom_templates if custom_templates else default_prompt_templates_for_type(label_type)
        base_terms = [label_text]
    else:
        aliases = crop_prompt_aliases()
        alias_terms = aliases.get(label_text.lower(), [label_text])
        base_terms = [term for term in [label_text, *alias_terms] if isinstance(term, str) and term.strip()]

        templates = custom_templates if custom_templates else default_prompt_templates_for_type(label_type)

    prompts: List[str] = []
    seen = set()
    for term in base_terms:
        clean_term = term.strip()
        for template in templates:
            prompt = template.format(term=clean_term)
            key = prompt.lower()
            if key not in seen:
                seen.add(key)
                prompts.append(prompt)
        if label_type == 'crop' and not custom_templates:
            for part_name in TURKEY_PRIORITY_CROP_PARTS.get(_crop_key(label_text), []):
                for prompt in _crop_part_prompt_variants(clean_term, part_name):
                    key = prompt.lower()
                    if key not in seen:
                        seen.add(key)
                        prompts.append(prompt)
    return prompts


def build_prompt_batch(labels: List[str], label_type: str, vlm_config: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """Build prompt list and class index mapping for class-level aggregation."""
    prompt_texts: List[str] = []
    prompt_to_class: List[int] = []

    for class_index, label in enumerate(labels):
        class_prompts = build_prompt_ensemble(label, label_type=label_type, vlm_config=vlm_config)
        if not class_prompts:
            class_prompts = [str(label)]
        prompt_texts.extend(class_prompts)
        prompt_to_class.extend([class_index] * len(class_prompts))

    return prompt_texts, prompt_to_class


def open_set_unknown_prompts(label_type: str, known_labels: Optional[List[str]] = None) -> List[str]:
    """Prompt set for unknown/out-of-scope rejection."""
    _ = known_labels
    if label_type == 'part':
        return [
            "a photo of a rock or stone",
            "a photo of water or liquid",
            "a photo of a building or concrete",
            "a photo of an animal or insect",
        ]

    return [
        "a photo of a rock or stone",
        "a photo of water or liquid surface",
        "a photo of soil or dirt only",
        "a photo of a building or concrete",
        "a photo of an unrelated object",
    ]


def aggregate_prompt_logits(logits: torch.Tensor, prompt_to_class: List[int], num_classes: int) -> torch.Tensor:
    """Aggregate prompt-level logits into class-level logits using max pooling."""
    if logits.ndim == 2:
        logits = logits.squeeze(0)

    class_logits = torch.full(
        (num_classes,),
        float('-inf'),
        device=logits.device,
        dtype=logits.dtype,
    )

    for prompt_index, class_index in enumerate(prompt_to_class):
        class_logits[class_index] = torch.maximum(class_logits[class_index], logits[prompt_index])

    return class_logits.unsqueeze(0)


def get_clip_logit_scale(model: Any) -> float:
    """Get CLIP logit scale (temperature inverse) with safe fallback."""
    logit_scale_attr = getattr(model, 'logit_scale', None)
    if logit_scale_attr is None:
        return 1.0

    try:
        if torch.is_tensor(logit_scale_attr):
            scale_value = logit_scale_attr.detach().float().squeeze().exp().item()
        elif hasattr(logit_scale_attr, 'data') and torch.is_tensor(logit_scale_attr.data):
            scale_value = logit_scale_attr.data.detach().float().squeeze().exp().item()
        else:
            scale_value = float(logit_scale_attr)
        return max(1.0, min(scale_value, 100.0))
    except (TypeError, ValueError, OverflowError, RuntimeError, AttributeError):
        return 1.0
