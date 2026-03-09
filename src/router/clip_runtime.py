"""Runtime scoring helpers for CLIP/BioCLIP-backed router inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.router.prompt_clip_utils import (
    aggregate_prompt_logits,
    build_prompt_batch,
    get_clip_logit_scale,
    get_prompt_templates_for_type,
    open_set_unknown_prompts,
)

logger = logging.getLogger(__name__)


@dataclass
class ClipScoreRequest:
    """Request-local state so one scoring flow does not re-encode the same image."""

    image: Image.Image
    image_embedding: Optional[torch.Tensor] = None


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def _open_clip_components(runtime: Any) -> tuple[Dict[str, Any], Any]:
    processor = runtime.bioclip_processor
    model = runtime.bioclip
    if not isinstance(processor, dict) or model is None:
        raise RuntimeError("open_clip encoding requested before BioCLIP runtime was initialized.")
    return processor, model


def _processor_components(runtime: Any) -> tuple[Any, Any]:
    processor = runtime.bioclip_processor
    model = runtime.bioclip
    if processor is None or model is None:
        raise RuntimeError("BioCLIP scoring requested before BioCLIP runtime was initialized.")
    return processor, model


def _resolve_logits_per_image(model_outputs: Any, model: Any) -> torch.Tensor:
    if hasattr(model_outputs, "logits_per_image") and model_outputs.logits_per_image is not None:
        return model_outputs.logits_per_image
    if hasattr(model_outputs, "image_embeds") and hasattr(model_outputs, "text_embeds"):
        image_embeds = _normalize_embeddings(model_outputs.image_embeds)
        text_embeds = _normalize_embeddings(model_outputs.text_embeds)
        logit_scale = get_clip_logit_scale(model)
        return (image_embeds @ text_embeds.T) * logit_scale
    raise RuntimeError("BioCLIP model output does not provide logits_per_image or embeddable outputs")


def _build_open_clip_image_batch(
    preprocess: Any,
    images: List[Image.Image],
    device: Any,
) -> torch.Tensor:
    processed_images = [preprocess(image) for image in images]
    return torch.stack(processed_images, dim=0).to(device)


def _limit_prompt_templates(
    runtime: Any,
    *,
    label_type: str,
    num_prompts: Optional[int],
) -> List[str]:
    templates = get_prompt_templates_for_type(runtime.vlm_config, label_type)
    if num_prompts is None:
        return templates
    try:
        prompt_limit = int(num_prompts)
    except Exception:
        prompt_limit = 0
    if prompt_limit > 0:
        return templates[:prompt_limit]
    return templates


def _average_label_scores(label_score_lists: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        label: float(np.mean(scores)) if scores else 0.0
        for label, scores in label_score_lists.items()
    }


def _best_label_and_score(label_scores: Dict[str, float]) -> tuple[str, float]:
    best_label = max(label_scores, key=lambda label: label_scores[label]) if label_scores else "unknown"
    return best_label, label_scores.get(best_label, 0.0)


def _score_processor_batch(
    runtime: Any,
    images: Image.Image | List[Image.Image],
    prompts: List[str],
) -> torch.Tensor:
    processor, model = _processor_components(runtime)
    model_inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {key: value.to(runtime.device) for key, value in model_inputs.items()}

    with torch.no_grad():
        model_outputs = model(**model_inputs)
        logits_per_image = _resolve_logits_per_image(model_outputs, model)
    return torch.softmax(logits_per_image, dim=-1)


def _aggregate_prompt_probabilities(
    runtime: Any,
    *,
    request: ClipScoreRequest,
    text_prompts: List[str],
    prompt_to_class: List[int],
    class_count: int,
) -> torch.Tensor:
    if runtime.bioclip_backend == "open_clip":
        if runtime.bioclip is None:
            raise RuntimeError("open_clip scoring requested before BioCLIP runtime was initialized.")
        logit_scale = get_clip_logit_scale(runtime.bioclip)
        image_embeds = ensure_open_clip_image_embedding(runtime, request)
        text_embeds = get_open_clip_text_embeddings(runtime, text_prompts)
        with torch.no_grad():
            prompt_logits = (image_embeds @ text_embeds.T) * logit_scale
    else:
        processor, model = _processor_components(runtime)
        model_inputs = processor(
            text=text_prompts,
            images=request.image,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {key: value.to(runtime.device) for key, value in model_inputs.items()}
        with torch.no_grad():
            model_outputs = model(**model_inputs)
            prompt_logits = _resolve_logits_per_image(model_outputs, model)

    with torch.no_grad():
        logits = aggregate_prompt_logits(prompt_logits, prompt_to_class, class_count)
    return torch.softmax(logits, dim=-1)


def get_open_clip_text_embeddings(runtime: Any, prompts: List[str]) -> torch.Tensor:
    """Get normalized text embeddings for prompts with lightweight in-memory caching."""
    cache_key = tuple(prompts)
    cached = runtime._open_clip_text_embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    processor, model = _open_clip_components(runtime)

    tokenizer = processor["tokenizer"]
    text_tokens = tokenizer(prompts).to(runtime.device)
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds = _normalize_embeddings(text_embeds)

    max_cache_size = int(runtime.vlm_config.get("open_clip_text_cache_size", 64))
    if len(runtime._open_clip_text_embedding_cache) >= max_cache_size and max_cache_size > 0:
        oldest_key = next(iter(runtime._open_clip_text_embedding_cache))
        runtime._open_clip_text_embedding_cache.pop(oldest_key, None)
    if max_cache_size > 0:
        runtime._open_clip_text_embedding_cache[cache_key] = text_embeds
    return text_embeds


def ensure_open_clip_image_embedding(runtime: Any, request: ClipScoreRequest) -> torch.Tensor:
    """Encode a single image once per scoring request for open_clip scoring."""
    if request.image_embedding is not None:
        return request.image_embedding

    processor, model = _open_clip_components(runtime)

    preprocess = processor["preprocess"]
    image_tensor = preprocess(request.image).unsqueeze(0).to(runtime.device)
    with torch.no_grad():
        image_embeds = model.encode_image(image_tensor)
        request.image_embedding = _normalize_embeddings(image_embeds)
    return request.image_embedding


def ensure_open_clip_image_embeddings(
    runtime: Any,
    images: List[Image.Image],
    *,
    image_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Encode many images with open_clip in image batches."""
    if not images:
        return torch.empty((0, 0), dtype=torch.float32, device=runtime.device)

    processor, model = _open_clip_components(runtime)

    preprocess = processor["preprocess"]
    batch_size = max(1, int(image_batch_size or len(images)))
    embeddings: List[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        image_tensor = _build_open_clip_image_batch(
            preprocess,
            images[start : start + batch_size],
            runtime.device,
        )
        with torch.no_grad():
            image_embeds = model.encode_image(image_tensor)
            embeddings.append(_normalize_embeddings(image_embeds))
    return torch.cat(embeddings, dim=0)


def score_open_clip_with_image_embedding(
    runtime: Any,
    image_embedding: torch.Tensor,
    prompts: List[str],
) -> List[float]:
    """Score prompts against a precomputed open_clip image embedding."""
    text_embeds = get_open_clip_text_embeddings(runtime, prompts)
    if runtime.bioclip is None:
        raise RuntimeError("open_clip scoring requested before BioCLIP runtime was initialized.")
    logit_scale = get_clip_logit_scale(runtime.bioclip)
    with torch.no_grad():
        logits_per_image = (image_embedding @ text_embeds.T) * logit_scale
        scores = torch.softmax(logits_per_image, dim=-1)[0]
    return scores.detach().cpu().numpy().tolist()


def score_open_clip_with_image_embeddings(
    runtime: Any,
    image_embeddings: torch.Tensor,
    prompts: List[str],
) -> torch.Tensor:
    """Score prompts against precomputed open_clip image embeddings."""
    text_embeds = get_open_clip_text_embeddings(runtime, prompts)
    if runtime.bioclip is None:
        raise RuntimeError("open_clip scoring requested before BioCLIP runtime was initialized.")
    logit_scale = get_clip_logit_scale(runtime.bioclip)
    with torch.no_grad():
        logits_per_image = (image_embeddings @ text_embeds.T) * logit_scale
    return torch.softmax(logits_per_image, dim=-1)


def encode_and_score(runtime: Any, request: ClipScoreRequest, prompts: List[str]) -> List[float]:
    """Encode image and score against text prompts."""
    try:
        if runtime.bioclip_backend == "open_clip":
            image_embedding = ensure_open_clip_image_embedding(runtime, request)
            return score_open_clip_with_image_embedding(runtime, image_embedding, prompts)

        scores = _score_processor_batch(runtime, request.image, prompts)[0]
        return scores.detach().cpu().numpy().tolist()
    except Exception as exc:
        logger.error("Encoding failed: %s", exc)
        return [0.0] * len(prompts)


def encode_and_score_batch(
    runtime: Any,
    images: List[Image.Image],
    prompts: List[str],
    *,
    image_batch_size: Optional[int] = None,
    image_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode many images and score against a shared prompt set."""
    if not images or not prompts:
        return torch.zeros((len(images), len(prompts)), dtype=torch.float32)

    try:
        if runtime.bioclip_backend == "open_clip":
            embeddings = image_embeddings
            if embeddings is None:
                embeddings = ensure_open_clip_image_embeddings(runtime, images, image_batch_size=image_batch_size)
            return score_open_clip_with_image_embeddings(runtime, embeddings, prompts).detach().cpu()

        batch_size = max(1, int(image_batch_size or len(images)))
        outputs: List[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            outputs.append(_score_processor_batch(runtime, chunk, prompts).detach().cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.zeros((0, len(prompts)), dtype=torch.float32)
    except Exception as exc:
        logger.error("Batch encoding failed: %s", exc)
        return torch.zeros((len(images), len(prompts)), dtype=torch.float32)


def clip_score_labels_ensemble(
    runtime: Any,
    image: Image.Image,
    labels: List[str],
    *,
    label_type: str = "generic",
    num_prompts: Optional[int] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Score text labels against image with multiple prompts for robustness.

    Returns: (best_label, best_score, all_scores_dict)
    """
    if not labels:
        return "unknown", 0.0, {}

    templates = _limit_prompt_templates(runtime, label_type=label_type, num_prompts=num_prompts)
    label_ensemble_scores: Dict[str, List[float]] = {label: [] for label in labels}

    request = ClipScoreRequest(image=image)
    for template in templates:
        prompts = [template.format(term=label) for label in labels]
        scores = encode_and_score(runtime, request, prompts)
        for label, score in zip(labels, scores):
            label_ensemble_scores[label].append(float(score))

    label_avg_scores = _average_label_scores(label_ensemble_scores)
    best_label, best_score = _best_label_and_score(label_avg_scores)
    return best_label, best_score, label_avg_scores


def clip_score_labels_ensemble_batch(
    runtime: Any,
    images: List[Image.Image],
    labels: List[str],
    *,
    label_type: str = "generic",
    num_prompts: Optional[int] = None,
    image_batch_size: Optional[int] = None,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Batch ensemble scoring for many images against a shared label set."""
    if not images:
        return []
    if not labels:
        return [("unknown", 0.0, {}) for _ in images]

    templates = _limit_prompt_templates(runtime, label_type=label_type, num_prompts=num_prompts)

    per_image_scores: List[Dict[str, List[float]]] = [{label: [] for label in labels} for _ in images]
    shared_image_embeddings: Optional[torch.Tensor] = None
    if runtime.bioclip_backend == "open_clip":
        shared_image_embeddings = ensure_open_clip_image_embeddings(
            runtime,
            images,
            image_batch_size=image_batch_size,
        )

    for template in templates:
        prompts = [template.format(term=label) for label in labels]
        batch_scores = encode_and_score_batch(
            runtime,
            images,
            prompts,
            image_batch_size=image_batch_size,
            image_embeddings=shared_image_embeddings,
        )
        for image_index, row in enumerate(batch_scores.tolist()):
            for label, score in zip(labels, row):
                per_image_scores[image_index][label].append(float(score))

    results: List[Tuple[str, float, Dict[str, float]]] = []
    for label_score_lists in per_image_scores:
        label_avg_scores = _average_label_scores(label_score_lists)
        best_label, best_score = _best_label_and_score(label_avg_scores)
        results.append((best_label, best_score, label_avg_scores))
    return results


def _log_open_set_debug(
    *,
    label_type: str,
    labels: List[str],
    known_probs: torch.Tensor,
    known_class_count: int,
    unknown_confidence: float,
    class_index: int,
    confidence: float,
    second_confidence: torch.Tensor,
    margin: float,
    open_set_min_confidence: float,
    open_set_margin: float,
    rejection_reasons: List[str],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    debug_info = {
        "label_type": label_type,
        "known_labels": labels,
        "known_class_probabilities": {
            labels[i]: float(known_probs[0, i].item()) if i < len(labels) else None
            for i in range(known_class_count)
        },
        "unknown_probability": unknown_confidence,
        "best_known_label": labels[class_index] if class_index < len(labels) else "unknown",
        "best_known_confidence": confidence,
        "second_known_confidence": float(second_confidence.item()),
        "margin_best_vs_second": margin,
        "threshold_min_confidence": open_set_min_confidence,
        "threshold_margin": open_set_margin,
        "rejection_reasons": list(rejection_reasons),
    }
    status = "REJECTED as unknown" if rejection_reasons else f"ACCEPTED {debug_info['best_known_label']}"
    logger.debug("[OPEN-SET] %s: %s", status, json.dumps(debug_info, indent=2, default=str))


def clip_score_labels(
    runtime: Any,
    image: Image.Image,
    labels: List[str],
    *,
    label_type: str = "generic",
) -> Tuple[str, float]:
    """Score text labels against image using CLIP/BioCLIP."""
    if not labels:
        return "unknown", 0.0

    text_prompts, prompt_to_class = build_prompt_batch(
        labels=labels,
        label_type=label_type,
        vlm_config=runtime.vlm_config,
    )
    if not text_prompts:
        return "unknown", 0.0

    known_class_count = len(labels)
    use_open_set = bool(runtime.open_set_enabled and label_type == "crop")
    unknown_class_index = known_class_count
    if use_open_set:
        unknown_prompts = open_set_unknown_prompts(label_type=label_type, known_labels=labels)
        text_prompts.extend(unknown_prompts)
        prompt_to_class.extend([unknown_class_index] * len(unknown_prompts))
        class_count = known_class_count + 1
    else:
        class_count = known_class_count

    request = ClipScoreRequest(image=image)
    probabilities = _aggregate_prompt_probabilities(
        runtime,
        request=request,
        text_prompts=text_prompts,
        prompt_to_class=prompt_to_class,
        class_count=class_count,
    )

    if use_open_set:
        known_probs = probabilities[:, :known_class_count]
        unknown_prob = probabilities[:, unknown_class_index]
        best_confidence, best_index = torch.max(known_probs, dim=-1)

        if known_class_count > 1:
            topk_conf, _ = torch.topk(known_probs, k=2, dim=-1)
            second_confidence = topk_conf[:, 1]
        else:
            second_confidence = torch.zeros_like(best_confidence)

        class_index = int(best_index.item())
        confidence = float(best_confidence.item())
        unknown_confidence = float(unknown_prob.item())
        margin = confidence - float(second_confidence.item())

        rejection_reasons: List[str] = []
        if unknown_confidence >= confidence:
            rejection_reasons.append(
                f"unknown_confidence ({unknown_confidence:.4f}) >= confidence ({confidence:.4f})"
            )
        if confidence < runtime.open_set_min_confidence:
            rejection_reasons.append(
                f"confidence ({confidence:.4f}) < threshold ({runtime.open_set_min_confidence:.4f})"
            )
        if margin < runtime.open_set_margin:
            rejection_reasons.append(
                f"margin ({margin:.4f}) < threshold ({runtime.open_set_margin:.4f})"
            )

        _log_open_set_debug(
            label_type=label_type,
            labels=labels,
            known_probs=known_probs,
            known_class_count=known_class_count,
            unknown_confidence=unknown_confidence,
            class_index=class_index,
            confidence=confidence,
            second_confidence=second_confidence,
            margin=margin,
            open_set_min_confidence=runtime.open_set_min_confidence,
            open_set_margin=runtime.open_set_margin,
            rejection_reasons=rejection_reasons,
        )

        if rejection_reasons:
            return "unknown", max(unknown_confidence, confidence)
        label = labels[class_index] if class_index < len(labels) else "unknown"
        return label, confidence

    best_confidence, best_index = torch.max(probabilities, dim=-1)
    class_index = int(best_index.item())
    confidence = float(best_confidence.item())
    label = labels[class_index] if class_index < len(labels) else "unknown"
    return label, confidence
