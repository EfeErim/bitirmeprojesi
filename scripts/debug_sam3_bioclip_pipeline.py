#!/usr/bin/env python3
"""
Debug SAM3 + BioCLIP pipeline outputs per ROI.

Purpose:
- Show what SAM3 segments
- Show what BioCLIP predicts for each ROI (crop + part, with top-k probabilities)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.vlm_pipeline import VLMPipeline  # noqa: E402

logger = logging.getLogger("debug_sam3_bioclip")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_json_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)
    except Exception:
        return float(default)


def _collect_class_probabilities(
    pipeline: VLMPipeline,
    image: Image.Image,
    labels: List[str],
    label_type: str,
    topk: int,
) -> Dict[str, Any]:
    if not labels:
        return {
            "best_label": "unknown",
            "best_confidence": 0.0,
            "topk": [],
            "unknown_probability": None,
            "known_probabilities": {},
        }

    text_prompts, prompt_to_class = pipeline._build_prompt_batch(labels, label_type=label_type)
    if not text_prompts:
        return {
            "best_label": "unknown",
            "best_confidence": 0.0,
            "topk": [],
            "unknown_probability": None,
            "known_probabilities": {},
        }

    known_count = len(labels)
    use_open_set = bool(pipeline.open_set_enabled and label_type == "crop")
    unknown_index = known_count

    if use_open_set:
        unknown_prompts = pipeline._open_set_unknown_prompts(label_type=label_type, known_labels=labels)
        text_prompts.extend(unknown_prompts)
        prompt_to_class.extend([unknown_index] * len(unknown_prompts))
        class_count = known_count + 1
    else:
        class_count = known_count

    if pipeline.bioclip_backend == "open_clip":
        preprocess = pipeline.bioclip_processor["preprocess"]
        tokenizer = pipeline.bioclip_processor["tokenizer"]
        image_tensor = preprocess(image).unsqueeze(0).to(pipeline.device)
        text_tokens = tokenizer(text_prompts).to(pipeline.device)

        with torch.no_grad():
            image_embeds = pipeline.bioclip.encode_image(image_tensor)
            text_embeds = pipeline.bioclip.encode_text(text_tokens)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            logit_scale = pipeline._get_clip_logit_scale(pipeline.bioclip)
            prompt_logits = (image_embeds @ text_embeds.T) * logit_scale
            class_logits = pipeline._aggregate_prompt_logits(prompt_logits, prompt_to_class, class_count)
            probabilities = torch.softmax(class_logits, dim=-1).squeeze(0)
    else:
        model_inputs = pipeline.bioclip_processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {k: v.to(pipeline.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = pipeline.bioclip(**model_inputs)
            if hasattr(outputs, "logits_per_image") and outputs.logits_per_image is not None:
                logits = outputs.logits_per_image
            elif hasattr(outputs, "image_embeds") and hasattr(outputs, "text_embeds"):
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                logit_scale = pipeline._get_clip_logit_scale(pipeline.bioclip)
                logits = (image_embeds @ text_embeds.T) * logit_scale
            else:
                raise RuntimeError("BioCLIP backend output missing logits_per_image and embeddings")
            class_logits = pipeline._aggregate_prompt_logits(logits, prompt_to_class, class_count)
            probabilities = torch.softmax(class_logits, dim=-1).squeeze(0)

    known_probs = probabilities[:known_count]
    known_prob_map = {
        label: float(known_probs[idx].item())
        for idx, label in enumerate(labels)
    }

    k = max(1, min(int(topk), known_count))
    top_values, top_indices = torch.topk(known_probs, k=k)
    top_entries = [
        {"label": labels[int(class_idx.item())], "probability": float(prob.item())}
        for prob, class_idx in zip(top_values, top_indices)
    ]

    best_value, best_index = torch.max(known_probs, dim=-1)
    best_label = labels[int(best_index.item())]
    best_conf = float(best_value.item())

    unknown_probability: Optional[float] = None
    if use_open_set:
        unknown_probability = float(probabilities[unknown_index].item())
        if known_count > 1:
            top2_vals, _ = torch.topk(known_probs, k=2, dim=-1)
            second_conf = float(top2_vals[1].item())
        else:
            second_conf = 0.0
        margin = best_conf - second_conf
        reject = (
            unknown_probability >= best_conf
            or best_conf < float(pipeline.open_set_min_confidence)
            or margin < float(pipeline.open_set_margin)
        )
        if reject:
            best_label = "unknown"
            best_conf = max(best_conf, unknown_probability)

    return {
        "best_label": best_label,
        "best_confidence": best_conf,
        "topk": top_entries,
        "unknown_probability": unknown_probability,
        "known_probabilities": known_prob_map,
    }


def _draw_sam_boxes(
    image: Image.Image,
    entries: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for item in entries:
        bbox = item["bbox"]
        idx = item["idx"]
        score = item["sam_score"]

        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text = f"{idx}: {score:.3f}"
        tx = int(max(0, x1 + 2))
        ty = int(max(0, y1 - 12))
        draw.text((tx, ty), text, fill="yellow")

    canvas.save(out_path)


def _safe_box_to_list(box: Any) -> Optional[List[float]]:
    try:
        if torch.is_tensor(box):
            box = box.detach().cpu().tolist()
        return [float(v) for v in box]
    except Exception:
        return None


def _count_raw_instances(masks: Any) -> int:
    if torch.is_tensor(masks):
        if masks.ndim == 0:
            return int(masks.numel() > 0)
        return int(masks.shape[0])
    if isinstance(masks, (list, tuple)):
        return len(masks)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug SAM3 + BioCLIP per-ROI predictions for a single image"
    )
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--config", default="config/colab.json", type=str, help="Path to config JSON")
    parser.add_argument("--out-dir", default="outputs/vlm_debug", type=str, help="Output directory")
    parser.add_argument("--device", default=None, type=str, help="Force device (e.g. cpu or cuda)")
    parser.add_argument("--sam-threshold", default=None, type=float, help="Override SAM3 threshold")
    parser.add_argument("--topk", default=10, type=int, help="Top-k class probabilities to report")
    return parser.parse_args()


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\nROI summary")
    print(f"{'idx':>4} {'sam':>8} {'crop':<24} {'c_conf':>8} {'part':<20} {'p_conf':>8}")
    print("-" * 78)
    for row in rows:
        idx = row["idx"]
        sam_score = row["sam_score"]
        crop_label = str(row["crop"]["best_label"])[:24]
        crop_conf = to_float(row["crop"]["best_confidence"])
        part_label = str(row["part"]["best_label"])[:20]
        part_conf = to_float(row["part"]["best_confidence"])
        print(f"{idx:>4d} {sam_score:>8.3f} {crop_label:<24} {crop_conf:>8.3f} {part_label:<20} {part_conf:>8.3f}")


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    config_path = Path(args.config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json_config(config_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing pipeline on device=%s", device)
    pipeline = VLMPipeline(config=config, device=device)
    pipeline.load_models()

    if not pipeline.is_ready():
        raise RuntimeError("VLM pipeline is not ready after load_models()")

    if args.sam_threshold is not None:
        chosen_threshold = float(args.sam_threshold)
    else:
        cfg_threshold = pipeline.vlm_config.get("sam3_mask_threshold", 0.60)
        chosen_threshold = float(cfg_threshold)
    chosen_threshold = clamp(chosen_threshold, 0.35, 0.95)

    pil_image, _ = pipeline._coerce_image_input(str(image_path))
    width, height = pil_image.size

    sam_results = pipeline._run_sam3(
        pil_image,
        prompt="plant leaf",
        threshold=chosen_threshold,
    )

    boxes = sam_results.get("boxes", [])
    scores = sam_results.get("scores", [])
    masks = sam_results.get("masks", [])
    total_raw_instances = _count_raw_instances(masks)

    sanitized_for_vis: List[Dict[str, Any]] = []
    roi_rows: List[Dict[str, Any]] = []

    kept_idx = 0
    for raw_idx, (box, score) in enumerate(zip(boxes, scores)):
        raw_bbox = _safe_box_to_list(box)
        if raw_bbox is None:
            continue

        sanitized_bbox = pipeline._sanitize_bbox(raw_bbox, width, height)
        if sanitized_bbox is None:
            continue

        sam_score = to_float(score)
        sanitized_for_vis.append(
            {
                "idx": raw_idx,
                "bbox": sanitized_bbox,
                "sam_score": sam_score,
            }
        )

        roi_image = pipeline._extract_roi(pil_image, sanitized_bbox)
        roi_filename = f"roi_{kept_idx:03d}.png"
        roi_path = out_dir / roi_filename
        roi_image.save(roi_path)

        crop_diag = _collect_class_probabilities(
            pipeline=pipeline,
            image=roi_image,
            labels=pipeline.crop_labels,
            label_type="crop",
            topk=args.topk,
        )
        part_diag = _collect_class_probabilities(
            pipeline=pipeline,
            image=roi_image,
            labels=pipeline.part_labels,
            label_type="part",
            topk=args.topk,
        )

        roi_rows.append(
            {
                "idx": kept_idx,
                "raw_index": raw_idx,
                "sam_score": sam_score,
                "bbox": [float(v) for v in sanitized_bbox],
                "roi_path": str(roi_path),
                "crop": crop_diag,
                "part": part_diag,
            }
        )
        kept_idx += 1

    boxes_out = out_dir / "sam_boxes.png"
    _draw_sam_boxes(pil_image, sanitized_for_vis, boxes_out)

    report = {
        "image_path": str(image_path),
        "sam_threshold": chosen_threshold,
        "total_raw_instances": int(total_raw_instances),
        "kept_instances": int(len(roi_rows)),
        "sam_boxes_visualization": str(boxes_out),
        "per_roi_diagnostics": roi_rows,
    }

    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print_summary_table(roi_rows)
    print(f"\nSaved report: {report_path}")
    print(f"Saved SAM visualization: {boxes_out}")
    print(f"Saved ROI crops: {len(roi_rows)} file(s) in {out_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.exception("Debug pipeline failed: %s", exc)
        raise SystemExit(1)
