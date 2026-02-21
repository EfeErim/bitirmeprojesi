#!/usr/bin/env python3
"""
SAM3 + BioCLIP-2.5 Pipeline (Primary)
Fallback to DINO + SAM2 + BioCLIP-2 on error

This module implements a unified VLM pipeline with SAM3 as the preferred
segmentation model and BioCLIP-2.5 for classification, with automatic
fallback to the original DINO+SAM2+BioCLIP-2 stack if needed.
"""

import torch
import logging
from typing import Dict, Optional, Any, List, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SAM3BioCLIP25Pipeline:
    """
    New pipeline: SAM3 (text-prompted segmentation) + BioCLIP-2.5 (improved classification)
    
    SAM3 advantages:
    - Text-prompted segmentation (no need for GroundingDINO)
    - Better instance segmentation quality
    - Can use generic prompts like "plant leaf"
    
    BioCLIP-2.5 advantages:
    - +5.7% accuracy over BioCLIP-2
    - Larger model (ViT-H/14 vs ViT-L/14)
    - Better on biological visual tasks
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.pipeline_type = 'sam3_bioclip25'
        
        # Config parsing
        router_config = config.get('router', {}) if isinstance(config.get('router'), dict) else {}
        self.vlm_config = router_config.get('vlm', {}) if isinstance(router_config, dict) else {}
        
        self.enabled = config.get('vlm_enabled', self.vlm_config.get('enabled', False))
        self.confidence_threshold = config.get('vlm_confidence_threshold', self.vlm_config.get('confidence_threshold', 0.7))
        self.max_detections = config.get('vlm_max_detections', self.vlm_config.get('max_detections', 10))
        self.open_set_enabled = config.get('vlm_open_set_enabled', self.vlm_config.get('open_set_enabled', True))
        self.open_set_min_confidence = float(
            config.get('vlm_open_set_min_confidence', self.vlm_config.get('open_set_min_confidence', 0.55))
        )
        self.open_set_margin = float(
            config.get('vlm_open_set_margin', self.vlm_config.get('open_set_margin', 0.10))
        )
        
        # Model IDs
        defaults = {
            'sam': 'facebook/sam3',  # SAM3 instead of sam2_b.pt
            'bioclip': 'imageomics/bioclip-2.5-vith14'  # BioCLIP-2.5 instead of 2
        }
        configured_ids = self.vlm_config.get('model_ids', {}) if isinstance(self.vlm_config.get('model_ids', {}), dict) else {}
        self.model_ids = {
            'sam': configured_ids.get('sam', defaults['sam']),
            'bioclip': configured_ids.get('bioclip', defaults['bioclip'])
        }
        
        # Crop/part labels (same as before)
        crop_mapping = router_config.get('crop_mapping', {}) if isinstance(router_config, dict) else {}
        self.use_dynamic_taxonomy = self.vlm_config.get('use_dynamic_taxonomy', False)
        self.taxonomy_path = self.vlm_config.get('taxonomy_path', 'config/plant_taxonomy.json')
        
        if self.use_dynamic_taxonomy:
            from src.router.vlm_pipeline import VLMPipeline
            self.crop_labels, self.part_labels = VLMPipeline._load_taxonomy(self.taxonomy_path)
        else:
            self.crop_labels = list(self.vlm_config.get('crop_labels', list(crop_mapping.keys())))
            parts_from_mapping = []
            for crop_data in crop_mapping.values() if isinstance(crop_mapping, dict) else []:
                if isinstance(crop_data, dict):
                    parts_from_mapping.extend(crop_data.get('parts', []))
            default_parts = sorted(set(parts_from_mapping))
            self.part_labels = list(self.vlm_config.get('part_labels', default_parts))
        
        # Model placeholders
        self.sam3 = None
        self.sam3_processor = None
        self.bioclip = None
        self.bioclip_processor = None
        self.models_loaded = False
        
        logger.info(f"SAM3BioCLIP25Pipeline initialized on {self.device}")
    
    def load_models(self):
        """Load SAM3 and BioCLIP-2.5 models."""
        logger.info("Loading SAM3 + BioCLIP-2.5 models...")
        
        if not self.enabled:
            logger.info("VLM pipeline is disabled; skipping model loading")
            self.models_loaded = False
            return
        
        try:
            # Load SAM3 (from transformers)
            from transformers import Sam3Processor, Sam3Model
            
            logger.info(f"Loading SAM3 from {self.model_ids['sam']}...")
            self.sam3_processor = Sam3Processor.from_pretrained(self.model_ids['sam'])
            self.sam3 = Sam3Model.from_pretrained(self.model_ids['sam'])
            self.sam3 = self.sam3.to(self.device)
            self.sam3.eval()
            logger.info("✅ SAM3 loaded")
            
            # Load BioCLIP-2.5 (from open_clip)
            import open_clip
            
            logger.info(f"Loading BioCLIP-2.5 from {self.model_ids['bioclip']}...")
            hub_model_id = f"hf-hub:{self.model_ids['bioclip']}"
            model, _, preprocess_val = open_clip.create_model_and_transforms(hub_model_id)
            tokenizer = open_clip.get_tokenizer(hub_model_id)
            
            self.bioclip = model.to(self.device)
            self.bioclip.eval()
            self.bioclip_processor = {
                'preprocess': preprocess_val,
                'tokenizer': tokenizer,
            }
            logger.info("✅ BioCLIP-2.5 loaded")
            
            self.models_loaded = True
            logger.info("✅ SAM3 + BioCLIP-2.5 models loaded successfully")
            
        except Exception as e:
            self.models_loaded = False
            logger.error(f"Failed to load SAM3 + BioCLIP-2.5: {e}")
            raise RuntimeError(f"SAM3+BioCLIP-2.5 initialization failed: {e}") from e
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready for inference."""
        if not self.enabled:
            return False
        return bool(
            self.models_loaded
            and self.sam3 is not None
            and self.sam3_processor is not None
            and self.bioclip is not None
            and self.bioclip_processor is not None
        )
    
    def analyze_image(
        self,
        image_tensor: torch.Tensor,
        confidence_threshold: float = 0.7,
        max_detections: int = 10,
        sam3_prompt: str = "plant leaf"
    ) -> Dict[str, Any]:
        """
        Analyze image using SAM3 + BioCLIP-2.5.
        
        Args:
            image_tensor: Preprocessed image tensor [C, H, W]
            confidence_threshold: Minimum confidence for detections
            max_detections: Maximum number of detections to return
            sam3_prompt: Text prompt for SAM3 segmentation
            
        Returns:
            Dictionary with analysis results
        """
        if not self.is_ready():
            logger.warning("Pipeline not ready")
            return {
                'detections': [],
                'image_size': tuple(image_tensor.shape),
                'processing_time_ms': 0.0
            }
        
        import time
        start_time = time.perf_counter()
        
        # Convert tensor to PIL
        pil_image = self._tensor_to_pil(image_tensor)
        
        # Run SAM3 with text prompt
        sam3_results = self._run_sam3(pil_image, prompt=sam3_prompt, threshold=confidence_threshold)
        masks = sam3_results.get('masks', [])
        boxes = sam3_results.get('boxes', [])
        scores = sam3_results.get('scores', [])
        
        if not masks:
            logger.warning(f"SAM3 found no instances with prompt '{sam3_prompt}'")
            return {
                'detections': [],
                'image_size': tuple(image_tensor.shape),
                'processing_time_ms': (time.perf_counter() - start_time) * 1000.0
            }
        
        # Classify each detected region with BioCLIP-2.5
        detections = []
        for i, (mask, box, score) in enumerate(zip(masks[:max_detections], boxes[:max_detections], 
                                                     scores[:max_detections])):
            if float(score) < confidence_threshold:
                continue
            
            # Extract ROI using the bounding box
            roi_image = self._extract_roi(pil_image, box.tolist() if torch.is_tensor(box) else box)
            
            # Classify crop and part
            crop_label, crop_conf = self._classify_with_bioclip(roi_image, 'crop')
            part_label, part_conf = self._classify_with_bioclip(roi_image, 'part')
            
            detections.append({
                'crop': crop_label,
                'part': part_label,
                'crop_confidence': crop_conf,
                'part_confidence': part_conf,
                'disease': None,
                'disease_confidence': 0.0,
                'bbox': box.tolist() if torch.is_tensor(box) else box,
                'mask': mask.tolist() if torch.is_tensor(mask) else mask,
                'sam3_score': float(score),
            })
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            'detections': detections,
            'image_size': tuple(image_tensor.shape),
            'processing_time_ms': elapsed_ms,
            'pipeline_type': self.pipeline_type,
            'sam3_instances': len(masks)
        }
    
    def _run_sam3(self, image: Image.Image, prompt: str, threshold: float = 0.7) -> Dict[str, Any]:
        """Run SAM3 instance segmentation with text prompt."""
        try:
            inputs = self.sam3_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.sam3(**inputs)
            
            # Post-process
            results = self.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            masks = results.get('masks', torch.tensor([]))
            boxes = results.get('boxes', torch.tensor([]))
            scores = results.get('scores', torch.tensor([]))
            
            return {
                'masks': masks,
                'boxes': boxes,
                'scores': scores
            }
        except Exception as e:
            logger.error(f"SAM3 inference failed: {e}")
            return {'masks': [], 'boxes': [], 'scores': []}
    
    def _classify_with_bioclip(self, image: Image.Image, label_type: str) -> Tuple[str, float]:
        """Classify image using BioCLIP-2.5."""
        if label_type == 'crop':
            labels = self.crop_labels
        elif label_type == 'part':
            labels = self.part_labels
        else:
            return 'unknown', 0.0
        
        if not labels:
            return 'unknown', 0.0
        
        try:
            preprocess = self.bioclip_processor['preprocess']
            tokenizer = self.bioclip_processor['tokenizer']
            
            # Build prompts (simplified vs original)
            prompts = [f"a photo of {label}" for label in labels]
            if self.open_set_enabled:
                prompts.append("an unknown plant")
            
            # Encode
            image_tensor = preprocess(image).unsqueeze(0).to(self.device)
            text_tokens = tokenizer(prompts).to(self.device)
            
            with torch.no_grad():
                image_embeds = self.bioclip.encode_image(image_tensor)
                text_embeds = self.bioclip.encode_text(text_tokens)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                logits = (image_embeds @ text_embeds.T) * 100.0  # CLIP logit scale
                probs = torch.softmax(logits, dim=-1)[0]
            
            if self.open_set_enabled and len(probs) > len(labels):
                unknown_prob = probs[-1].item()
                best_prob = probs[:len(labels)].max().item()
                if unknown_prob > best_prob or best_prob < self.open_set_min_confidence:
                    return 'unknown', max(unknown_prob, best_prob)
            
            best_idx = probs[:len(labels)].argmax().item()
            best_conf = probs[best_idx].item()
            label = labels[best_idx]
            
            return label, best_conf
            
        except Exception as e:
            logger.error(f"BioCLIP-2.5 classification failed: {e}")
            return 'unknown', 0.0
    
    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL image."""
        tensor = image_tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        
        tensor_min = float(tensor.min())
        tensor_max = float(tensor.max())
        
        if tensor_max <= 1.0 and tensor_min >= 0.0:
            normalized = tensor
        else:
            normalized = tensor.clone()
            if tensor_min < 0.0:
                normalized = (normalized + 1.0) / 2.0
            normalized = normalized.clamp(0.0, 1.0)
        
        uint8_img = (normalized * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(uint8_img)
    
    @staticmethod
    def _extract_roi(image: Image.Image, bbox: List[float], pad_ratio: float = 0.08) -> Image.Image:
        """Extract ROI from image using bounding box."""
        if not bbox or len(bbox) != 4:
            return image
        
        width, height = image.size
        x1, y1, x2, y2 = [float(v) for v in bbox]
        
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        pad_x = box_w * pad_ratio
        pad_y = box_h * pad_ratio
        
        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(width, int(x2 + pad_x))
        bottom = min(height, int(y2 + pad_y))
        
        if right <= left or bottom <= top:
            return image
        
        return image.crop((left, top, right, bottom))
