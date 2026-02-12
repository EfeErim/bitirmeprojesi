#!/usr/bin/env python3
"""
Enhanced Crop Router for AADS-ULoRA v5.5
Routes to specific (crop, part) adapters based on dual classification.

Architecture:
1. Classify crop type (tomato, pepper, corn)
2. Classify plant part (leaf, stem, fruit, etc.)
3. Combine to determine adapter key: f"{crop}_{part}"
4. Load appropriate adapter for that specific crop+part
5. Adapter identifies disease

Based on research from "Researching Plant Detection Methods.pdf" (2026),
uses multi-stage VLM pipeline (Scenario B) for diagnostic scouting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
from functools import lru_cache
import json
import time

logger = logging.getLogger(__name__)

class EnhancedCropRouter:
    """
    Enhanced router that classifies both crop and plant part,
    then routes to appropriate (crop, part) adapter.
    
    Only Scenario B (Diagnostic Scouting) is supported, using the
    multi-stage VLM pipeline: Grounding DINO + SAM-2 + BioCLIP 2.
    
    Args:
        crops: List of supported crop names
        parts: List of supported plant parts (leaf, stem, fruit, etc.)
        config: Configuration dictionary
        device: Device for inference
    """
    
    def __init__(
        self,
        crops: List[str],
        parts: List[str],
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.crops = crops
        self.parts = parts
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize dual classifiers for crop and part
        self._init_routing_models(config)
        
        # Adapter registry: {(crop, part): adapter_instance}
        self.adapters = {}
        
        # Cache
        self.routing_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Metrics
        self.routing_metrics = {
            'adapter_hits': {},
            'adapter_misses': {},
            'total_routes': 0
        }
        
        logger.info(f"EnhancedCropRouter initialized on {self.device}")
        logger.info(f"Supported crops: {crops}")
        logger.info(f"Supported parts: {parts}")
    
    def _init_routing_models(self, config: Dict[str, Any]):
        """Initialize dual classifiers for crop and part."""
        router_config = config.get('crop_router', {})
        model_name = router_config.get('model_name', 'facebook/dinov3-base')
        
        logger.info(f"Loading routing backbone: {model_name}")
        
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config_model = AutoConfig.from_pretrained(model_name)
        
        # Get hidden size
        if hasattr(self.config_model, 'hidden_size'):
            self.hidden_size = self.config_model.hidden_size
        elif hasattr(self.config_model, 'dim'):
            self.hidden_size = self.config_model.dim
        else:
            raise ValueError(f"Cannot determine hidden size")
        
        # Two classification heads
        self.crop_classifier = nn.Linear(self.hidden_size, len(self.crops)).to(self.device)
        self.part_classifier = nn.Linear(self.hidden_size, len(self.parts)).to(self.device)
        
        self.backbone = self.backbone.to(self.device)
        
        logger.info(f"Crop classifier: {len(self.crops)} classes")
        logger.info(f"Part classifier: {len(self.parts)} classes")
    
    def register_adapter(
        self,
        crop: str,
        part: str,
        adapter_path: str,
        config: Optional[Dict] = None
    ) -> bool:
        """
        Register an adapter for a specific (crop, part) combination.
        
        Args:
            crop: Crop name (e.g., 'tomato')
            part: Plant part (e.g., 'leaf', 'fruit', 'stem')
            adapter_path: Path to trained adapter
            config: Optional adapter configuration
            
        Returns:
            True if successful
        """
        if crop not in self.crops:
            logger.error(f"Unsupported crop: {crop}")
            return False
        
        if part not in self.parts:
            logger.error(f"Unsupported part: {part}")
            return False
        
        adapter_key = (crop, part)
        
        try:
            from src.adapter.independent_crop_adapter import IndependentCropAdapter
            
            adapter = IndependentCropAdapter(
                crop_name=f"{crop}_{part}",
                device=self.device
            )
            adapter.load_adapter(adapter_path)
            
            self.adapters[adapter_key] = adapter
            
            # Initialize metrics for this adapter
            self.routing_metrics['adapter_hits'][adapter_key] = 0
            self.routing_metrics['adapter_misses'][adapter_key] = 0
            
            logger.info(f"Registered adapter for {crop} - {part}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register adapter for {crop}_{part}: {e}")
            return False
    
    def route(
        self,
        image: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, float, float, Dict[str, Any]]:
        """
        Route image to appropriate (crop, part) adapter.
        
        Args:
            image: Preprocessed image tensor (batch size 1)
            metadata: Optional metadata (ignored for now - only Scenario B)
            
        Returns:
            Tuple: (crop, part, crop_confidence, part_confidence, routing_info)
        """
        # Check cache
        cache_key = self._generate_cache_key(image)
        if cache_key in self.routing_cache:
            self.cache_hits += 1
            return self.routing_cache[cache_key]
        
        self.cache_misses += 1
        
        # Classify crop and part
        crop, crop_confidence = self._classify_crop(image)
        part, part_confidence = self._classify_part(image)
        
        # Check if adapter exists
        adapter_key = (crop, part)
        if adapter_key not in self.adapters:
            # Try fallback: use any adapter for this crop (any part)
            fallback_adapter = self._find_fallback_adapter(crop)
            if fallback_adapter:
                logger.warning(f"No adapter for {crop}_{part}, using fallback: {fallback_adapter}")
                part = fallback_adapter[1]
                adapter_key = fallback_adapter
            else:
                # No adapter available for this crop at all
                error_info = {
                    'status': 'error',
                    'message': f'No adapter available for {crop} - {part}',
                    'crop': crop,
                    'part': part,
                    'crop_confidence': crop_confidence,
                    'part_confidence': part_confidence,
                    'scenario': 'diagnostic_scouting'
                }
                if len(self.routing_cache) < self.config.get('routing_cache_size', 1000):
                    self.routing_cache[cache_key] = (None, None, 0, 0, error_info)
                return None, None, 0, 0, error_info
        
        # Update adapter hit metrics
        self.routing_metrics['adapter_hits'][adapter_key] += 1
        self.routing_metrics['total_routes'] += 1
        
        routing_info = {
            'status': 'success',
            'crop': crop,
            'part': part,
            'crop_confidence': crop_confidence,
            'part_confidence': part_confidence,
            'adapter_key': adapter_key,
            'scenario': 'diagnostic_scouting',  # Only Scenario B supported
            'fallback_used': adapter_key != (crop, part)
        }
        
        result = (crop, part, crop_confidence, part_confidence, routing_info)
        
        # Cache
        if len(self.routing_cache) < self.config.get('routing_cache_size', 1000):
            self.routing_cache[cache_key] = result
        
        return result
    
    def _classify_crop(self, image: torch.Tensor) -> Tuple[str, float]:
        """Classify crop type."""
        self.backbone.eval()
        self.crop_classifier.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.backbone(image)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.crop_classifier(pooled_output)
            
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            crop = self.crops[predicted_idx]
            return crop, confidence
    
    def _classify_part(self, image: torch.Tensor) -> Tuple[str, float]:
        """Classify plant part."""
        self.backbone.eval()
        self.part_classifier.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.backbone(image)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.part_classifier(pooled_output)
            
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            part = self.parts[predicted_idx]
            return part, confidence
    
    def _find_fallback_adapter(self, crop: str) -> Optional[Tuple[str, str]]:
        """Find any available adapter for this crop (any part)."""
        for (c, p) in self.adapters.keys():
            if c == crop:
                return (c, p)
        return None
    
    def _generate_cache_key(
        self,
        image: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key."""
        import hashlib
        tensor_bytes = image.cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()[:32]
    
    def get_adapter(self, crop: str, part: str) -> Optional[Any]:
        """Get adapter for specific (crop, part) combination."""
        return self.adapters.get((crop, part))
    
    def list_available_adapters(self) -> List[Tuple[str, str]]:
        """List all registered (crop, part) adapters."""
        return list(self.adapters.keys())
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_routes = self.routing_metrics['total_routes']
        
        # Adapter hit rates
        adapter_hit_rates = {}
        for key in self.adapters.keys():
            hits = self.routing_metrics['adapter_hits'].get(key, 0)
            adapter_hit_rates[str(key)] = hits / total_routes if total_routes > 0 else 0.0
        
        return {
            **self.routing_metrics,
            'total_routes': total_routes,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
            'adapter_hit_rates': adapter_hit_rates,
            'registered_adapters': self.list_available_adapters()
        }
    
    def clear_cache(self):
        """Clear routing cache."""
        self.routing_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Routing cache cleared")
    
    def save_router_state(self, save_dir: str):
        """Save router state."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'crop_classifier_state_dict': self.crop_classifier.state_dict(),
            'part_classifier_state_dict': self.part_classifier.state_dict(),
            'crops': self.crops,
            'parts': self.parts,
            'hidden_size': self.hidden_size,
            'config': self.config
        }, save_path / 'enhanced_router.pt')
        
        with open(save_path / 'routing_metrics.json', 'w') as f:
            json.dump(self.get_routing_metrics(), f, indent=2)
        
        logger.info(f"Router state saved to {save_path}")
    
    def load_router_state(self, load_dir: str):
        """Load router state."""
        load_path = Path(load_dir)
        checkpoint = torch.load(load_path / 'enhanced_router.pt', map_location=self.device)
        
        self.crop_classifier.load_state_dict(checkpoint['crop_classifier_state_dict'])
        self.part_classifier.load_state_dict(checkpoint['part_classifier_state_dict'])
        self.crops = checkpoint['crops']
        self.parts = checkpoint['parts']
        self.hidden_size = checkpoint['hidden_size']
        self.config = checkpoint['config']
        
        logger.info(f"Router state loaded from {load_path}")

def create_enhanced_router_from_config(
    config_path: str,
    device: str = 'cuda'
) -> EnhancedCropRouter:
    """Create router from configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    crops = config['data']['crops']
    # Extract parts from class names or use default
    # In practice, this would be derived from the dataset structure
    parts = ['leaf', 'fruit', 'stem', 'root', 'flower']
    
    router = EnhancedCropRouter(
        crops=crops,
        parts=parts,
        config=config,
        device=device
    )
    
    return router

if __name__ == "__main__":
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/adapter_spec_v55.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_image', type=str)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create router
    router = create_enhanced_router_from_config(args.config, args.device)
    
    # Test routing
    if args.test_image:
        from src.utils.data_loader import preprocess_image
        from PIL import Image
        
        image = Image.open(args.test_image).convert('RGB')
        img_tensor = preprocess_image(image).unsqueeze(0)
        
        # Route
        crop, part, crop_conf, part_conf, info = router.route(img_tensor)
        
        print(f"\nRouting Result:")
        print(f"  Crop: {crop} (conf: {crop_conf:.4f})")
        print(f"  Part: {part} (conf: {part_conf:.4f})")
        print(f"  Adapter key: {info['adapter_key']}")
        print(f"  Scenario: {info['scenario']}")
        
        # Check if adapter exists
        adapter = router.get_adapter(crop, part)
        if adapter:
            print(f"  Adapter found: Yes")
        else:
            print(f"  Adapter found: No (need to register)")
    
    # Print metrics
    print("\nRouting Metrics:")
    print(json.dumps(router.get_routing_metrics(), indent=2))
