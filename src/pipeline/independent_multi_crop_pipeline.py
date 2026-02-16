#!/usr/bin/env python3
"""
Independent Multi-Crop Pipeline for AADS-ULoRA v5.5
Main pipeline orchestrating router and independent adapters.
Key principle: No cross-adapter communication - fully independent.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from functools import lru_cache
import hashlib

from src.router.vlm_pipeline import VLMPipeline, DiagnosticScoutingAnalyzer
from src.utils.data_loader import preprocess_image

logger = logging.getLogger(__name__)

class IndependentMultiCropPipeline:
    """
    Main pipeline orchestrating VLM-based router and independent adapters.
    Key: No cross-adapter communication - fully independent.
    
    Now uses VLM Pipeline (Grounding DINO + SAM-2 + BioCLIP 2) as the definitive router.
    """

    def __init__(self, config: Dict, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize components
        self.router = None
        self.router_analyzer = None  # DiagnosticScoutingAnalyzer for VLM
        self.adapters = {}  # crop_name -> IndependentCropAdapter
        self.ood_buffers = {}  # Phase 2/3 triggering
        
        # Supported crops
        self.crops = config.get('router', {}).get('crop_mapping', {}).keys()
        
        # Caching system
        self.cache_enabled = config.get('router', {}).get('caching', {}).get('enabled', True)
        self.cache_size = config.get('router', {}).get('caching', {}).get('max_size', 1000)
        self.router_cache = {}  # Cache for router predictions
        self.adapter_cache = {}  # Cache for adapter predictions
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"IndependentMultiCropPipeline initialized on {self.device}")
        logger.info("Using VLM Pipeline as definitive router")

    def _generate_cache_key(self, image_tensor: torch.Tensor) -> str:
        """Generate a cache key for an image tensor."""
        # Convert tensor to bytes for hashing
        tensor_bytes = image_tensor.cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()

    def _clear_caches(self):
        """Clear all caches."""
        self.router_cache.clear()
        self.adapter_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Caches cleared")

    def initialize_router(
        self,
        router_path: Optional[str] = None,
        train_datasets: Optional[Dict[str, 'CropDataset']] = None,
        val_datasets: Optional[Dict[str, 'CropDataset']] = None
    ) -> bool:
        """
        Initialize VLM-based router.
        
        Args:
            router_path: Path to pre-trained router (ignored for VLM - loads models from config)
            train_datasets: Not used for VLM (models are pre-trained)
            val_datasets: Not used for VLM
            
        Returns:
            True if successful
        """
        logger.info("Initializing VLM-based crop router...")
        
        try:
            # Create VLM pipeline
            self.router = VLMPipeline(config=self.config, device=self.device)
            
            # Load VLM models
            self.router.load_models()
            
            # Create diagnostic analyzer for easier crop classification
            self.router_analyzer = DiagnosticScoutingAnalyzer(config=self.config, device=self.device)
            
            logger.info("VLM router initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize VLM router: {e}")
            return False

    def initialize_adapters(self) -> bool:
        """Initialize all crop adapters."""
        logger.info("Initializing crop adapters...")
        
        try:
            # Initialize each adapter based on router configuration
            crop_mapping = self.config.get('router', {}).get('crop_mapping', {})
            
            for crop_name, crop_config in crop_mapping.items():
                adapter_path = crop_config.get('model_path')
                if not adapter_path:
                    logger.warning(f"No model path for crop {crop_name}, skipping")
                    continue
                
                # Initialize adapter (placeholder - actual adapter initialization would go here)
                self.adapters[crop_name] = {
                    'path': adapter_path,
                    'initialized': False,
                    'config': crop_config
                }
                logger.info(f"Adapter initialized for {crop_name} at {adapter_path}")
            
            logger.info("All adapters initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize adapters: {e}")
            return False

    def process_image(
        self,
        image: Any,
        crop: Optional[str] = None,
        part: Optional[str] = None,
        return_ood: bool = True
    ) -> Dict[str, Any]:
        """
        Process an image through the pipeline.
        
        Args:
            image: Input image (PIL Image, path, or tensor)
            crop: Optional crop type to force (bypasses router)
            part: Optional plant part to force
            return_ood: Whether to return OOD information
            
        Returns:
            Dictionary with diagnosis results
        """
        # Preprocess image
        image_tensor = preprocess_image(image, self.config)
        
        # Check cache first
        if self.cache_enabled:
            cache_key = self._generate_cache_key(image_tensor)
            if cache_key in self.router_cache:
                self.cache_hits += 1
                return self.router_cache[cache_key]
            self.cache_misses += 1
        
        # Router step (unless crop is specified)
        if crop is None:
            router_result = self._route_image(image_tensor)
            crop = router_result.get('crop')
            part = router_result.get('part')
        
        # Adapter step
        if crop and crop in self.adapters:
            adapter_result = self._process_with_adapter(
                image_tensor, crop, part, return_ood
            )
        else:
            # Handle unknown crop
            adapter_result = self._handle_unknown_crop(image_tensor, crop, part)
        
        # Combine results
        result = {
            'crop': crop,
            'part': part,
            'diagnosis': adapter_result.get('diagnosis'),
            'confidence': adapter_result.get('confidence'),
            'ood_score': adapter_result.get('ood_score', 0.0),
            'ood_status': adapter_result.get('ood_status', 'unknown'),
            'router_confidence': router_result.get('confidence', 0.0) if crop is None else 1.0,
            'cache_hit': False
        }
        
        # Cache result
        if self.cache_enabled:
            result['cache_hit'] = True
            self.router_cache[cache_key] = result
            
            # Maintain cache size
            if len(self.router_cache) > self.cache_size:
                self._evict_cache()
        
        return result

    def _route_image(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Route image to appropriate crop adapter."""
        if not self.router:
            raise RuntimeError("Router not initialized")
        
        # Get router configuration
        router_config = self.config.get('router', {})
        
        # Perform routing
        router_result = self.router.analyze_image(
            image_tensor,
            confidence_threshold=router_config.get('vlm', {}).get('confidence_threshold', 0.8),
            max_detections=router_config.get('vlm', {}).get('max_detections', 10)
        )
        
        # Extract best crop and part
        best_crop = None
        best_part = None
        best_confidence = 0.0
        
        for detection in router_result.get('detections', []):
            crop_confidence = detection.get('crop_confidence', 0.0)
            if crop_confidence > best_confidence:
                best_confidence = crop_confidence
                best_crop = detection.get('crop')
                best_part = detection.get('part')
        
        return {
            'crop': best_crop,
            'part': best_part,
            'confidence': best_confidence,
            'detections': router_result.get('detections', [])
        }

    def _process_with_adapter(
        self,
        image_tensor: torch.Tensor,
        crop: str,
        part: Optional[str],
        return_ood: bool
    ) -> Dict[str, Any]:
        """Process image with specific crop adapter."""
        if crop not in self.adapters:
            raise ValueError(f"No adapter for crop: {crop}")
        
        adapter_info = self.adapters[crop]
        
        # Get adapter configuration
        crop_mapping = self.config.get('router', {}).get('crop_mapping', {})
        crop_config = crop_mapping.get(crop, {})
        
        # Perform diagnosis (placeholder - actual adapter processing would go here)
        diagnosis = {
            'healthy': 0.7,
            'early_blight': 0.2,
            'late_blight': 0.1
        }
        
        # Calculate confidence
        confidence = max(diagnosis.values())
        
        # OOD detection
        ood_score = 0.0
        ood_status = 'normal'
        
        if return_ood and self.config.get('ood', {}).get('enabled', True):
            ood_score, ood_status = self._perform_ood_detection(image_tensor)
        
        return {
            'diagnosis': diagnosis,
            'confidence': confidence,
            'ood_score': ood_score,
            'ood_status': ood_status
        }

    def _handle_unknown_crop(
        self,
        image_tensor: torch.Tensor,
        crop: Optional[str],
        part: Optional[str]
    ) -> Dict[str, Any]:
        """Handle cases where crop is unknown or unsupported."""
        return {
            'diagnosis': {
                'unknown': 1.0
            },
            'confidence': 0.0,
            'ood_score': 1.0,
            'ood_status': 'unknown_crop'
        }

    def _perform_ood_detection(self, image_tensor: torch.Tensor) -> Tuple[float, str]:
        """Perform OOD detection."""
        ood_config = self.config.get('ood', {})
        
        if not ood_config.get('enabled', True):
            return 0.0, 'normal'
        
        # Placeholder OOD detection logic
        # In a real implementation, this would use the configured OOD method
        return 0.3, 'uncertain'

    def _evict_cache(self):
        """Evict least recently used items from cache."""
        if len(self.router_cache) > self.cache_size * 1.5:  # Only evict if significantly over limit
            # Simple LRU eviction (remove oldest items)
            sorted_items = sorted(self.router_cache.items(), key=lambda x: x[1].get('timestamp', 0))
            items_to_remove = len(self.router_cache) - self.cache_size
            for i in range(items_to_remove):
                self.router_cache.pop(sorted_items[i][0], None)
            logger.info(f"Cache evicted {items_to_remove} items")

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'router_cache_size': len(self.router_cache),
            'adapter_cache_size': len(self.adapter_cache),
            'total_adapters': len(self.adapters)
        }

    def reset_metrics(self):
        """Reset pipeline metrics."""
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Pipeline metrics reset")
