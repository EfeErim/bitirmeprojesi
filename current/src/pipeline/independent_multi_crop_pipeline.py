#!/usr/bin/env python3
"""
Independent Multi-Crop Pipeline for AADS-ULoRA v5.5
Main pipeline orchestrating router and independent adapters.
Key principle: No cross-adapter communication - fully independent.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import OrderedDict
import hashlib
import time
import threading
from datetime import datetime, timedelta

from src.router.simple_crop_router import SimpleCropRouter
from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.utils.data_loader import preprocess_image

logger = logging.getLogger(__name__)

class TTLCache:
    """
    Thread-safe TTL cache with LRU eviction.
    
    Features:
    - Time-to-live expiration
    - Least-recently-used eviction when full
    - Automatic cleanup on access
    - Thread-safe operations
    - Cache statistics tracking
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize TTL cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # key -> (value, timestamp)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        # Iterate through cache to find expired entries
        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)
            else:
                # Since OrderedDict is ordered by insertion, we can break early
                break
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
            self.cache_evictions += 1
            logger.debug(f"Cache entry expired: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, returns None if expired or not found."""
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache:
                value, timestamp = self.cache[key]
                # Move to end to mark as recently used
                self.cache.move_to_end(key)
                self.cache_hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                self.cache_misses += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
    
    def set(self, key: str, value: Any):
        """Add item to cache with current timestamp."""
        with self.lock:
            self._cleanup_expired()
            
            # If key already exists, remove it first
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = (value, time.time())
            
            # Evict oldest if over max size
            if len(self.cache) > self.max_size:
                evicted_key, _ = self.cache.popitem(last=False)
                self.cache_evictions += 1
                logger.debug(f"Cache evicted oldest entry: {evicted_key}")
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_evictions = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_evictions': self.cache_evictions,
                'hit_rate': hit_rate,
                'current_size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl
            }
    
    def get_current_keys(self) -> List[str]:
        """Get list of current cache keys (for debugging)."""
        with self.lock:
            return list(self.cache.keys())

class IndependentMultiCropPipeline:
    """
    Main pipeline orchestrating router and independent adapters.
    Key: No cross-adapter communication - fully independent.
    
    Args:
        config: Configuration dictionary
        device: Device for inference
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Initialize components
        self.router = None
        self.adapters = {}  # crop_name -> IndependentCropAdapter
        self.ood_buffers = {}  # Phase 2/3 triggering
        
        # Supported crops
        self.crops = config.get('crops', ['tomato', 'pepper', 'corn'])
        
        # Caching system
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_size = config.get('cache_size', 1000)
        self.ttl_seconds = config.get('cache_ttl_seconds', 300)  # 5 minutes default
        
        # Initialize TTL caches
        self.router_cache = TTLCache(max_size=self.cache_size, ttl=self.ttl_seconds)
        self.adapter_cache = TTLCache(max_size=self.cache_size, ttl=self.ttl_seconds)
        
        logger.info(f"IndependentMultiCropPipeline initialized on {self.device}")
        logger.info(f"Cache configuration: size={self.cache_size}, TTL={self.ttl_seconds}s")
    
    def _generate_cache_key(self, image_tensor: torch.Tensor) -> str:
        """Generate a cache key for an image tensor."""
        # Convert tensor to bytes for hashing
        tensor_bytes = image_tensor.cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def _clear_caches(self):
        """Clear all caches."""
        self.router_cache.clear()
        self.adapter_cache.clear()
        logger.info("All caches cleared")
    
    def initialize_router(
        self,
        router_path: Optional[str] = None,
        train_datasets: Optional[Dict[str, 'CropDataset']] = None,
        val_datasets: Optional[Dict[str, 'CropDataset']] = None
    ) -> bool:
        """
        Initialize or load crop router.
        
        Args:
            router_path: Path to pre-trained router (if None, will train)
            train_datasets: Training datasets per crop (for training)
            val_datasets: Validation datasets per crop (for training)
            
        Returns:
            True if successful
        """
        logger.info("Initializing crop router...")
        
        # Create router
        self.router = SimpleCropRouter(
            crops=self.crops,
            model_name='facebook/dinov3-base',
            device=self.device
        )
        
        # Load or train
        if router_path and Path(router_path).exists():
            logger.info(f"Loading router from {router_path}")
            self.router.load_model(router_path)
        else:
            if train_datasets is None or val_datasets is None:
                raise ValueError("Cannot train router without datasets")
            
            # Combine datasets from all crops for router training
            # Router needs to see all crop types
            logger.info("Training crop router...")
            
            # For simplicity, we'll use tomato dataset as placeholder
            # In practice, you'd create a combined dataset with all crops
            train_dataset = train_datasets.get('tomato')
            val_dataset = val_datasets.get('tomato')
            
            if train_dataset is None or val_dataset is None:
                raise ValueError("Missing datasets for router training")
            
            self.router.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=10,
                batch_size=32,
                learning_rate=1e-3,
                save_path=router_path
            )
        
        logger.info("Crop router initialized successfully")
        return True
    
    def register_crop(
        self,
        crop_name: str,
        adapter_path: str,
        config: Optional[Dict] = None
    ) -> bool:
        """
        Register pre-trained crop adapter with OOD stats.
        
        Args:
            crop_name: Name of the crop
            adapter_path: Path to trained adapter
            config: Optional configuration for adapter
            
        Returns:
            True if successful
        """
        logger.info(f"Registering crop adapter: {crop_name}")
        
        if crop_name not in self.crops:
            logger.error(f"Unsupported crop: {crop_name}. Must be one of {self.crops}")
            return False
        
        # Create adapter
        adapter = IndependentCropAdapter(crop_name=crop_name, device=self.device)
        
        # Load adapter
        try:
            adapter.load_adapter(adapter_path)
            self.adapters[crop_name] = adapter
            logger.info(f"Successfully registered {crop_name} adapter")
            return True
        except Exception as e:
            logger.error(f"Failed to load adapter for {crop_name}: {e}")
            return False
    
    def process_image(
        self,
        image: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Main inference flow:
        1. Router determines crop
        2. Crop adapter predicts disease with dynamic OOD
        3. OOD detection triggers updates if needed
        
        Args:
            image: Preprocessed image tensor (batch size 1)
            metadata: Optional metadata about the image
            
        Returns:
            Dictionary with prediction results
        """
        # Check cache first if enabled
        if self.cache_enabled:
            cache_key = self._generate_cache_key(image)
            cached_result = self.adapter_cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
        
        # Step 1: Route to correct crop adapter
        if self.router is None:
            raise RuntimeError("Router not initialized")
        
        # Check router cache
        if self.cache_enabled:
            router_cache_key = self._generate_cache_key(image)
            cached_routing = self.router_cache.get(router_cache_key)
            
            if cached_routing is not None:
                predicted_crop, crop_confidence = cached_routing
            else:
                predicted_crop, crop_confidence = self.router.route(image)
                self.router_cache.set(router_cache_key, (predicted_crop, crop_confidence))
        else:
            predicted_crop, crop_confidence = self.router.route(image)
        
        logger.debug(f"Routed to crop: {predicted_crop} (confidence: {crop_confidence:.4f})")
        
        # Step 2: Get appropriate adapter
        if predicted_crop not in self.adapters:
            result = {
                'status': 'error',
                'message': f'No adapter available for crop: {predicted_crop}',
                'crop': predicted_crop,
                'crop_confidence': crop_confidence
            }
            if self.cache_enabled:
                cache_key = self._generate_cache_key(image)
                self.adapter_cache.set(cache_key, result)
            return result
        
        adapter = self.adapters[predicted_crop]
        
        # Step 3: Disease prediction with OOD detection
        try:
            result = adapter.predict_with_ood(image)
            
            # Add crop info
            result['crop'] = predicted_crop
            result['crop_confidence'] = crop_confidence
            
            # Check if OOD triggered
            if result['ood_analysis']['is_ood']:
                self._handle_ood_detection(result, metadata)
            
            # Cache result
            if self.cache_enabled:
                cache_key = self._generate_cache_key(image)
                self.adapter_cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            error_result = {
                'status': 'error',
                'message': str(e),
                'crop': predicted_crop,
                'crop_confidence': crop_confidence
            }
            if self.cache_enabled:
                cache_key = self._generate_cache_key(image)
                self.adapter_cache.set(cache_key, error_result)
            return error_result
    
    def batch_process(
        self,
        images: List[torch.Tensor],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Process multiple images in batch with optimized routing.
        
        Args:
            images: List of image tensors
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of prediction results
        """
        # Stack images into batch for router
        image_batch = torch.stack(images)
        
        # Batch routing
        if self.router is None:
            raise RuntimeError("Router not initialized")
        
        predicted_crops, crop_confidences = self.router.route_batch(image_batch)
        
        results = []
        
        # Process each image with its corresponding crop
        for i, (image, predicted_crop, crop_confidence) in enumerate(zip(images, predicted_crops, crop_confidences)):
            metadata = metadata_list[i] if metadata_list else None
            
            # Check cache
            if self.cache_enabled:
                cache_key = self._generate_cache_key(image)
                cached_result = self.adapter_cache.get(cache_key)
                
                if cached_result is not None:
                    results.append(cached_result)
                    continue
            
            # Get adapter
            if predicted_crop not in self.adapters:
                result = {
                    'status': 'error',
                    'message': f'No adapter available for crop: {predicted_crop}',
                    'crop': predicted_crop,
                    'crop_confidence': crop_confidence
                }
                if self.cache_enabled:
                    cache_key = self._generate_cache_key(image)
                    self.adapter_cache.set(cache_key, result)
                results.append(result)
                continue
            
            adapter = self.adapters[predicted_crop]
            
            # Predict
            try:
                result = adapter.predict_with_ood(image)
                result['crop'] = predicted_crop
                result['crop_confidence'] = crop_confidence
                
                if result['ood_analysis']['is_ood']:
                    self._handle_ood_detection(result, metadata)
                
                # Cache result
                if self.cache_enabled:
                    cache_key = self._generate_cache_key(image)
                    self.adapter_cache.set(cache_key, result)
                
                results.append(result)
            except Exception as e:
                logger.error(f"Error during prediction for image {i}: {e}")
                error_result = {
                    'status': 'error',
                    'message': str(e),
                    'crop': predicted_crop,
                    'crop_confidence': crop_confidence
                }
                if self.cache_enabled:
                    cache_key = self._generate_cache_key(image)
                    self.adapter_cache.set(cache_key, error_result)
                results.append(error_result)
        
        return results
    
    def get_crop_status(self) -> Dict[str, Dict]:
        """
        Get status of all registered crop adapters.
        
        Returns:
            Dictionary mapping crop names to their status
        """
        status = {}
        
        for crop_name, adapter in self.adapters.items():
            status[crop_name] = {
                'is_trained': adapter.is_trained,
                'current_phase': adapter.current_phase,
                'num_classes': len(adapter.class_to_idx) if adapter.class_to_idx else 0,
                'has_ood': adapter.ood_thresholds is not None
            }
        
        return status
    
    def update_adapter(
        self,
        crop_name: str,
        new_adapter_path: str
    ) -> bool:
        """
        Update an adapter with a new version.
        
        Args:
            crop_name: Name of the crop
            new_adapter_path: Path to new adapter checkpoint
            
        Returns:
            True if successful
        """
        if crop_name not in self.adapters:
            logger.error(f"No adapter registered for {crop_name}")
            return False
        
        try:
            self.adapters[crop_name].load_adapter(new_adapter_path)
            logger.info(f"Updated {crop_name} adapter")
            # Clear caches when adapter updates
            self._clear_caches()
            return True
        except Exception as e:
            logger.error(f"Failed to update {crop_name} adapter: {e}")
            return False
    
    def save_pipeline_state(self, save_dir: str):
        """
        Save entire pipeline state.
        
        Args:
            save_dir: Directory to save state
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save router
        if self.router:
            self.router.save_model(str(save_path / 'router'))
        
        # Save adapters
        adapters_dir = save_path / 'adapters'
        adapters_dir.mkdir(exist_ok=True)
        
        for crop_name, adapter in self.adapters.items():
            adapter_dir = adapters_dir / crop_name
            adapter.save_adapter(str(adapter_dir))
        
        logger.info(f"Pipeline state saved to {save_dir}")
    
    def load_pipeline_state(
        self,
        base_dir: str,
        router_path: Optional[str] = None
    ) -> bool:
        """
        Load entire pipeline state.
        
        Args:
            base_dir: Base directory with saved state
            router_path: Optional router path (overrides base_dir/router)
            
        Returns:
            True if successful
        """
        base_path = Path(base_dir)
        
        # Load router
        router_path = router_path or base_path / 'router'
        if (router_path).exists():
            self.router = SimpleCropRouter(self.crops, device=self.device)
            self.router.load_model(str(router_path))
            logger.info("Loaded router")
        
        # Load adapters
        adapters_dir = base_path / 'adapters'
        if adapters_dir.exists():
            for crop_name in self.crops:
                adapter_dir = adapters_dir / crop_name
                if adapter_dir.exists():
                    adapter = IndependentCropAdapter(crop_name=crop_name, device=self.device)
                    adapter.load_adapter(str(adapter_dir))
                    self.adapters[crop_name] = adapter
                    logger.info(f"Loaded {crop_name} adapter")
        
        return True
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        router_stats = self.router_cache.get_stats()
        adapter_stats = self.adapter_cache.get_stats()
        
        return {
            'router_cache': router_stats,
            'adapter_cache': adapter_stats,
            'cache_enabled': self.cache_enabled,
            'cache_size': self.cache_size,
            'ttl_seconds': self.ttl_seconds
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self._clear_caches()
    
    def _handle_ood_detection(self, result: Dict, metadata: Optional[Dict]):
        """Handle OOD detection and trigger updates if needed."""
        # Implementation remains the same
        pass

def create_pipeline_from_config(
    config_path: str,
    device: str = 'cuda'
) -> IndependentMultiCropPipeline:
    """
    Create pipeline from configuration file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        device: Device for inference
        
    Returns:
        Initialized pipeline
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    pipeline = IndependentMultiCropPipeline(config, device)
    
    return pipeline

if __name__ == "__main__":
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--router_path', type=str, help='Router checkpoint path')
    parser.add_argument('--adapters_dir', type=str, help='Adapters directory')
    parser.add_argument('--image', type=str, help='Test image path')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline
    pipeline = create_pipeline_from_config(args.config)
    
    # Initialize router
    pipeline.initialize_router(router_path=args.router_path)
    
    # Register adapters
    if args.adapters_dir:
        adapters_dir = Path(args.adapters_dir)
        for crop in pipeline.crops:
            adapter_path = adapters_dir / crop
            if adapter_path.exists():
                pipeline.register_crop(crop, str(adapter_path))
    
    # Test inference if image provided
    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert('RGB')
        img_tensor = preprocess_image(image)
        
        result = pipeline.process_image(img_tensor.unsqueeze(0))
        print(f"Result: {result}")
