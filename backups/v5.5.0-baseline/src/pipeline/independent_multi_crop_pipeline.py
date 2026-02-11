#!/usr/bin/env python3
"""
Independent Multi-Crop Pipeline for AADS-ULoRA v5.5
Main pipeline orchestrating router and independent adapters.
Key principle: No cross-adapter communication - fully independent.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.router.simple_crop_router import SimpleCropRouter
from src.adapter.independent_crop_adapter import IndependentCropAdapter
from src.utils.data_loader import preprocess_image

logger = logging.getLogger(__name__)

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
        
        logger.info(f"IndependentMultiCropPipeline initialized on {self.device}")
    
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
            model_name='facebook/dinov2-base',
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
        # Step 1: Route to correct crop adapter
        if self.router is None:
            raise RuntimeError("Router not initialized")
        
        predicted_crop, crop_confidence = self.router.route(image)
        
        logger.debug(f"Routed to crop: {predicted_crop} (confidence: {crop_confidence:.4f})")
        
        # Step 2: Get appropriate adapter
        if predicted_crop not in self.adapters:
            return {
                'status': 'error',
                'message': f'No adapter available for crop: {predicted_crop}',
                'crop': predicted_crop,
                'crop_confidence': crop_confidence
            }
        
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
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'crop': predicted_crop,
                'crop_confidence': crop_confidence
            }
    
    def _handle_ood_detection(
        self,
        result: Dict,
        metadata: Optional[Dict]
    ):
        """
        Handle OOD detection event.
        
        Args:
            result: Prediction result with OOD analysis
            metadata: Optional metadata
        """
        logger.warning(f"OOD detected for {result['crop']}: {result['ood_analysis']}")
        
        # Store OOD sample for later expert labeling
        # In a full implementation, this would:
        # 1. Save the image to storage
        # 2. Queue for expert review
        # 3. Trigger Phase 2 training when enough samples collected
        
        # For now, just log
        logger.info("OOD sample queued for expert review")
    
    def batch_process(
        self,
        images: List[torch.Tensor],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            images: List of image tensors
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, image in enumerate(images):
            metadata = metadata_list[i] if metadata_list else None
            result = self.process_image(image, metadata)
            results.append(result)
        
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