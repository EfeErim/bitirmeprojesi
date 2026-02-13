#!/usr/bin/env python3
"""
Simple Crop Router for AADS-ULoRA v5.5
Routes input images to the correct crop adapter using a lightweight classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from typing import List, Tuple, Optional, Dict, Union
import logging
from pathlib import Path
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class SimpleCropRouter:
    """
    Lightweight crop classifier using DINOv3 linear probe.
    Target: 98%+ crop classification accuracy
    
    Args:
        crops: List of supported crop names
        model_name: Pretrained model name (default: 'facebook/dinov3-base')
        device: Device for inference
    """
    
    def __init__(
        self,
        crops: List[str],
        model_name: str = 'facebook/dinov3-base',
        device: str = 'cuda',
        confidence_threshold: float = 0.92,
        top_k_alternatives: int = 3
    ):
        self.crops = crops
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.top_k_alternatives = min(top_k_alternatives, len(crops))
        
        # Load pretrained model
        logger.info(f"Loading crop router model: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Get hidden size
        if hasattr(self.config, 'hidden_size'):
            self.hidden_size = self.config.hidden_size
        elif hasattr(self.config, 'dim'):
            self.hidden_size = self.config.dim
        else:
            raise ValueError(f"Cannot determine hidden size from config: {self.config}")
        
        # Add classification head
        self.classifier = nn.Linear(self.hidden_size, len(crops)).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Cache for frequently accessed images
        self.image_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cache TTL support
        self.cache_ttl_seconds = None  # Optional TTL
        
        # Confidence statistics tracking
        self.confidence_stats = {
            'total_predictions': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'rejected_predictions': 0,
            'fallback_used': 0,
            'confidence_sum': 0.0
        }
        
        logger.info(f"CropRouter initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.backbone.parameters())}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.classifier.parameters())}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"Top-K alternatives: {self.top_k_alternatives}")
    
    def _generate_cache_key(self, image_tensor: torch.Tensor) -> str:
        """Generate a cache key for an image tensor."""
        # Convert tensor to bytes for hashing
        tensor_bytes = image_tensor.cpu().numpy().tobytes()
        import hashlib
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def train(
        self,
        train_dataset: 'CropDataset',
        val_dataset: 'CropDataset',
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the crop router.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_path: Path to save trained model
            
        Returns:
            Dictionary with training metrics
        """
        self.backbone.eval()  # Freeze backbone
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=learning_rate)
        
        best_val_accuracy = 0.0
        
        logger.info(f"Starting crop router training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            self.backbone.eval()
            self.classifier.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.backbone(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(pooled_output)
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total if total > 0 else 0.0
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Best model saved with val_acc: {best_val_accuracy:.4f}")
        
        return {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': best_val_accuracy
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the crop router.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.backbone.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.backbone(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(pooled_output)
                
                # Loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Predictions
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def route(
        self,
        image: torch.Tensor
    ) -> Tuple[str, float]:
        """
        Route an image to the correct crop adapter with confidence-based rejection.
        
        Args:
            image: Preprocessed image tensor (batch size 1)
            
        Returns:
            Tuple of (predicted_crop, confidence_score)
            
        Raises:
            ValueError: If confidence is below threshold
        """
        self.backbone.eval()
        self.classifier.eval()
        
        # Check cache first
        cache_key = self._generate_cache_key(image)
        if cache_key in self.image_cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for image (key: {cache_key[:8]}...)")
            cached_result = self.image_cache[cache_key]
            # Update stats from cached result
            self.confidence_stats['total_predictions'] += 1
            self.confidence_stats['confidence_sum'] += cached_result[1]
            self.confidence_stats['high_confidence'] += 1
            return cached_result
        
        self.cache_misses += 1
        
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.backbone(image)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # Get prediction
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            # Update confidence statistics
            self.confidence_stats['total_predictions'] += 1
            self.confidence_stats['confidence_sum'] += confidence
            
            if confidence < self.confidence_threshold:
                self.confidence_stats['low_confidence'] += 1
                self.confidence_stats['rejected_predictions'] += 1
                
                # Get top-K alternatives for debugging
                top_k_values, top_k_indices = torch.topk(probabilities, self.top_k_alternatives)
                top_k_crops = [self.crops[idx.item()] for idx in top_k_indices[0]]
                top_k_confidences = top_k_values[0].tolist()
                
                # Log low confidence event
                logger.warning(f"Low confidence prediction: {confidence:.4f} < {self.confidence_threshold}")
                logger.debug(f"Top-{self.top_k_alternatives} alternatives: {list(zip(top_k_crops, top_k_confidences))}")
                
                # Raise exception to trigger fallback
                # Store additional info in the exception args for debugging
                raise ValueError(
                    f"Prediction confidence {confidence:.4f} below threshold {self.confidence_threshold}. "
                    f"Predicted: {self.crops[predicted_idx]}, "
                    f"Top-{self.top_k_alternatives}: {list(zip(top_k_crops, top_k_confidences))}"
                )
            else:
                self.confidence_stats['high_confidence'] += 1
                
                predicted_crop = self.crops[predicted_idx]
                result = (predicted_crop, confidence)
                
                # Store in cache
                self.image_cache[cache_key] = result
                
                return result
    
    def route_batch(
        self,
        images: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        """
        Route multiple images in batch for improved performance with confidence-based rejection.
        
        Args:
            images: Preprocessed image tensor (batch size > 1)
            
        Returns:
            Tuple of (predicted_crops, confidence_scores)
            
        Raises:
            ValueError: If any prediction confidence is below threshold
        """
        self.backbone.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.backbone(images)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            predicted_indices = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1).values
            
            # Check for low confidence predictions
            low_confidence_mask = confidences < self.confidence_threshold
            
            if torch.any(low_confidence_mask):
                # Update statistics
                num_low_confidence = torch.sum(low_confidence_mask).item()
                self.confidence_stats['total_predictions'] += len(images)
                self.confidence_stats['low_confidence'] += num_low_confidence
                self.confidence_stats['rejected_predictions'] += num_low_confidence
                self.confidence_stats['confidence_sum'] += torch.sum(confidences).item()
                
                # Get indices of low confidence predictions
                low_confidence_indices = torch.nonzero(low_confidence_mask).flatten()
                
                # Log low confidence events
                for idx in low_confidence_indices:
                    confidence = confidences[idx].item()
                    predicted_crop = self.crops[predicted_indices[idx].item()]
                    
                    # Get top-K alternatives for this specific image
                    top_k_values, top_k_indices = torch.topk(probabilities[idx], self.top_k_alternatives)
                    top_k_crops = [self.crops[i.item()] for i in top_k_indices]
                    top_k_confidences = top_k_values.tolist()
                    
                    logger.warning(f"Low confidence prediction for image {idx}: {confidence:.4f} < {self.confidence_threshold}")
                    logger.debug(f"Top-{self.top_k_alternatives} alternatives: {list(zip(top_k_crops, top_k_confidences))}")
                
                # Build detailed error message
                top_k_alternatives_list = []
                for idx in low_confidence_indices:
                    top_k_values, top_k_indices = torch.topk(probabilities[idx], self.top_k_alternatives)
                    alternatives = [(self.crops[i.item()], v.item()) for i, v in zip(top_k_indices, top_k_values)]
                    top_k_alternatives_list.append(alternatives)
                
                error_msg = (
                    f"Found {num_low_confidence} predictions with confidence below threshold {self.confidence_threshold}. "
                    f"Low confidence indices: {low_confidence_indices.tolist()}, "
                    f"Low confidence values: {confidences[low_confidence_mask].tolist()}, "
                    f"Top-K alternatives: {top_k_alternatives_list}"
                )
                raise ValueError(error_msg)
            else:
                # All predictions are high confidence
                self.confidence_stats['total_predictions'] += len(images)
                self.confidence_stats['high_confidence'] += len(images)
                self.confidence_stats['confidence_sum'] += torch.sum(confidences).item()
                
                predicted_crops = [self.crops[idx.item()] for idx in predicted_indices]
                confidence_scores = confidences.tolist()
                
                return predicted_crops, confidence_scores
    
    def save_model(self, save_path: str):
        """
        Save the trained crop router.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        torch.save(
            self.classifier.state_dict(),
            save_path / 'crop_router_classifier.pth'
        )
        
        logger.info(f"Crop router saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load a trained crop router.
        
        Args:
            load_path: Path to load model from
        """
        load_path = Path(load_path)
        
        # Load classifier
        self.classifier.load_state_dict(
            torch.load(load_path / 'crop_router_classifier.pth', map_location=self.device)
        )
        
        logger.info(f"Crop router loaded from {load_path}")
    
    def get_confidence_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get confidence statistics.
        
        Returns:
            Dictionary with confidence statistics
        """
        stats = self.confidence_stats.copy()
        if stats['total_predictions'] > 0:
            stats['mean_confidence'] = stats['confidence_sum'] / stats['total_predictions']
            stats['high_confidence_rate'] = stats['high_confidence'] / stats['total_predictions']
            stats['low_confidence_rate'] = stats['low_confidence'] / stats['total_predictions']
            stats['rejection_rate'] = stats['rejected_predictions'] / stats['total_predictions']
        else:
            stats['mean_confidence'] = 0.0
            stats['high_confidence_rate'] = 0.0
            stats['low_confidence_rate'] = 0.0
            stats['rejection_rate'] = 0.0
        
        return stats
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.image_cache)
        }
    
    def get_full_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive statistics including cache and confidence stats.
        
        Returns:
            Dictionary with all statistics
        """
        stats = {
            'cache': self.get_cache_stats(),
            'confidence': self.get_confidence_stats()
        }
        return stats
    
    def reset_confidence_stats(self):
        """Reset confidence statistics to zero."""
        self.confidence_stats = {
            'total_predictions': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'rejected_predictions': 0,
            'fallback_used': 0,
            'confidence_sum': 0.0
        }
        logger.info("Confidence statistics reset")

if __name__ == "__main__":
    """Example usage of SimpleCropRouter."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--crop', type=str, default='tomato', help='Crop name')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create datasets
    from src.utils.data_loader import CropDataset
    
    train_dataset = CropDataset(
        data_dir=args.data_dir,
        crop=args.crop,
        split='train',
        transform=True
    )
    val_dataset = CropDataset(
        data_dir=args.data_dir,
        crop=args.crop,
        split='val',
        transform=False
    )
    
    # Initialize router
    crops = ['tomato', 'pepper', 'corn']
    router = SimpleCropRouter(
        crops=crops,
        model_name='facebook/dinov3-base',
        device='cuda'
    )
    
    # Train
    metrics = router.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.output_dir
    )
    
    logger.info(f"Training completed with metrics: {metrics}")