#!/usr/bin/env python3
"""
Simple Crop Router for AADS-ULoRA v5.5
Lightweight crop classification using frozen DINOv2 backbone with trainable linear classifier.
Target accuracy: ≥98%
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from importlib.machinery import ModuleSpec

_peft_module = sys.modules.get("peft")
if _peft_module is not None and getattr(_peft_module, "__spec__", None) is None:
    _peft_module.__spec__ = ModuleSpec("peft", loader=None)

from transformers import AutoModel
import logging
from pathlib import Path
from typing import Optional, List
import json

logger = logging.getLogger(__name__)


class SimpleCropRouter:
    """
    Simple crop router for routing images to per-crop adapters.
    
    Architecture:
    - Frozen DINOv2-giant backbone for feature extraction
    - Trainable linear classifier on top
    - Target: ≥98% crop classification accuracy
    
    This is the Layer 1 of the v5.5 independent multi-crop architecture.
    Once trained, routes images to the appropriate per-crop adapter (Layer 2).
    """
    
    def __init__(self, crops: List[str], model_name: str = 'facebook/dinov2-giant', device: str = 'cuda'):
        """
        Initialize simple crop router.
        
        Args:
            crops: List of crop names (e.g., ['tomato', 'pepper', 'corn'])
            model_name: Hugging Face model id for DINOv2 backbone
            device: Device to use ('cuda' or 'cpu')
        """
        self.crops = crops
        self.num_crops = len(crops)
        self.model_name = str(model_name)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing SimpleCropRouter for {self.num_crops} crops: {crops}")
        logger.info(f"Device: {self.device}")
        
        # Load frozen DINOv2-giant backbone
        logger.info(f"Loading backbone (frozen): {self.model_name}")
        try:
            self.backbone = AutoModel.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load backbone {self.model_name}: {e}")
            raise RuntimeError(f"Cannot load backbone {self.model_name}: {e}")
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone = self.backbone.to(self.device)
        
        backbone_config = getattr(self.backbone, 'config', None)
        hidden_size = getattr(backbone_config, 'hidden_size', None)
        self.feature_dim = int(hidden_size) if hidden_size is not None else 1536
        
        # Add trainable linear classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_crops).to(self.device)
        
        # Training configuration
        self.best_accuracy = 0.0
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        logger.info(f"SimpleCropRouter initialized")
        logger.info(f"Backbone: {self.model_name} (frozen, feature_dim={self.feature_dim})")
        logger.info(f"Classifier: Linear({self.feature_dim}, {self.num_crops})")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.classifier.parameters()):,}")
    
    def train(self, train_loader: DataLoader, epochs: int = 10, lr: float = 1e-3,
              val_loader: Optional[DataLoader] = None) -> float:
        """
        Train the crop classifier on crop classification data.
        Backbone remains frozen; only classifier is trained.
        
        Args:
            train_loader: DataLoader with (images, crop_labels)
            epochs: Number of training epochs (default: 10)
            lr: Learning rate (default: 1e-3)
            val_loader: Optional validation DataLoader
            
        Returns:
            Best validation accuracy achieved (≥0.98 target)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting SimpleCropRouter Training")
        logger.info(f"Epochs: {epochs}, LR: {lr}")
        logger.info(f"Target accuracy: ≥98%")
        logger.info(f"{'='*60}\n")
        
        # Optimizer - trains only classifier (backbone frozen)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            # Training phase
            self.classifier.train()
            self.backbone.eval()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass - extract features from frozen backbone
                with torch.no_grad():
                    # DINOv2 returns a model output with last_hidden_state
                    backbone_output = self.backbone(images)
                    if hasattr(backbone_output, 'last_hidden_state'):
                        features = backbone_output.last_hidden_state[:, 0]  # [CLS] token
                    else:
                        # Fallback: assume output is features
                        features = backbone_output
                
                # Classify crops using trainable classifier
                logits = self.classifier(features)
                loss = criterion(logits, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item() * labels.size(0)
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= train_total
            train_accuracy = 100.0 * train_correct / train_total
            
            # Validation phase
            val_accuracy = 0.0
            if val_loader is not None:
                val_accuracy = self._evaluate(val_loader)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Train Accuracy: {train_accuracy:.2f}%")
            if val_loader is not None:
                logger.info(f"  Val Accuracy: {val_accuracy:.2f}%")
            
            # Save best checkpoint
            if val_loader is not None and val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                logger.info(f"  ✓ New best accuracy: {self.best_accuracy:.2f}%")
            
            # Record history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            if val_loader is not None:
                self.training_history['val_accuracy'].append(val_accuracy)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete")
        if val_loader is not None:
            logger.info(f"Best accuracy: {self.best_accuracy:.2f}%")
            if self.best_accuracy < 0.95:
                logger.warning(f"⚠️  Warning: Accuracy {self.best_accuracy:.2f}% < 95% (target: 98%)")
        logger.info(f"{'='*60}\n")
        
        return self.best_accuracy
    
    def _evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate router on validation set.
        
        Args:
            val_loader: Validation DataLoader
            
        Returns:
            Accuracy as percentage (0-100)
        """
        self.classifier.eval()
        self.backbone.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                backbone_output = self.backbone(images)
                if hasattr(backbone_output, 'last_hidden_state'):
                    features = backbone_output.last_hidden_state[:, 0]
                else:
                    features = backbone_output
                
                # Classify
                logits = self.classifier(features)
                _, predicted = logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def route(self, image: torch.Tensor) -> str:
        """
        Route a single image to appropriate crop adapter.
        
        Args:
            image: Tensor of shape (1, 3, H, W) or (3, H, W)
            
        Returns:
            Crop name (e.g., 'tomato')
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        self.classifier.eval()
        self.backbone.eval()
        
        with torch.no_grad():
            # Extract features
            backbone_output = self.backbone(image)
            if hasattr(backbone_output, 'last_hidden_state'):
                features = backbone_output.last_hidden_state[:, 0]
            else:
                features = backbone_output
            
            # Classify
            logits = self.classifier(features)
            crop_idx = logits.argmax(dim=1).item()
        
        crop_name = self.crops[crop_idx]
        logger.debug(f"Routed image to crop: {crop_name} (logits: {logits.cpu().numpy()})")
        
        return crop_name
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save router checkpoint to disk.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'classifier_state_dict': self.classifier.state_dict(),
            'crops': self.crops,
            'num_crops': self.num_crops,
            'feature_dim': self.feature_dim,
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Router checkpoint saved to {path}")

    def save_model(self, path: str) -> None:
        """Backward-compatible alias for save_checkpoint."""
        self.save_checkpoint(path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load router checkpoint from disk.
        
        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Verify compatibility
        if checkpoint['num_crops'] != self.num_crops:
            raise ValueError(
                f"Checkpoint num_crops ({checkpoint['num_crops']}) != "
                f"current num_crops ({self.num_crops})"
            )
        
        # Load classifier weights
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', {
            'epoch': [], 'train_loss': [], 'train_accuracy': [], 'val_accuracy': []
        })
        
        logger.info(f"Router checkpoint loaded from {path}")
        logger.info(f"Best accuracy from checkpoint: {self.best_accuracy:.2f}%")

    def load_model(self, path: str) -> None:
        """Backward-compatible alias for load_checkpoint."""
        self.load_checkpoint(path)
    
    def get_summary(self) -> dict:
        """Get router summary information."""
        return {
            'crops': self.crops,
            'num_crops': self.num_crops,
            'backbone': self.model_name,
            'feature_dim': self.feature_dim,
            'classifier_params': sum(p.numel() for p in self.classifier.parameters()),
            'best_accuracy': self.best_accuracy,
            'target_accuracy': 0.98
        }
