#!/usr/bin/env python3
"""
Simple Crop Router for AADS-ULoRA v5.5
Routes input images to the correct crop adapter using a lightweight classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleCropRouter:
    """
    Lightweight crop classifier using DINOv2 linear probe.
    Target: 98%+ crop classification accuracy
    
    Args:
        crops: List of supported crop names
        model_name: Pretrained model name (default: 'facebook/dinov2-base')
        device: Device for inference
    """
    
    def __init__(
        self,
        crops: List[str],
        model_name: str = 'facebook/dinov2-base',
        device: str = 'cuda'
    ):
        self.crops = crops
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
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
        
        logger.info(f"CropRouter initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.backbone.parameters())}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.classifier.parameters())}")
    
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
        Route an image to the correct crop adapter.
        
        Args:
            image: Preprocessed image tensor (batch size 1)
            
        Returns:
            Tuple of (predicted_crop, confidence_score)
        """
        self.backbone.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.backbone(image)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # Get prediction
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            predicted_crop = self.crops[predicted_idx]
            
            return predicted_crop, confidence
    
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
        model_name='facebook/dinov2-base',
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