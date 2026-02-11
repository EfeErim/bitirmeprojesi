#!/usr/bin/env python3
"""
Phase 2 Training: SD-LoRA for New Disease Addition
Implements class-incremental learning with SD-LoRA for adding new diseases to existing adapters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, SDLoRAConfig
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

from src.utils.data_loader import CropDataset
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

class Phase2Trainer:
    """
    Phase 2 trainer for SD-LoRA-based disease addition.
    
    Args:
        adapter_path: Path to existing adapter checkpoint
        new_classes: List of new disease classes to add
        lora_r: LoRA rank (default: 32)
        lora_alpha: LoRA alpha (default: 32)
        device: Device for training
    """
    
    def __init__(
        self,
        adapter_path: str,
        new_classes: List[str],
        lora_r: int = 32,
        lora_alpha: int = 32,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.new_classes = new_classes
        
        # Load existing adapter
        logger.info(f"Loading adapter from: {adapter_path}")
        self.base_model = AutoModel.from_pretrained(adapter_path)
        self.config = AutoConfig.from_pretrained(adapter_path)
        
        # Get hidden size
        if hasattr(self.config, 'hidden_size'):
            self.hidden_size = self.config.hidden_size
        elif hasattr(self.config, 'dim'):
            self.hidden_size = self.config.dim
        else:
            raise ValueError(f"Cannot determine hidden size from config: {self.config}")
        
        # Add new classifier head
        self.classifier = nn.Linear(self.hidden_size, len(self.new_classes) + len(self.base_model.classifier.weight.data))
        self.classifier.to(self.device)
        
        # Configure SD-LoRA
        logger.info("Configuring SD-LoRA adapter...")
        lora_config = SDLoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            target_modules=['query', 'value']  # Target attention mechanisms
        )
        
        # Apply PEFT model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model = self.model.to(self.device)
        
        # Initialize new classifier weights
        self._initialize_new_classifier()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Phase2Trainer initialized on device: {self.device}")
    
    def _initialize_new_classifier(self):
        """
        Initialize new classifier weights using Xavier initialization
        """
        # Get existing classifier weights
        existing_weights = self.base_model.classifier.weight.data
        existing_bias = self.base_model.classifier.bias.data
        
        # Create new weights with Xavier initialization
        new_weights = torch.nn.init.xavier_uniform_(torch.empty(len(self.new_classes), existing_weights.size(1)))
        new_bias = torch.zeros(len(self.new_classes))
        
        # Combine with existing weights
        self.classifier.weight.data = torch.cat([existing_weights, new_weights], dim=0)
        self.classifier.bias.data = torch.cat([existing_bias, new_bias], dim=0)
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # Loss
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            # Predictions
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute detailed metrics
        metrics = compute_metrics(
            predictions=np.array(all_predictions),
            labels=np.array(all_labels)
        )
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        save_dir: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_accuracy = 0.0
        
        logger.info(f"Starting Phase 2 training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save best checkpoint
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                if save_dir:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'val_accuracy': val_metrics['accuracy'],
                        'config': {
                            'new_classes': self.new_classes
                        }
                    }
                    torch.save(checkpoint, save_path / 'phase2_best.pth')
                    logger.info(f"Best checkpoint saved with val_acc: {best_val_accuracy:.4f}")
        
        logger.info(f"Phase 2 training completed. Best val accuracy: {best_val_accuracy:.4f}")
        return history
    
    def save_adapter(self, save_path: str):
        """Save the trained adapter and classifier."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PEFT model
        self.model.save_pretrained(save_path / 'adapter')
        
        # Save classifier
        torch.save(
            self.classifier.state_dict(),
            save_path / 'classifier.pth'
        )
        
        logger.info(f"Adapter saved to {save_path}")
    
    def load_adapter(self, load_path: str):
        """Load a trained adapter and classifier."""
        load_path = Path(load_path)
        
        # Load PEFT model
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.base_model,
            load_path / 'adapter'
        )
        self.model = self.model.to(self.device)
        
        # Load classifier
        self.classifier.load_state_dict(
            torch.load(load_path / 'classifier.pth', map_location=self.device)
        )
        
        logger.info(f"Adapter loaded from {load_path}")


def main():
    """Example usage of Phase2Trainer."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_path', type=str, required=True, help='Path to existing adapter checkpoint')
    parser.add_argument('--new_classes', type=str, nargs='+', required=True, help='New disease classes to add')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create datasets
    train_dataset = CropDataset(
        data_dir=args.data_dir,
        crop='tomato',  # Assuming we're adding to tomato adapter
        split='train',
        transform=True
    )
    val_dataset = CropDataset(
        data_dir=args.data_dir,
        crop='tomato',
        split='val',
        transform=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = Phase2Trainer(
        adapter_path=args.adapter_path,
        new_classes=args.new_classes,
        lora_r=32,
        lora_alpha=32,
        device='cuda'
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.output_dir
    )
    
    # Save adapter
    trainer.save_adapter(args.output_dir)
    
    logger.info("Phase 2 training completed successfully!")


if __name__ == "__main__":
    main()