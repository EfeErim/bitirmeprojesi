#!/usr/bin/env python3
"""
Phase 1 Training: DoRA Initialization for Independent Crop Adapter
Trains base adapter with DoRA for a specific crop (e.g., tomato).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, DoRAConfig
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

from src.utils.data_loader import CropDataset
from src.utils.metrics import compute_metrics
from src.ood.prototypes import compute_class_prototypes

logger = logging.getLogger(__name__)

class Phase1Trainer:
    """
    Phase 1 trainer for DoRA-based adapter initialization.
    
    Args:
        model_name: Pretrained model name (e.g., 'facebook/dinov2-giant')
        num_classes: Number of disease classes for this crop
        lora_r: LoRA rank (default: 32)
        lora_alpha: LoRA alpha (default: 32)
        lora_dropout: Dropout rate (default: 0.1)
        loraplus_lr_ratio: LR ratio for B matrices (default: 16)
        device: Device for training
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/dinov2-giant',
        num_classes: int = 5,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        loraplus_lr_ratio: int = 16,
        gradient_accumulation_steps: int = 1,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Load pretrained model
        logger.info(f"Loading pretrained model: {model_name}")
        self.base_model = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Mixed precision training
        self.use_amp = torch.cuda.is_available() and torch.backends.cuda.is_built()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 1
        self.current_step = 0
        
        # Get hidden size
        if hasattr(self.config, 'hidden_size'):
            self.hidden_size = self.config.hidden_size
        elif hasattr(self.config, 'dim'):
            self.hidden_size = self.config.dim
        else:
            raise ValueError(f"Cannot determine hidden size from config: {self.config}")
        
        # Add classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes).to(self.device)
        
        # Configure DoRA
        logger.info("Configuring DoRA adapter...")
        lora_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",  # For feature extraction before classification
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['query', 'value'],  # Target attention mechanisms
            lora_dropout=lora_dropout,
            use_dora=True,  # Enable DoRA
        )
        
        # Apply PEFT model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model = self.model.to(self.device)
        
        # Separate parameters for LoRA+ optimization
        self.optimizer = self._create_loraplus_optimizer(loraplus_lr_ratio)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Phase1Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def _create_loraplus_optimizer(self, loraplus_lr_ratio: int):
        """
        Create LoRA+ optimizer with different learning rates for A and B matrices.
        
        LoRA+: B matrices (magnitude) get higher LR than A matrices (direction).
        """
        # Separate parameters
        lora_a_params = []
        lora_b_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora_A' in name:
                    lora_a_params.append(param)
                elif 'lora_B' in name:
                    lora_b_params.append(param)
                else:
                    other_params.append(param)
        
        # Add classifier parameters
        other_params.extend(self.classifier.parameters())
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': lora_a_params, 'lr': 1e-4},  # Base LR for A
            {'params': lora_b_params, 'lr': 1e-4 * loraplus_lr_ratio},  # Higher LR for B
            {'params': other_params, 'lr': 1e-4}  # Standard LR for others
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        return optimizer
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with mixed precision and gradient accumulation.
        
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
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.base_model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
                logits = self.classifier(pooled_output)
                
                # Compute loss
                loss = self.criterion(logits, labels)
            
            # Backward pass with gradient accumulation
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            self.current_step += 1
            if self.current_step % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.current_step = 0
            
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
            outputs = self.base_model(images)
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
        num_epochs: int = 50,
        save_dir: Optional[str] = None,
        early_stopping_patience: int = 10
    ) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            
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
        patience_counter = 0
        
        logger.info(f"Starting Phase 1 training for {num_epochs} epochs")
        
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
                patience_counter = 0
                
                if save_dir:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_accuracy': val_metrics['accuracy'],
                        'config': {
                            'num_classes': self.num_classes,
                            'hidden_size': self.hidden_size
                        }
                    }
                    torch.save(checkpoint, save_path / 'phase1_best.pth')
                    logger.info(f"Best checkpoint saved with val_acc: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Phase 1 training completed. Best val accuracy: {best_val_accuracy:.4f}")
        return history
    
    @torch.no_grad()
    def compute_prototypes(
        self, 
        data_loader: DataLoader
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Compute class prototypes from training data for OOD detection.
        
        Returns:
            class_means: Tensor of shape (num_classes, hidden_size)
            class_stds: Dictionary mapping class index to std tensor
        """
        self.model.eval()
        
        # Collect features per class
        features_per_class = {i: [] for i in range(self.num_classes)}
        
        for images, labels in data_loader:
            images = images.to(self.device)
            
            # Extract features
            outputs = self.base_model(images)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            features = pooled_output
            
            # Store by class
            for feat, label in zip(features, labels):
                class_idx = label.item()
                features_per_class[class_idx].append(feat.cpu())
        
        # Compute means and stds
        class_means = torch.zeros(self.num_classes, self.hidden_size)
        class_stds = {}
        
        for class_idx, feat_list in features_per_class.items():
            if len(feat_list) == 0:
                logger.warning(f"No samples for class {class_idx}")
                continue
            
            feats = torch.stack(feat_list)
            mean = feats.mean(dim=0)
            std = feats.std(dim=0)
            
            class_means[class_idx] = mean
            class_stds[class_idx] = std
        
        logger.info(f"Computed prototypes for {len(class_stds)} classes")
        return class_means, class_stds
    
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
    """Example usage of Phase1Trainer."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--crop', type=str, default='tomato', help='Crop name')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create datasets
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
    
    # Get number of classes
    num_classes = len(train_dataset.classes)
    logger.info(f"Number of classes: {num_classes}")
    
    # Initialize trainer
    trainer = Phase1Trainer(
        model_name='facebook/dinov2-giant',
        num_classes=num_classes,
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        loraplus_lr_ratio=16,
        device='cuda'
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.output_dir
    )
    
    # Compute prototypes
    logger.info("Computing class prototypes...")
    prototypes, stds = trainer.compute_prototypes(train_loader)
    
    # Save prototypes
    torch.save({
        'prototypes': prototypes,
        'stds': stds,
        'class_to_idx': train_dataset.class_to_idx
    }, Path(args.output_dir) / 'prototypes.pt')
    
    logger.info("Phase 1 training completed successfully!")


if __name__ == "__main__":
    main()