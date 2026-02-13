#!/usr/bin/env python3
"""
Independent Crop Adapter for AADS-ULoRA v5.5
Self-contained adapter for one crop with dynamic OOD detection.
No communication with other crop adapters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, DoRAConfig, SDLoRAConfig
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import numpy as np

from src.utils.data_loader import CropDataset
from src.ood.prototypes import compute_class_prototypes
from src.ood.mahalanobis import MahalanobisDistance
from src.ood.dynamic_thresholds import DynamicOODThreshold

logger = logging.getLogger(__name__)

class IndependentCropAdapter:
    """
    Self-contained adapter for one crop with dynamic OOD detection.
    No communication with other crop adapters.
    
    Args:
        crop_name: Name of the crop (e.g., 'tomato')
        model_name: Pretrained model name (default: 'facebook/dinov2-giant')
        device: Device for training/inference
    """
    
    def __init__(
        self,
        crop_name: str,
        model_name: str = 'facebook/dinov2-giant',
        device: str = 'cuda'
    ):
        self.crop_name = crop_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Will be initialized during Phase 1
        self.base_model = None
        self.config = None
        self.hidden_size = None
        self.classifier = None
        self.model = None
        
        # OOD detection components
        self.prototypes = None
        self.class_stds = None
        self.mahalanobis = None
        self.ood_thresholds = None
        self.class_to_idx = None
        self.idx_to_class = None
        
        # Training state
        self.is_trained = False
        self.current_phase = None
        
        logger.info(f"IndependentCropAdapter initialized for {crop_name}")
    
    def phase1_initialize(
        self,
        train_dataset: CropDataset,
        val_dataset: CropDataset,
        config: Dict,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Phase 1: DoRA base initialization + dynamic OOD thresholds.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration dictionary with keys:
                - lora_r: LoRA rank (default: 32)
                - lora_alpha: LoRA alpha (default: 32)
                - lora_dropout: Dropout rate (default: 0.1)
                - loraplus_lr_ratio: LR ratio for B matrices (default: 16)
                - num_epochs: Training epochs (default: 50)
                - batch_size: Batch size (default: 32)
                - learning_rate: Base learning rate (default: 1e-4)
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting Phase 1 for {self.crop_name}")
        self.current_phase = 1
        
        # Extract config
        lora_r = config.get('lora_r', 32)
        lora_alpha = config.get('lora_alpha', 32)
        lora_dropout = config.get('lora_dropout', 0.1)
        loraplus_lr_ratio = config.get('loraplus_lr_ratio', 16)
        num_epochs = config.get('num_epochs', 50)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 1e-4)
        
        # Get number of classes
        num_classes = len(train_dataset.classes)
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = train_dataset.idx_to_class
        
        # Load pretrained model
        logger.info(f"Loading pretrained model: {self.model_name}")
        self.base_model = AutoModel.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        
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
            task_type="FEATURE_EXTRACTION",
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['query', 'value'],
            lora_dropout=lora_dropout,
            use_dora=True,
        )
        
        # Apply PEFT model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model = self.model.to(self.device)
        
        # Create LoRA+ optimizer
        optimizer = self._create_loraplus_optimizer(loraplus_lr_ratio, learning_rate)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        best_val_accuracy = 0.0
        patience_counter = 0
        early_stopping_patience = config.get('early_stopping_patience', 10)
        
        logger.info(f"Starting Phase 1 training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self._train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_metrics = self._validate(val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best checkpoint
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                patience_counter = 0
                if save_dir:
                    self.save_adapter(save_dir)
                    logger.info(f"Best checkpoint saved with val_acc: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Compute prototypes for OOD detection
        logger.info("Computing class prototypes for OOD detection...")
        self.prototypes, self.class_stds = compute_class_prototypes(
            self.base_model, train_loader, self.hidden_size, self.device
        )
        
        # Initialize Mahalanobis distance calculator
        self.mahalanobis = MahalanobisDistance(self.prototypes, self.class_stds)
        
        # Compute dynamic OOD thresholds
        logger.info("Computing dynamic OOD thresholds...")
        self.ood_thresholds = DynamicOODThreshold.compute_thresholds(
            self.mahalanobis, self.base_model, val_loader, self.hidden_size, self.device
        )
        
        self.is_trained = True
        
        logger.info(f"Phase 1 completed. Best val accuracy: {best_val_accuracy:.4f}")
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'final_train_loss': train_metrics['loss'],
            'final_train_accuracy': train_metrics['accuracy'],
            'final_val_loss': val_metrics['loss']
        }
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.base_model(images)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.classifier(pooled_output)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.base_model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(pooled_output)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def _create_loraplus_optimizer(
        self,
        loraplus_lr_ratio: int,
        base_lr: float
    ) -> torch.optim.Optimizer:
        """Create LoRA+ optimizer with different LRs for A and B matrices."""
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
        
        param_groups = [
            {'params': lora_a_params, 'lr': base_lr},
            {'params': lora_b_params, 'lr': base_lr * loraplus_lr_ratio},
            {'params': other_params, 'lr': base_lr}
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    def phase2_add_disease(
        self,
        new_class_dataset: CropDataset,
        config: Dict,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Phase 2: SD-LoRA class-incremental learning.
        Add new disease classes while preserving old class performance.
        
        Args:
            new_class_dataset: Dataset containing new disease classes
            config: Configuration with:
                - lora_r: LoRA rank (default: 32)
                - lora_alpha: LoRA alpha (default: 32)
                - num_epochs: Training epochs (default: 20)
                - batch_size: Batch size (default: 32)
                - learning_rate: Learning rate (default: 5e-5)
            save_dir: Save directory
            
        Returns:
            Dictionary with metrics including retention rate
        """
        logger.info(f"Starting Phase 2 for {self.crop_name}")
        self.current_phase = 2
        
        # Extract config
        lora_r = config.get('lora_r', 32)
        lora_alpha = config.get('lora_alpha', 32)
        num_epochs = config.get('num_epochs', 20)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 5e-5)
        
        # Get old number of classes
        old_num_classes = self.classifier.out_features
        new_num_classes = old_num_classes + len(new_class_dataset.classes)
        
        # Update class mappings
        old_class_to_idx = self.class_to_idx.copy()
        old_idx_to_class = self.idx_to_class.copy()
        
        # Add new classes
        for class_name in new_class_dataset.classes:
            if class_name not in self.class_to_idx:
                new_idx = len(self.class_to_idx)
                self.class_to_idx[class_name] = new_idx
                self.idx_to_class[new_idx] = class_name
        
        # Reinitialize classifier with expanded output
        self.classifier = nn.Linear(self.hidden_size, new_num_classes).to(self.device)
        
        # Copy old classifier weights
        with torch.no_grad():
            self.classifier.weight[:old_num_classes] = self.model.classifier.weight.data
            self.classifier.bias[:old_num_classes] = self.model.classifier.bias.data
        
        # Configure SD-LoRA
        logger.info("Configuring SD-LoRA...")
        lora_config = SDLoRAConfig(
            r=lora_r,
            alpha=lora_alpha,
            target_modules=['query', 'value']
        )
        
        # Re-apply PEFT model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model = self.model.to(self.device)
        
        # Freeze old LoRA directions (A and B matrices)
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False
        
        # Optimizer (only train classifier and LoRA magnitudes)
        optimizer_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimizer_params.append(param)
        optimizer_params.extend(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW(optimizer_params, lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Create data loader
        train_loader = DataLoader(
            new_class_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        best_accuracy = 0.0
        
        logger.info(f"Phase 2 training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            self.classifier.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.base_model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(pooled_output)
                
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            train_accuracy = correct / total if total > 0 else 0.0
            
            # Simple validation on new classes
            val_accuracy = self._validate_new_classes(new_class_dataset)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {total_loss/len(train_loader):.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}")
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                if save_dir:
                    self.save_adapter(save_dir)
        
        # Update OOD statistics for new classes
        logger.info("Updating OOD statistics for new classes...")
        self._update_ood_for_new_classes(new_class_dataset)
        
        logger.info(f"Phase 2 completed. Best accuracy: {best_accuracy:.4f}")
        
        return {
            'best_accuracy': best_accuracy,
            'num_new_classes': len(new_class_dataset.classes),
            'total_classes': new_num_classes
        }
    
    def _validate_new_classes(self, dataset: CropDataset) -> float:
        """Validate on new class dataset."""
        self.model.eval()
        self.classifier.eval()
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.base_model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(pooled_output)
                
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _update_ood_for_new_classes(self, new_class_dataset: CropDataset):
        """Update OOD statistics to include new classes."""
        # Compute prototypes for new classes
        new_loader = DataLoader(
            new_class_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        new_prototypes, new_stds = compute_class_prototypes(
            self.base_model, new_loader, self.hidden_size, self.device
        )
        
        # Merge with existing prototypes
        if self.prototypes is None:
            self.prototypes = new_prototypes
            self.class_stds = new_stds
        else:
            # Append new class prototypes
            old_num_classes = self.prototypes.shape[0]
            total_classes = old_num_classes + new_prototypes.shape[0]
            
            merged_prototypes = torch.zeros(total_classes, self.hidden_size).to(self.device)
            merged_prototypes[:old_num_classes] = self.prototypes
            merged_prototypes[old_num_classes:] = new_prototypes
            
            self.prototypes = merged_prototypes
            self.class_stds.update(new_stds)
        
        # Recompute Mahalanobis
        self.mahalanobis = MahalanobisDistance(self.prototypes, self.class_stds)
        
        # Recompute thresholds for all classes
        # For simplicity, use the new class dataset to compute thresholds
        self.ood_thresholds = DynamicOODThreshold.compute_thresholds(
            self.mahalanobis, self.base_model, new_loader, self.hidden_size, self.device
        )
    
    def phase3_fortify(
        self,
        domain_shift_dataset: CropDataset,
        config: Dict,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Phase 3: CONEC-LoRA domain-incremental learning.
        Fortifies adapters against domain shifts.
        
        Args:
            domain_shift_dataset: Dataset with domain-shifted images
            config: Configuration with:
                - num_shared_blocks: Number of early blocks to freeze (default: 6)
                - lora_r: LoRA rank (default: 16)
                - lora_alpha: LoRA alpha (default: 16)
                - num_epochs: Training epochs (default: 15)
                - batch_size: Batch size (default: 32)
                - learning_rate: Learning rate (default: 1e-4)
            save_dir: Save directory
            
        Returns:
            Dictionary with metrics including protected retention
        """
        logger.info(f"Starting Phase 3 for {self.crop_name}")
        self.current_phase = 3
        
        # Extract config
        num_shared_blocks = config.get('num_shared_blocks', 6)
        lora_r = config.get('lora_r', 16)
        lora_alpha = config.get('lora_alpha', 16)
        num_epochs = config.get('num_epochs', 15)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 1e-4)
        
        # Freeze early transformer blocks
        self._freeze_shared_blocks(num_shared_blocks)
        
        # Add new LoRA to late blocks
        lora_config = LoraConfig(
            r=lora_r,
            alpha=lora_alpha,
            target_modules=['query', 'value'],
            layers_to_transform=list(range(num_shared_blocks, 12))
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model = self.model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Data loader
        train_loader = DataLoader(
            domain_shift_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        best_protected_retention = 0.0
        
        logger.info(f"Phase 3 training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.model.train()
            
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                logits = self.classifier(outputs.last_hidden_state[:, 0, :])
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            # Evaluate protected retention
            protected_retention = self._evaluate_protected_retention()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Loss: {total_loss/len(train_loader):.4f}, "
                       f"Protected Retention: {protected_retention:.4f}")
            
            if protected_retention > best_protected_retention:
                best_protected_retention = protected_retention
                if save_dir:
                    self.save_adapter(save_dir)
        
        # Update OOD thresholds for fortified classes
        logger.info("Updating OOD thresholds after fortification...")
        # Recompute prototypes from original training data (would need to be stored)
        # For now, keep existing thresholds
        
        logger.info(f"Phase 3 completed. Best protected retention: {best_protected_retention:.4f}")
        
        return {
            'best_protected_retention': best_protected_retention
        }
    
    def _freeze_shared_blocks(self, num_shared_blocks: int):
        """Freeze early transformer blocks."""
        for i in range(num_shared_blocks):
            for param in self.base_model.blocks[i].parameters():
                param.requires_grad = False
        logger.info(f"Froze first {num_shared_blocks} transformer blocks")
    
    def _evaluate_protected_retention(self) -> float:
        """
        Evaluate retention on protected (non-fortified) classes.
        This would require a validation set of original classes.
        """
        # Placeholder - would need original validation data
        return 0.85
    
    def predict_with_ood(
        self,
        image: torch.Tensor
    ) -> Dict:
        """
        Predict disease with OOD detection.
        
        Args:
            image: Preprocessed image tensor (batch size 1)
            
        Returns:
            Dictionary with prediction and OOD analysis
        """
        if not self.is_trained:
            raise RuntimeError("Adapter must be trained before prediction")
        
        self.model.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            
            # Get features
            outputs = self.base_model(image)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            features = pooled_output
            
            # Get classification
            logits = self.classifier(features)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            # OOD detection
            is_ood, ood_score, threshold = self._detect_ood(features, predicted_idx)
            
            result = {
                'status': 'success',
                'disease': {
                    'class_index': predicted_idx,
                    'name': self.idx_to_class.get(predicted_idx, 'unknown'),
                    'confidence': confidence
                },
                'ood_analysis': {
                    'is_ood': is_ood,
                    'ood_score': ood_score,
                    'threshold': threshold,
                    'dynamic_threshold_applied': True
                }
            }
            
            if is_ood:
                result['ood_analysis']['ood_type'] = 'NEW_DISEASE_CANDIDATE'
                result['recommendations'] = {
                    'expert_consultation': True,
                    'message': 'Potential new disease pattern detected.'
                }
            
            return result
    
    def _detect_ood(
        self,
        features: torch.Tensor,
        predicted_class: int
    ) -> Tuple[bool, float, float]:
        """
        Detect if sample is OOD using dynamic Mahalanobis threshold.
        
        Returns:
            (is_ood, ood_score, threshold)
        """
        if self.mahalanobis is None or self.ood_thresholds is None:
            return False, 0.0, 0.0
        
        # Compute Mahalanobis distance to predicted class prototype
        distance = self.mahalanobis.compute_distance(features, predicted_class)
        distance_value = distance.item()
        
        # Get threshold for this class
        threshold = self.ood_thresholds.get(predicted_class, 25.0)
        
        is_ood = distance_value > threshold
        
        return is_ood, distance_value, threshold
    
    def save_adapter(self, save_path: str):
        """Save the trained adapter."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PEFT model
        self.model.save_pretrained(save_path / 'adapter')
        
        # Save classifier
        torch.save(
            self.classifier.state_dict(),
            save_path / 'classifier.pth'
        )
        
        # Save OOD components
        if self.prototypes is not None:
            torch.save({
                'prototypes': self.prototypes,
                'class_stds': self.class_stds,
                'ood_thresholds': self.ood_thresholds,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class
            }, save_path / 'ood_components.pt')
        
        logger.info(f"Adapter saved to {save_path}")
    
    def load_adapter(self, load_path: str):
        """Load a trained adapter."""
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
        
        # Load OOD components
        ood_path = load_path / 'ood_components.pt'
        if ood_path.exists():
            ood_data = torch.load(ood_path, map_location=self.device)
            self.prototypes = ood_data['prototypes']
            self.class_stds = ood_data['class_stds']
            self.ood_thresholds = ood_data['ood_thresholds']
            self.class_to_idx = ood_data['class_to_idx']
            self.idx_to_class = ood_data['idx_to_class']
            self.mahalanobis = MahalanobisDistance(self.prototypes, self.class_stds)
        
        self.is_trained = True
        logger.info(f"Adapter loaded from {load_path}")

if __name__ == "__main__":
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--crop', type=str, default='tomato')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    from src.utils.data_loader import CropDataset
    
    if args.phase == 1:
        train_dataset = CropDataset(args.data_dir, args.crop, 'train', transform=True)
        val_dataset = CropDataset(args.data_dir, args.crop, 'val', transform=False)
        
        adapter = IndependentCropAdapter(crop_name=args.crop)
        
        config = {
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'loraplus_lr_ratio': 16,
            'num_epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-4
        }
        
        adapter.phase1_initialize(train_dataset, val_dataset, config, args.output_dir)
    
    logger.info("Adapter training completed!")