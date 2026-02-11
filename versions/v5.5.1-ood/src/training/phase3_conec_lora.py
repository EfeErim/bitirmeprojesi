#!/usr/bin/env python3
"""
Phase 3 Training: CONEC-LoRA for Domain-Incremental Learning
Fortifies adapters against domain shifts using CONEC-LoRA technique.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
import logging
from pathlib import Path
from typing import Dict, Optional

from src.utils.data_loader import DomainShiftDataset
from src.utils.metrics import compute_protected_retention

logger = logging.getLogger(__name__)

class Phase3Trainer:
    """
    Phase 3 trainer for CONEC-LoRA domain adaptation.
    
    Args:
        adapter_path: Path to existing adapter checkpoint
        num_shared_blocks: Number of transformer blocks to freeze (default: 6)
        lora_r: LoRA rank for new adapters (default: 16)
        lora_alpha: LoRA alpha (default: 16)
        device: Device for training
    """
    
    def __init__(
        self,
        adapter_path: str,
        num_shared_blocks: int = 6,
        lora_r: int = 16,
        lora_alpha: int = 16,
        gradient_accumulation_steps: int = 1,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_shared_blocks = num_shared_blocks
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Mixed precision training
        self.use_amp = torch.cuda.is_available() and torch.backends.cuda.is_built()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 1
        self.current_step = 0
        
        # Load existing adapter
        logger.info(f"Loading adapter from: {adapter_path}")
        self.base_model = AutoModel.from_pretrained(adapter_path)
        self.classifier = nn.Linear(self.base_model.config.hidden_size,
                                  self.base_model.classifier.out_features)
        
        # Freeze shared blocks
        self._freeze_shared_blocks()
        
        # Configure CONEC-LoRA
        logger.info("Configuring CONEC-LoRA adapter...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['query', 'value'],
            layers_to_transform=list(range(num_shared_blocks, 12))  # Adapt only late blocks
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        
        logger.info(f"Phase3Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def _freeze_shared_blocks(self):
        """Freeze early transformer blocks for shared features."""
        for i in range(self.num_shared_blocks):
            for param in self.base_model.blocks[i].parameters():
                param.requires_grad = False
        logger.info(f"Froze first {self.num_shared_blocks} transformer blocks")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with mixed precision and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                logits = self.classifier(outputs.last_hidden_state[:, 0, :])
                
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
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        return {'loss': total_loss / len(train_loader)}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model performance."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                logits = self.classifier(outputs.last_hidden_state[:, 0, :])
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = compute_protected_retention(np.array(all_preds), np.array(all_labels))
        metrics['loss'] = total_loss / len(val_loader)
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int = 15, save_dir: Optional[str] = None) -> Dict:
        """Main training loop."""
        history = {'train_loss': [], 'val_loss': [], 'protected_retention': []}
        best_retention = 0.0
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['protected_retention'].append(val_metrics['protected_retention'])
            
            if val_metrics['protected_retention'] > best_retention:
                best_retention = val_metrics['protected_retention']
                if save_dir:
                    self.save_adapter(save_dir)
        
        logger.info(f"Best protected retention: {best_retention:.4f}")
        return history

    def save_adapter(self, save_path: str):
        """Save fortified adapter."""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        logger.info(f"CONEC-LoRA adapter saved to {save_path}")


def main():
    """Example usage."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_path', required=True)
    parser.add_argument('--domain_shift_dir', required=True)
    parser.add_argument('--output_dir', default='./outputs/phase3')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load domain-shifted data
    train_set = DomainShiftDataset(args.domain_shift_dir, split='train')
    val_set = DomainShiftDataset(args.domain_shift_dir, split='val')
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    
    # Initialize and train
    trainer = Phase3Trainer(args.adapter_path)
    history = trainer.train(train_loader, val_loader, args.epochs, args.output_dir)
    
    logger.info("Phase 3 training completed")

if __name__ == "__main__":
    main()