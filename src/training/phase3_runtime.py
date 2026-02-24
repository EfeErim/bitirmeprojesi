#!/usr/bin/env python3
"""Runtime helpers for Phase 3 CoNeC trainer.

These helpers keep ColabPhase3Trainer's public API stable while
moving large training/validation/checkpoint routines out of the main module.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict
import gc
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def phase3_training_step(trainer, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    trainer.model.train()
    trainer.classifier.train()

    images = batch['images'].to(trainer.device)
    labels = batch['labels'].to(trainer.device)

    with torch.amp.autocast('cuda', enabled=trainer.use_amp):
        pooled = trainer.extract_pooled_output(trainer.model, images)

        if trainer.classifier.in_features != pooled.shape[1]:
            logger.warning(
                f"Classifier input mismatch ({trainer.classifier.in_features} != {pooled.shape[1]}), rebuilding classifier"
            )
            trainer.classifier = nn.Linear(pooled.shape[1], trainer.classifier.out_features).to(trainer.device)
            trainer.setup_optimizer()

        logits = trainer.classifier(pooled)
        classification_loss = nn.CrossEntropyLoss()(logits, labels)
        conec_loss, _, _ = trainer._compute_conec_loss(pooled, labels)

    return classification_loss + conec_loss


def phase3_train_epoch(trainer, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
    trainer.model.train()
    trainer.classifier.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_orthogonal_loss = 0.0
    num_batches = 0
    accumulation_steps = max(1, int(trainer.gradient_accumulation_steps))

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.current_step = 0

    try:
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(trainer.device)
            labels = batch['labels'].to(trainer.device)

            if batch_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            with torch.amp.autocast('cuda', enabled=trainer.use_amp):
                pooled = trainer.extract_pooled_output(trainer.model, images)
                conec_loss, contrastive_loss, orthogonal_loss = trainer._compute_conec_loss(pooled, labels)
                logits = trainer.classifier(pooled)
                classification_loss = nn.CrossEntropyLoss()(logits, labels)
                total_batch_loss = conec_loss + classification_loss

            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                logger.error(
                    f"NaN/Inf loss detected at batch {batch_idx}, epoch {epoch}: {total_batch_loss.item()}"
                )
                raise RuntimeError("Training diverged - loss is NaN/Inf. Check gradients and loss scales.")

            trainer.scaler.scale(total_batch_loss).backward()

            trainer.current_step += 1
            if trainer.current_step % accumulation_steps == 0:
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(trainer.classifier.parameters(), max_norm=1.0)
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                trainer.optimizer.zero_grad(set_to_none=True)

            if batch_idx % 50 == 0:
                trainer._update_prototypes(pooled, labels)

            if batch_idx % 100 == 0:
                ood_metrics = trainer._perform_ood_detection(pooled, labels)
                trainer.history['ood_metrics'].append(ood_metrics)

            total_loss += total_batch_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_orthogonal_loss += orthogonal_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss={total_batch_loss.item():.4f}, "
                    f"Contrastive={contrastive_loss.item():.4f}, "
                    f"Orthogonal={orthogonal_loss.item():.4f}"
                )

            trainer._log_memory_usage()

        if trainer.current_step % accumulation_steps != 0:
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(trainer.classifier.parameters(), max_norm=1.0)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad(set_to_none=True)
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.current_step = 0

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0.0
    avg_orthogonal_loss = total_orthogonal_loss / num_batches if num_batches > 0 else 0.0

    return {
        'loss': avg_loss,
        'contrastive_loss': avg_contrastive_loss,
        'orthogonal_loss': avg_orthogonal_loss,
    }


def phase3_validate(trainer, val_loader: DataLoader) -> Dict[str, float]:
    prev_model_mode = trainer.model.training
    prev_classifier_mode = trainer.classifier.training
    trainer.model.eval()
    trainer.classifier.eval()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_orthogonal_loss = 0.0
    all_preds = []
    all_labels = []

    try:
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(trainer.device)
                labels = batch['labels'].to(trainer.device)

                pooled = trainer.extract_pooled_output(trainer.model, images)

                if trainer.classifier.in_features != pooled.shape[1]:
                    logger.warning("Classifier input mismatch in validation, rebuilding classifier")
                    trainer.classifier = nn.Linear(pooled.shape[1], trainer.classifier.out_features).to(trainer.device)
                    trainer.setup_optimizer()

                conec_loss, contrastive_loss, orthogonal_loss = trainer._compute_conec_loss(pooled, labels)
                logits = trainer.classifier(pooled)
                classification_loss = nn.CrossEntropyLoss()(logits, labels)
                total_batch_loss = conec_loss + classification_loss

                total_loss += total_batch_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_orthogonal_loss += orthogonal_loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = np.mean(all_preds == all_labels)

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_contrastive_loss = total_contrastive_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_orthogonal_loss = total_orthogonal_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        metrics = {
            'loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'orthogonal_loss': avg_orthogonal_loss,
            'accuracy': float(accuracy),
            'num_samples': len(all_labels),
        }
    finally:
        trainer.model.train(prev_model_mode)
        trainer.classifier.train(prev_classifier_mode)

    return metrics


def phase3_save_checkpoint(trainer, path: str, epoch: int, loss: float):
    path_obj = Path(path)
    if path_obj.suffix:
        save_path = path_obj.parent
        filename = path_obj.name
    else:
        save_path = path_obj
        filename = f'checkpoint_epoch_{epoch}.pth'

    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': trainer.model.state_dict(),
        'classifier_state_dict': trainer.classifier.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scaler_state_dict': trainer.scaler.state_dict(),
        'prototype_embeddings': trainer.prototype_manager.get_prototypes(),
        'config': asdict(trainer.config),
        'history': trainer.history,
    }

    checkpoint_path = save_path / filename
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def phase3_load_checkpoint(trainer, path: str):
    checkpoint = torch.load(path, map_location=trainer.device)

    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    trainer.prototype_manager.set_prototypes(checkpoint['prototype_embeddings'])
    trainer.history = checkpoint['history']

    trainer.current_epoch = checkpoint['epoch']
    trainer.best_val_loss = checkpoint['loss']

    logger.info(f"Checkpoint loaded from: {path}")
    logger.info(f"Resuming from epoch {trainer.current_epoch + 1}")
