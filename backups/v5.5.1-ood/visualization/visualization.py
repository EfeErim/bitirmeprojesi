#!/usr/bin/env python3
"""
Visualization Utilities for AADS-ULoRA v5.5
Provides plotting functions for training metrics, OOD analysis, and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[Path] = None
):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2 = axes[1]
    ax2.plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training plot to {save_path}")
    
    plt.show()

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[Path] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize by row
        save_path: Optional path to save figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    plt.show()

def plot_ood_analysis(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
    thresholds: Optional[float] = None,
    title: str = "OOD Score Distribution",
    save_path: Optional[Path] = None
):
    """
    Plot OOD score distributions for in-distribution and OOD samples.
    
    Args:
        in_dist_scores: OOD scores for in-distribution samples
        ood_scores: OOD scores for OOD samples
        thresholds: Optional threshold line to draw
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 5))
    
    # Plot histograms
    plt.hist(in_dist_scores, bins=50, alpha=0.6, label='In-Distribution', color='blue', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.6, label='OOD', color='red', density=True)
    
    # Plot threshold if provided
    if thresholds is not None:
        plt.axvline(thresholds, color='black', linestyle='--', linewidth=2, label=f'Threshold: {thresholds:.2f}')
    
    plt.xlabel('OOD Score (Mahalanobis Distance)')
    plt.ylabel('Density')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved OOD analysis plot to {save_path}")
    
    plt.show()

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None
):
    """
    Plot ROC curve for OOD detection.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: Area under ROC curve
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
    
    plt.show()

def plot_prototype_quality(
    prototypes: torch.Tensor,
    class_stds: Dict[int, torch.Tensor],
    class_names: List[str],
    title: str = "Class Prototypes (PCA)",
    save_path: Optional[Path] = None
):
    """
    Plot class prototypes in 2D using PCA.
    
    Args:
        prototypes: Tensor of shape (num_classes, feature_dim)
        class_stds: Dictionary of class stds
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save figure
    """
    from sklearn.decomposition import PCA
    
    # Convert to numpy
    prototypes_np = prototypes.cpu().numpy()
    
    # Apply PCA
    if prototypes_np.shape[1] > 2:
        pca = PCA(n_components=2)
        prototypes_2d = pca.fit_transform(prototypes_np)
    else:
        prototypes_2d = prototypes_np
    
    plt.figure(figsize=(10, 8))
    
    # Plot prototypes
    scatter = plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], s=200, c=range(len(class_names)), cmap='tab20')
    
    # Add labels
    for i, (x, y) in enumerate(prototypes_2d):
        plt.text(x, y, class_names[i], fontsize=10, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved prototype plot to {save_path}")
    
    plt.show()

def plot_retention_analysis(
    retention_scores: Dict[str, float],
    title: str = "Retention Analysis",
    save_path: Optional[Path] = None
):
    """
    Plot retention scores for continual learning phases.
    
    Args:
        retention_scores: Dictionary with retention metrics
        title: Plot title
        save_path: Optional path to save figure
    """
    labels = []
    values = []
    
    for key, value in retention_scores.items():
        if 'retention' in key.lower() or 'accuracy' in key.lower():
            labels.append(key.replace('_', ' ').title())
            values.append(value * 100)  # Convert to percentage
    
    plt.figure(figsize=(8, 5))
    bars = plt.barh(labels, values, color=sns.color_palette("husl", len(labels)))
    
    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(val + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=10)
    
    plt.xlabel('Score (%)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, 105)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved retention plot to {save_path}")
    
    plt.show()

def create_evaluation_report(
    metrics: Dict,
    output_dir: Path,
    crop_name: str
):
    """
    Create comprehensive visualization report for a crop adapter.
    
    Args:
        metrics: Dictionary with all evaluation metrics
        output_dir: Directory to save plots
        crop_name: Name of the crop
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating evaluation report for {crop_name}")
    
    # Plot training history if available
    if 'history' in metrics:
        plot_training_history(
            metrics['history'],
            title=f"{crop_name} - Training History",
            save_path=output_dir / 'training_history.png'
        )
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in metrics and 'class_names' in metrics:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            metrics['class_names'],
            title=f"{crop_name} - Confusion Matrix",
            normalize=True,
            save_path=output_dir / 'confusion_matrix.png'
        )
    
    # Plot OOD analysis if available
    if 'in_dist_scores' in metrics and 'ood_scores' in metrics:
        plot_ood_analysis(
            metrics['in_dist_scores'],
            metrics['ood_scores'],
            thresholds=metrics.get('threshold'),
            title=f"{crop_name} - OOD Analysis",
            save_path=output_dir / 'ood_analysis.png'
        )
    
    # Plot ROC curve if available
    if 'fpr' in metrics and 'tpr' in metrics:
        plot_roc_curve(
            metrics['fpr'],
            metrics['tpr'],
            metrics.get('auroc', 0.0),
            title=f"{crop_name} - OOD Detection ROC",
            save_path=output_dir / 'roc_curve.png'
        )
    
    # Plot retention if available
    retention_metrics = {k: v for k, v in metrics.items() if 'retention' in k.lower() or 'accuracy' in k.lower()}
    if retention_metrics:
        plot_retention_analysis(
            retention_metrics,
            title=f"{crop_name} - Performance Metrics",
            save_path=output_dir / 'retention_analysis.png'
        )
    
    logger.info(f"Evaluation report saved to {output_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy data
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.25],
        'train_accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.65, 0.75, 0.8, 0.85, 0.88]
    }
    
    plot_training_history(history, "Test Training History")