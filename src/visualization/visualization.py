#!/usr/bin/env python3
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path


class PlotGenerator:
    """Base class for generating plots."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100, style: str = "seaborn"):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.set_style(style)

    def set_style(self, style: str):
        """Set plot style."""
        self.style = style
        plt.style.use(style)

    def create_figure(self):
        """Create a new figure."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        return fig, ax

    def save_plot(self, filepath: str, bbox_inches: str = 'tight'):
        """Save plot to file."""
        plt.savefig(filepath, bbox_inches=bbox_inches)
        plt.close()


class ConfusionMatrixPlotter(PlotGenerator):
    """Plot confusion matrix."""

    def plot(self, confusion_matrix: np.ndarray, class_names: List[str] = None,
             title: str = "Confusion Matrix", cmap: str = "Blues"):
        """Plot confusion matrix."""
        fig, ax = self.create_figure()
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        if class_names:
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax)
        return fig


class ROCPlotter(PlotGenerator):
    """Plot ROC curves."""

    def plot(self, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
             title: str = "ROC Curve"):
        """Plot ROC curve."""
        fig, ax = self.create_figure()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        return fig


class PrecisionRecallPlotter(PlotGenerator):
    """Plot precision-recall curves."""

    def plot(self, precision: np.ndarray, recall: np.ndarray, ap: float,
             title: str = "Precision-Recall Curve"):
        """Plot precision-recall curve."""
        fig, ax = self.create_figure()
        ax.plot(recall, precision, color='green', lw=2, label=f'AP = {ap:.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        return fig


class TrainingCurvePlotter(PlotGenerator):
    """Plot training curves."""

    def plot(self, train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]],
             title: str = "Training Curves"):
        """Plot training and validation metrics."""
        fig, ax = self.create_figure()
        for metric in train_metrics:
            ax.plot(train_metrics[metric], label=f'Train {metric}')
        for metric in val_metrics:
            ax.plot(val_metrics[metric], label=f'Val {metric}', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        return fig


class AttentionVisualizer(PlotGenerator):
    """Visualize attention maps."""

    def plot_attention(self, attention_weights: np.ndarray, title: str = "Attention Map"):
        """Plot attention weights."""
        fig, ax = self.create_figure()
        im = ax.imshow(attention_weights, cmap='hot', interpolation='nearest')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        return fig


class GradCAMVisualizer(PlotGenerator):
    """Visualize GradCAM heatmaps."""

    def plot_heatmap(self, heatmap: np.ndarray, original_image: np.ndarray = None,
                     title: str = "GradCAM Heatmap", alpha: float = 0.5):
        """Plot GradCAM heatmap over original image."""
        fig, ax = self.create_figure()
        if original_image is not None:
            ax.imshow(original_image)
            ax.imshow(heatmap, cmap='jet', alpha=alpha)
        else:
            ax.imshow(heatmap, cmap='jet')
        ax.set_title(title)
        ax.axis('off')
        return fig


def save_plot(fig, filepath: str, bbox_inches: str = 'tight'):
    """Save plot to file."""
    fig.savefig(filepath, bbox_inches=bbox_inches)
    plt.close(fig)


def set_plot_style(style: str = "seaborn"):
    """Set global plot style."""
    plt.style.use(style)


def get_color_palette(palette_or_n: object = 'husl', n_colors: int = None):
    """Return a color palette list of RGB tuples.

    Accepts either (n_colors, palette) or (palette, n_colors) calling styles
    for backwards compatibility with tests.
    """
    # If first arg is int, treat as n_colors
    if isinstance(palette_or_n, int):
        n = int(palette_or_n)
        pal = 'husl' if n_colors is None else n_colors
        # If pal is int it means user passed two ints; fallback to default
        if isinstance(pal, int):
            pal = 'husl'
        return sns.color_palette(pal, n)

    # Otherwise first arg is palette name
    pal = str(palette_or_n)
    n = n_colors if n_colors is not None else 8
    return sns.color_palette(pal, n)


def get_colormap(name: str = 'viridis'):
    """Return a matplotlib colormap by name."""
    return plt.get_cmap(name)


"""
Visualization Module for AADS-ULoRA v5.5
Generates training/evaluation plots and diagnostic visuals.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class Visualizer:
    """Produces standardized visualizations for model analysis"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure style
        plt.style.use('seaborn')
        self.palette = sns.color_palette('husl', 8)
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })

    def plot_training_curves(self, history: Dict[str, List[float]], title: str):
        """Plot loss and accuracy curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train', color=self.palette[0])
        plt.plot(history['val_loss'], label='Validation', color=self.palette[1])
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train', color=self.palette[0])
        plt.plot(history['val_accuracy'], label='Validation', color=self.palette[1])
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.suptitle(title)
        plt.savefig(self.output_dir / f'{title}_curves.png', bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], title: str):
        """Plot normalized confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {title}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.output_dir / f'{title}_cm.png', bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float, title: str):
        """Plot ROC curve with AUC"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=self.palette[2], 
                label=f'ROC Curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve: {title}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.savefig(self.output_dir / f'{title}_roc.png', bbox_inches='tight')
        plt.close()

    def plot_retention_comparison(self, metrics: Dict[str, float], title: str):
        """Bar plot comparing retention metrics"""
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), 
                   palette=self.palette[3:5])
        plt.title(f'Retention Analysis: {title}')
        plt.ylabel('Value')
        plt.ylim(0, 1)
        plt.savefig(self.output_dir / f'{title}_retention.png', bbox_inches='tight')
        plt.close()

    def plot_ood_detection(self, in_scores: np.ndarray, ood_scores: np.ndarray, title: str):
        """Histogram of OOD detection scores"""
        plt.figure(figsize=(8, 5))
        sns.histplot(in_scores, color=self.palette[0], label='In-Distribution', kde=True)
        sns.histplot(ood_scores, color=self.palette[1], label='OOD', kde=True)
        plt.title(f'OOD Detection: {title}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(self.output_dir / f'{title}_ood.png', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    # Example usage
    vis = Visualizer()
    
    # Test training curves
    history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3],
        'val_loss': [0.7, 0.5, 0.45, 0.35],
        'train_accuracy': [0.65, 0.75, 0.82, 0.88],
        'val_accuracy': [0.63, 0.72, 0.79, 0.85]
    }
    vis.plot_training_curves(history, 'Phase1_Tomato')
    
    # Test confusion matrix
    cm = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]])
    vis.plot_confusion_matrix(cm, ['Healthy', 'Disease1', 'Disease2'], 'Tomato_Phase1')
    
    # Test ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)
    vis.plot_roc_curve(fpr, tpr, 0.92, 'OOD_Detection')
    
    # Test retention
    retention_metrics = {'Old Accuracy': 0.85, 'New Accuracy': 0.82, 'Retention Rate': 0.96}
    vis.plot_retention_comparison(retention_metrics, 'Phase2_Retention')
    
    # Test OOD detection
    in_scores = np.random.normal(0.2, 0.1, 1000)
    ood_scores = np.random.normal(0.7, 0.2, 800)
    vis.plot_ood_detection(in_scores, ood_scores, 'Tomato_OOD')