#!/usr/bin/env python3
from typing import Dict, List, Optional, Any, Tuple
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)


def _resolve_style(style: str) -> str:
    """Resolve style name across matplotlib/seaborn version differences."""
    candidates = [style, "seaborn-v0_8", "seaborn", "default"]
    available = set(plt.style.available)
    for candidate in candidates:
        if candidate in available or candidate == "default":
            return candidate
    return "default"


class PlotGenerator:
    """Base class for generating plots."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100, style: str = "seaborn"):
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.set_style(style)

    def set_style(self, style: str):
        """Set plot style."""
        resolved = _resolve_style(style)
        self.style = style
        plt.style.use(resolved)

    def create_figure(self):
        """Create a new figure."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        return fig, ax

    def save_plot(
        self,
        fig_or_filepath,
        filepath: Optional[str] = None,
        bbox_inches: str = "tight",
        format: Optional[str] = None,
    ):
        """Save a plot.

        Supports both call styles:
        - save_plot("path.png")
        - save_plot(fig, "path", format="png")
        """
        if filepath is None:
            fig = plt.gcf()
            target = Path(fig_or_filepath)
        else:
            fig = fig_or_filepath
            target = Path(filepath)

        if format:
            ext = f".{str(format).lstrip('.')}"
            if target.suffix == "":
                target = target.with_suffix(ext)

        fig.savefig(str(target), bbox_inches=bbox_inches, format=format)
        plt.close(fig)
        return str(target)


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

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        normalize: bool = False,
        cmap: str = "Blues",
    ):
        """Compatibility API expected by tests."""
        matrix = np.asarray(cm, dtype=float)
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            matrix = matrix / row_sums
        return self.plot(matrix, class_names=class_names, title=title, cmap=cmap)


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

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        multiclass: bool = False,
        title: str = "ROC Curve",
    ):
        """Compute and plot ROC curve(s) from labels/scores."""
        y_true_arr = np.asarray(y_true)
        y_score_arr = np.asarray(y_score)

        if multiclass:
            fig, ax = self.create_figure()
            classes = np.unique(y_true_arr)
            for class_idx in classes:
                class_mask = (y_true_arr == class_idx).astype(int)
                fpr, tpr, _ = roc_curve(class_mask, y_score_arr[:, int(class_idx)])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"Class {class_idx} (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(title)
            ax.legend(loc="lower right")
            return fig

        fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
        roc_auc = auc(fpr, tpr)
        return self.plot(fpr, tpr, roc_auc, title=title)


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

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str = "Precision-Recall Curve",
    ):
        """Compatibility API expected by tests."""
        y_true_arr = np.asarray(y_true)
        y_score_arr = np.asarray(y_score)
        precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
        ap = average_precision_score(y_true_arr, y_score_arr)
        return self.plot(precision, recall, ap, title=title)


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

    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = "Loss Curves",
    ):
        """Plot train/validation loss over epochs."""
        fig, ax = self.create_figure()
        ax.plot(train_losses, label="Train Loss")
        if val_losses is not None:
            ax.plot(val_losses, label="Val Loss", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        return fig

    def plot_accuracy_curves(
        self,
        train_acc: List[float],
        val_acc: Optional[List[float]] = None,
        title: str = "Accuracy Curves",
    ):
        """Plot train/validation accuracy over epochs."""
        fig, ax = self.create_figure()
        ax.plot(train_acc, label="Train Accuracy")
        if val_acc is not None:
            ax.plot(val_acc, label="Val Accuracy", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        return fig

    def plot_metrics_comparison(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Metrics Comparison",
    ):
        """Plot multiple metrics on one chart for quick trend comparison."""
        fig, ax = self.create_figure()
        for name, values in metrics.items():
            ax.plot(values, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
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

    def plot_attention_heatmap(
        self,
        attention: np.ndarray,
        tokens: Optional[List[str]] = None,
        title: str = "Attention Heatmap",
    ):
        """Compatibility API for plotting a single attention matrix."""
        fig, ax = self.create_figure()
        matrix = np.asarray(attention)
        im = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
        ax.set_title(title)
        if tokens is not None:
            ticks = np.arange(len(tokens))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(tokens, rotation=45, ha="right")
            ax.set_yticklabels(tokens)
        plt.colorbar(im, ax=ax)
        return fig

    def plot_multihead_attention(
        self,
        attention: np.ndarray,
        tokens: Optional[List[str]] = None,
        title: str = "Multi-head Attention",
    ):
        """Plot average attention over heads."""
        attention_arr = np.asarray(attention)
        if attention_arr.ndim != 3:
            raise ValueError("Expected attention with shape [num_heads, seq_len, seq_len]")
        mean_attention = attention_arr.mean(axis=0)
        return self.plot_attention_heatmap(mean_attention, tokens=tokens, title=title)

    def plot_attention_rollout(
        self,
        attentions: List[np.ndarray],
        title: str = "Attention Rollout",
    ):
        """Plot rollout as the mean attention matrix across layers."""
        if not attentions:
            raise ValueError("attentions must contain at least one layer")
        stack = np.stack([np.asarray(a) for a in attentions], axis=0)
        rollout = stack.mean(axis=0)
        return self.plot_attention_heatmap(rollout, tokens=None, title=title)


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

    def generate_gradcam(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a single input."""
        activations: Dict[str, torch.Tensor] = {}
        gradients: Dict[str, torch.Tensor] = {}

        def _forward_hook(_module, _inp, out):
            activations["value"] = out

        def _backward_hook(_module, _grad_in, grad_out):
            gradients["value"] = grad_out[0]

        f_handle = target_layer.register_forward_hook(_forward_hook)
        b_handle = target_layer.register_full_backward_hook(_backward_hook)

        try:
            model.zero_grad(set_to_none=True)
            output = model(input_tensor)
            if output.ndim == 1:
                output = output.unsqueeze(0)
            if target_class is None:
                target_class = int(output.argmax(dim=1).item())
            score = output[:, target_class].sum()
            score.backward()

            acts = activations["value"]
            grads = gradients["value"]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            cam = cam[0, 0]
            cam = cam - cam.min()
            denom = cam.max().clamp(min=1e-8)
            cam = cam / denom
            return cam.detach().cpu().numpy()
        finally:
            f_handle.remove()
            b_handle.remove()

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        cmap: str = "jet",
    ) -> np.ndarray:
        """Overlay normalized heatmap on an RGB image."""
        image_arr = np.asarray(image)
        heatmap_arr = np.asarray(heatmap, dtype=float)

        if heatmap_arr.ndim != 2:
            raise ValueError("heatmap must be 2D")
        if image_arr.ndim != 3 or image_arr.shape[2] != 3:
            raise ValueError("image must have shape [H, W, 3]")

        heatmap_arr = heatmap_arr - heatmap_arr.min()
        denom = heatmap_arr.max() if heatmap_arr.max() > 0 else 1.0
        heatmap_arr = heatmap_arr / denom

        color_heatmap = plt.get_cmap(cmap)(heatmap_arr)[..., :3]
        if image_arr.dtype == np.uint8:
            base = image_arr.astype(np.float32) / 255.0
        else:
            base = image_arr.astype(np.float32)
            max_val = max(float(base.max()), 1.0)
            if max_val > 1.0:
                base = base / max_val

        overlay = (1.0 - alpha) * base + alpha * color_heatmap
        overlay = np.clip(overlay, 0.0, 1.0)
        return (overlay * 255.0).astype(np.uint8)

    def plot_gradcam_comparison(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        title: str = "Grad-CAM Comparison",
    ):
        """Plot original image, heatmap, and overlay side by side."""
        overlay = self.overlay_heatmap(image, heatmap, alpha=0.5)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        fig.suptitle(title)
        fig.tight_layout()
        return fig


def save_plot(fig, filepath: str, bbox_inches: str = 'tight'):
    """Save plot to file."""
    fig.savefig(filepath, bbox_inches=bbox_inches)
    plt.close(fig)


def set_plot_style(style: str = "seaborn"):
    """Set global plot style."""
    plt.style.use(_resolve_style(style))


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
        plt.style.use(_resolve_style('seaborn-v0_8'))
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
