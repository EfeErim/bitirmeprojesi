#!/usr/bin/env python3
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