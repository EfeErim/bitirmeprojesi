#!/usr/bin/env python3
"""
Evaluation Metrics for AADS-ULoRA v5.5
Implements comprehensive metrics for multi-crop disease detection system.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from typing import Dict, List, Optional
import logging
import torch

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Classification metrics calculator."""
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy."""
        return accuracy_score(y_true, y_pred)
    
    def precision_recall_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """Compute precision, recall, F1."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: Optional[int] = None
    ) -> np.ndarray:
        """Compute confusion matrix with optional explicit class count."""
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        labels = list(range(int(num_classes))) if num_classes is not None else None
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return torch.as_tensor(cm)
    
    def per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """Compute per-class metrics."""
        cm = self.confusion_matrix(y_true, y_pred, num_classes=num_classes)
        cm = cm.to(torch.float32) if torch.is_tensor(cm) else torch.as_tensor(cm, dtype=torch.float32)
        per_class: List[Dict[str, float]] = []
        
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class.append({
                'class_index': int(i),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(cm[i, :].sum())
            })
        
        return per_class


class SegmentationMetrics:
    """Segmentation metrics calculator."""
    
    def iou(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: Optional[int] = None) -> float:
        """Compute Intersection over Union (IoU)."""
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()

        if num_classes is None:
            labels = np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)]))
            num_classes = int(labels.max()) + 1 if labels.size > 0 else 0

        ious = []
        for cls in range(int(num_classes)):
            intersection = ((y_true == cls) & (y_pred == cls)).sum()
            union = ((y_true == cls) | (y_pred == cls)).sum()
            if union > 0:
                ious.append(intersection / union)
        return float(np.mean(ious)) if ious else 0.0
    
    def dice_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Dice coefficient."""
        intersection = ((y_true == 1) & (y_pred == 1)).sum()
        total = y_true.sum() + y_pred.sum()
        return float(2 * intersection / total) if total > 0 else 0.0
    
    def mean_iou(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        """Compute mean IoU across all classes."""
        return self.iou(y_true, y_pred, num_classes)
    
    def pixel_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute pixel-wise accuracy."""
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu()

        eq = (y_true == y_pred)
        if torch.is_tensor(eq):
            return float(eq.float().mean().item())
        return float(np.mean(eq))


class DetectionMetrics:
    """Object detection metrics calculator."""
    
    def iou_bbox(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        # box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def bbox_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Backward-compatible alias for iou_bbox."""
        return self.iou_bbox(box1, box2)

    def compute_ap(
        self,
        pred_boxes,
        pred_scores,
        gt_boxes,
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute a simplified AP/precision/recall tuple for one class."""
        if torch.is_tensor(pred_boxes):
            pred_boxes = pred_boxes.detach().cpu().numpy()
        if torch.is_tensor(pred_scores):
            pred_scores = pred_scores.detach().cpu().numpy()
        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.detach().cpu().numpy()

        if len(pred_boxes) == 0:
            recall = 0.0 if len(gt_boxes) > 0 else 1.0
            return {"ap": 0.0, "precision": 0.0, "recall": recall}

        order = np.argsort(-np.asarray(pred_scores))
        pred_boxes = np.asarray(pred_boxes)[order]
        gt_boxes = np.asarray(gt_boxes)

        matched_gt = set()
        tp = 0
        fp = 0

        for pred in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = self.iou_bbox(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = max(len(gt_boxes) - tp, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ap = precision * recall
        return {"ap": float(ap), "precision": float(precision), "recall": float(recall)}

    def compute_map(self, all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold: float = 0.5) -> float:
        """Compute mAP across classes."""
        aps = []
        for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
            aps.append(self.compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold)["ap"])
        return float(np.mean(aps)) if aps else 0.0
    
    def precision_recall_at_iou(
        self,
        detections: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute precision and recall at given IoU threshold."""
        # Simplified implementation
        tp = 0
        fp = 0
        fn = len(ground_truth)
        
        for det in detections:
            matched = False
            for gt in ground_truth:
                iou = self.iou_bbox(det['bbox'], gt['bbox'])
                if iou >= iou_threshold and det['class'] == gt['class']:
                    tp += 1
                    matched = True
                    break
            if not matched:
                fp += 1
        
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall}
    
    def map(self, detections: List[Dict], ground_truth: List[Dict]) -> float:
        """Compute mean Average Precision (mAP)."""
        # Simplified: compute AP at single IoU threshold
        pr_data = self.precision_recall_at_iou(detections, ground_truth)
        return pr_data['precision']


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return ClassificationMetrics().accuracy(y_true, y_pred)


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> Dict[str, float]:
    """Compute precision, recall, F1."""
    return ClassificationMetrics().precision_recall_f1(y_true, y_pred, average)


def compute_ap(detections, ground_truth) -> float:
    """Compute Average Precision.

    Supports:
    - Detection-style input: list[dict] detections + ground-truth boxes.
    - Classification-style input: score tensor/array + binary labels.
    """
    if torch.is_tensor(detections) or isinstance(detections, np.ndarray):
        scores = detections.detach().cpu().numpy() if torch.is_tensor(detections) else np.asarray(detections)
        labels = ground_truth.detach().cpu().numpy() if torch.is_tensor(ground_truth) else np.asarray(ground_truth)
        try:
            return float(average_precision_score(labels, scores))
        except Exception:
            return 0.0
    return DetectionMetrics().map(detections, ground_truth)


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Backward-compatible support for callers that pass (scores, labels).
    true_set = set(np.unique(y_true).tolist())
    score_set = set(np.unique(y_scores).tolist())
    is_true_binary = true_set.issubset({0, 1})
    is_score_binary = score_set.issubset({0, 1})
    if not is_true_binary and is_score_binary:
        y_true, y_scores = y_scores, y_true

    return roc_auc_score(y_true, y_scores)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Predicted class indices
        labels: True class indices
        num_classes: Number of classes (optional)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(labels, predictions)
    
    # Per-class precision, recall, f1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Macro averages
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    
    # Weighted averages
    metrics['weighted_precision'] = np.average(precision, weights=support)
    metrics['weighted_recall'] = np.average(recall, weights=support)
    metrics['weighted_f1'] = np.average(f1, weights=support)
    
    # Store per-class metrics
    if num_classes is not None:
        metrics['per_class'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }
    
    return metrics

def compute_retention_rate(
    old_predictions: np.ndarray,
    old_labels: np.ndarray,
    new_predictions: np.ndarray,
    new_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute retention rate for continual learning scenarios.
    
    Args:
        old_predictions: Predictions before adding new classes
        old_labels: True labels before addition
        new_predictions: Predictions after adding new classes
        new_labels: True labels after addition (same as old_labels for old classes)
        
    Returns:
        Dictionary with retention metrics
    """
    # Ensure same length
    if not (len(old_predictions) == len(new_predictions) == len(old_labels) == len(new_labels)):
        raise ValueError("old/new predictions and labels must have identical lengths")
    
    # Compute old accuracy before and after
    old_accuracy = accuracy_score(old_labels, old_predictions)
    new_accuracy = accuracy_score(old_labels, new_predictions)
    
    # Retention rate (percentage of old performance maintained)
    retention = new_accuracy / old_accuracy if old_accuracy > 0 else 0.0
    
    metrics = {
        'old_accuracy': old_accuracy,
        'new_accuracy': new_accuracy,
        'retention_rate': retention,
        'absolute_drop': old_accuracy - new_accuracy
    }
    
    return metrics

def compute_protected_retention(
    predictions: np.ndarray,
    labels: np.ndarray,
    protected_classes: List[int]
) -> Dict[str, float]:
    """
    Compute retention for protected (non-fortified) classes in Phase 3.
    
    Args:
        predictions: Model predictions after fortification
        labels: True labels
        protected_classes: List of class indices that should be protected
        
    Returns:
        Dictionary with protected retention metrics
    """
    # Filter to protected classes only
    mask = np.isin(labels, protected_classes)
    if not mask.any():
        logger.warning("No samples from protected classes found")
        return {'protected_retention': 0.0, 'protected_accuracy': 0.0}
    
    protected_preds = predictions[mask]
    protected_labels = labels[mask]
    
    protected_accuracy = accuracy_score(protected_labels, protected_preds)
    
    metrics = {
        'protected_retention': protected_accuracy,  # For Phase 3, this is the retention
        'protected_accuracy': protected_accuracy,
        'protected_samples': int(mask.sum())
    }
    
    return metrics

def compute_ood_metrics(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray,
    method: str = 'mahalanobis'
) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Args:
        in_dist_scores: OOD scores for in-distribution samples (lower is better for distance-based)
        ood_scores: OOD scores for out-of-distribution samples
        method: OOD detection method ('mahalanobis' or 'probability')
        
    Returns:
        Dictionary with OOD metrics
    """
    # Combine scores
    all_scores = np.concatenate([in_dist_scores, ood_scores])
    # Labels: 0 for in-distribution, 1 for ood
    all_labels = np.concatenate([
        np.zeros(len(in_dist_scores)),
        np.ones(len(ood_scores))
    ])
    
    # For distance-based methods, higher distance = more OOD
    # So we can directly use scores
    # For probability-based, we might need to invert (1 - confidence)
    
    # Compute AUROC
    auroc = roc_auc_score(all_labels, all_scores)
    
    # Compute AUPR (Area Under Precision-Recall)
    from sklearn.metrics import average_precision_score
    aupr = average_precision_score(all_labels, all_scores)
    
    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute metrics at optimal threshold
    predictions_at_optimal = (all_scores >= optimal_threshold).astype(int)
    
    # True Positive Rate (TPR) = Recall
    tpr_opt = tpr[optimal_idx]
    # False Positive Rate (FPR)
    fpr_opt = fpr[optimal_idx]
    
    metrics = {
        'auroc': auroc,
        'aupr': aupr,
        'optimal_threshold': optimal_threshold,
        'tpr_at_optimal': tpr_opt,
        'fpr_at_optimal': fpr_opt,
        'detection_rate': tpr_opt  # Same as TPR
    }
    
    return metrics

def compute_dynamic_threshold_quality(
    distances: np.ndarray,
    labels: np.ndarray,
    class_thresholds: Dict[int, float],
    class_means: np.ndarray,
    class_stds: Dict[int, np.ndarray]
) -> Dict[str, float]:
    """
    Evaluate quality of dynamic thresholds.
    
    Args:
        distances: Mahalanobis distances for samples
        labels: True class labels
        class_thresholds: Per-class threshold values
        class_means: Class mean vectors
        class_stds: Class std vectors
        
    Returns:
        Dictionary with threshold quality metrics
    """
    # For each class, compute how many samples are correctly classified as in-distribution
    class_metrics = {}
    
    for class_idx in np.unique(labels):
        class_mask = (labels == class_idx)
        class_distances = distances[class_mask]
        threshold = class_thresholds.get(class_idx, 0.0)
        
        if len(class_distances) == 0:
            continue
        
        # Percentage of in-class samples below threshold (should be high, e.g., 95%)
        in_threshold_rate = (class_distances <= threshold).mean()
        
        class_metrics[class_idx] = {
            'in_threshold_rate': in_threshold_rate,
            'threshold': threshold,
            'mean_distance': class_distances.mean(),
            'std_distance': class_distances.std(),
            'num_samples': len(class_distances)
        }
    
    # Overall metrics
    overall_in_rate = np.mean([m['in_threshold_rate'] for m in class_metrics.values()])
    
    metrics = {
        'overall_in_threshold_rate': overall_in_rate,
        'per_class': class_metrics
    }
    
    return metrics

def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    normalize: str = 'true'
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(labels, predictions, normalize=normalize)
    return cm

def compute_roc_curve_data(
    scores: np.ndarray,
    labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve data for plotting.
    
    Returns:
        Dictionary with fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

def compute_multi_crop_metrics(
    crop_results: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple crops.
    
    Args:
        crop_results: Dictionary mapping crop names to their metrics
        
    Returns:
        Aggregated metrics
    """
    all_accuracies = []
    all_retentions = []
    
    for crop, metrics in crop_results.items():
        if 'accuracy' in metrics:
            all_accuracies.append(metrics['accuracy'])
        if 'retention_rate' in metrics:
            all_retentions.append(metrics['retention_rate'])
    
    aggregated = {
        'mean_accuracy': np.mean(all_accuracies) if all_accuracies else 0.0,
        'min_accuracy': np.min(all_accuracies) if all_accuracies else 0.0,
        'max_accuracy': np.max(all_accuracies) if all_accuracies else 0.0,
        'mean_retention': np.mean(all_retentions) if all_retentions else 0.0,
        'min_retention': np.min(all_retentions) if all_retentions else 0.0
    }
    
    return aggregated

def evaluate_phase1(
    predictions: np.ndarray,
    labels: np.ndarray,
    prototypes: torch.Tensor,
    features: torch.Tensor
) -> Dict[str, float]:
    """
    Comprehensive Phase 1 evaluation.
    
    Args:
        predictions: Model predictions
        labels: True labels
        prototypes: Class prototype vectors
        features: Feature vectors from model
        
    Returns:
        Dictionary with all Phase 1 metrics
    """
    metrics = compute_metrics(predictions, labels)
    
    # Add feature quality metrics
    if prototypes is not None and features is not None:
        # Compute average within-class distance to prototype
        within_class_distances = []
        for class_idx in range(len(prototypes)):
            class_mask = (labels == class_idx)
            if class_mask.any():
                class_features = features[class_mask]
                prototype = prototypes[class_idx]
                distances = torch.norm(class_features - prototype, dim=1)
                within_class_distances.append(distances.mean().item())
        
        if within_class_distances:
            metrics['avg_within_class_distance'] = np.mean(within_class_distances)
    
    return metrics

def evaluate_phase2(
    old_predictions: np.ndarray,
    old_labels: np.ndarray,
    new_predictions: np.ndarray,
    new_labels: np.ndarray,
    old_classes: List[int],
    new_classes: List[int]
) -> Dict[str, float]:
    """
    Comprehensive Phase 2 evaluation (CIL).
    
    Returns:
        Dictionary with Phase 2 metrics including retention
    """
    # Overall accuracy on all classes
    overall_accuracy = accuracy_score(new_labels, new_predictions)
    
    # Retention on old classes
    old_mask = np.isin(new_labels, old_classes)
    if old_mask.any():
        old_retention_metrics = compute_retention_rate(
            old_predictions[old_mask],
            old_labels[old_mask],
            new_predictions[old_mask],
            new_labels[old_mask]
        )
    else:
        old_retention_metrics = {'retention_rate': 0.0}
    
    # Accuracy on new classes
    new_mask = np.isin(new_labels, new_classes)
    if new_mask.any():
        new_accuracy = accuracy_score(new_labels[new_mask], new_predictions[new_mask])
    else:
        new_accuracy = 0.0
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'old_class_retention': old_retention_metrics['retention_rate'],
        'new_class_accuracy': new_accuracy,
        **old_retention_metrics
    }
    
    return metrics

def evaluate_phase3(
    predictions: np.ndarray,
    labels: np.ndarray,
    protected_classes: List[int],
    fortified_classes: List[int]
) -> Dict[str, float]:
    """
    Comprehensive Phase 3 evaluation (DIL).
    
    Returns:
        Dictionary with Phase 3 metrics including protected retention
    """
    # Overall accuracy
    overall_accuracy = accuracy_score(labels, predictions)
    
    # Protected retention
    protected_metrics = compute_protected_retention(predictions, labels, protected_classes)
    
    # Fortified class accuracy
    fortified_mask = np.isin(labels, fortified_classes)
    if fortified_mask.any():
        fortified_accuracy = accuracy_score(labels[fortified_mask], predictions[fortified_mask])
    else:
        fortified_accuracy = 0.0
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'protected_retention': protected_metrics['protected_retention'],
        'fortified_accuracy': fortified_accuracy,
        **protected_metrics
    }
    
    return metrics

class MetricsTracker:
    """
    Track metrics across training epochs.
    """
    
    def __init__(self):
        self.history = {
            'train': {},
            'val': {}
        }
    
    def update(self, split: str, metrics: Dict[str, float], epoch: int):
        """Update metrics for a given epoch."""
        if split not in self.history:
            self.history[split] = {}
        
        for key, value in metrics.items():
            if key not in self.history[split]:
                self.history[split][key] = []
            self.history[split][key].append((epoch, value))
    
    def get_best_epoch(self, metric: str, split: str = 'val', mode: str = 'max') -> int:
        """Get epoch with best value for given metric."""
        if metric not in self.history[split]:
            return -1
        
        values = self.history[split][metric]
        epochs, metric_values = zip(*values)
        
        if mode == 'max':
            best_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmin(metric_values)
        
        return epochs[best_idx]
    
    def get_best_value(self, metric: str, split: str = 'val', mode: str = 'max') -> float:
        """Get best value for given metric."""
        if metric not in self.history[split]:
            return 0.0
        
        values = [v for _, v in self.history[split][metric]]
        return max(values) if mode == 'max' else min(values)

if __name__ == "__main__":
    # Example usage
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    
    # Test with dummy data
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2])
    
    metrics = compute_metrics(preds, labels, num_classes=3)
    logger.info(f"Test metrics: {metrics}")
    
    cm = compute_confusion_matrix(preds, labels)
    logger.info(f"Confusion matrix:\n{cm}")
