#!/usr/bin/env python3
"""
Dynamic OOD Threshold Computation for AADS-ULoRA v5.5
Computes per-class OOD thresholds using Mahalanobis distance statistics.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class DynamicOODThreshold:
    """
    Compute dynamic OOD thresholds per class using Mahalanobis distance.
    
    The threshold for each class is computed as:
        threshold = mean_distance + k * std_distance
    
    where k is a configurable factor (default: 2.0 for 95% confidence).
    """
    
    def __init__(
        self,
        threshold_factor: float = 2.0,
        min_val_samples_per_class: int = 10,
        fallback_threshold: float = 25.0
    ):
        """
        Initialize dynamic threshold calculator.
        
        Args:
            threshold_factor: Multiplier for std (k-sigma, default: 2.0 for 95%)
            min_val_samples_per_class: Minimum validation samples per class
            fallback_threshold: Default threshold if insufficient data
        """
        self.threshold_factor = threshold_factor
        self.min_val_samples_per_class = min_val_samples_per_class
        self.fallback_threshold = fallback_threshold
        
        # Will be populated after computing thresholds
        self.thresholds = {}
        self.class_stats = {}
    
    @classmethod
    def compute_thresholds(
        cls,
        mahalanobis,
        model: torch.nn.Module,
        val_loader: DataLoader,
        feature_dim: int,
        device: str = 'cuda'
    ) -> Dict[int, float]:
        """
        Compute dynamic OOD thresholds for all classes.
        
        Args:
            mahalanobis: MahalanobisDistance object
            model: Model to extract features from
            val_loader: Validation data loader
            feature_dim: Feature dimension
            device: Device for computation
            
        Returns:
            Dictionary mapping class index to threshold
        """
        logger.info("Computing dynamic OOD thresholds...")
        
        # Initialize threshold computer
        threshold_computer = cls()
        
        # Collect distances for each class
        distances_per_class = {i: [] for i in range(mahalanobis.num_classes)}
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Extract features
                outputs = model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                features = pooled_output
                
                # Compute distances to true class prototypes
                for feat, label in zip(features, labels):
                    class_idx = label.item()
                    if class_idx < mahalanobis.num_classes:
                        distance = mahalanobis.compute_distance(feat.unsqueeze(0), class_idx)
                        distances_per_class[class_idx].append(distance.item())
        
        # Compute thresholds per class
        thresholds = {}
        
        for class_idx, distances in distances_per_class.items():
            if len(distances) >= threshold_computer.min_val_samples_per_class:
                distances_array = np.array(distances)
                mean_dist = np.mean(distances_array)
                std_dist = np.std(distances_array)
                
                threshold = mean_dist + threshold_computer.threshold_factor * std_dist
                
                thresholds[class_idx] = float(threshold)
                
                logger.debug(f"Class {class_idx}: "
                           f"mean={mean_dist:.4f}, "
                           f"std={std_dist:.4f}, "
                           f"threshold={threshold:.4f} "
                           f"({len(distances)} samples)")
            else:
                logger.warning(f"Class {class_idx} has insufficient validation samples "
                             f"({len(distances)} < {threshold_computer.min_val_samples_per_class}), "
                             f"using fallback threshold")
                thresholds[class_idx] = threshold_computer.fallback_threshold
        
        logger.info(f"Computed dynamic thresholds for {len(thresholds)} classes")
        
        return thresholds
    
    def compute_thresholds_from_distances(
        self,
        distances_per_class: Dict[int, list]
    ) -> Dict[int, float]:
        """
        Compute thresholds from pre-collected distances.
        
        Args:
            distances_per_class: Dictionary mapping class index to list of distances
            
        Returns:
            Dictionary mapping class index to threshold
        """
        thresholds = {}
        
        for class_idx, distances in distances_per_class.items():
            if len(distances) >= self.min_val_samples_per_class:
                distances_array = np.array(distances)
                mean_dist = np.mean(distances_array)
                std_dist = np.std(distances_array)
                
                threshold = mean_dist + self.threshold_factor * std_dist
                thresholds[class_idx] = float(threshold)
            else:
                logger.warning(f"Class {class_idx} has insufficient samples, using fallback")
                thresholds[class_idx] = self.fallback_threshold
        
        return thresholds
    
    def validate_thresholds(
        self,
        thresholds: Dict[int, float],
        val_loader: DataLoader,
        model: torch.nn.Module,
        mahalanobis,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Validate computed thresholds on validation set.
        
        Args:
            thresholds: Dictionary of class thresholds
            val_loader: Validation data loader
            model: Model to extract features from
            mahalanobis: MahalanobisDistance object
            device: Device for computation
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating OOD thresholds...")
        
        model.eval()
        
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        total_in_dist = 0
        total_ood = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Extract features
                outputs = model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                features = pooled_output
                
                # For each sample, check if it's correctly classified as in-distribution
                for feat, label in zip(features, labels):
                    class_idx = label.item()
                    if class_idx not in thresholds:
                        continue
                    
                    threshold = thresholds[class_idx]
                    distance = mahalanobis.compute_distance(feat.unsqueeze(0), class_idx).item()
                    
                    is_ood = distance > threshold
                    is_correct_class = True  # Assuming we're evaluating on in-distribution
                    
                    total_in_dist += 1
                    
                    if is_ood:
                        false_positive_rate = 1  # This would be a false positive
                        false_positives += 1
                    else:
                        true_negatives += 1
        
        # Compute metrics
        if total_in_dist > 0:
            false_positive_rate = false_positives / total_in_dist
            true_negative_rate = true_negatives / total_in_dist
        else:
            false_positive_rate = 0.0
            true_negative_rate = 0.0
        
        metrics = {
            'false_positive_rate': false_positive_rate,
            'true_negative_rate': true_negative_rate,
            'total_in_dist_samples': total_in_dist
        }
        
        logger.info(f"Threshold validation - FPR: {false_positive_rate:.4f}")
        
        return metrics
    
    def get_threshold_statistics(
        self,
        thresholds: Dict[int, float]
    ) -> Dict[str, float]:
        """
        Compute statistics about the thresholds.
        
        Returns:
            Dictionary with threshold statistics
        """
        if not thresholds:
            return {
                'mean_threshold': 0.0,
                'std_threshold': 0.0,
                'min_threshold': 0.0,
                'max_threshold': 0.0,
                'num_classes': 0
            }
        
        threshold_values = list(thresholds.values())
        
        return {
            'mean_threshold': float(np.mean(threshold_values)),
            'std_threshold': float(np.std(threshold_values)),
            'min_threshold': float(np.min(threshold_values)),
            'max_threshold': float(np.max(threshold_values)),
            'num_classes': len(threshold_values)
        }

class AdaptiveThresholdManager:
    """
    Manages adaptive threshold updates over time.
    """
    
    def __init__(
        self,
        initial_thresholds: Dict[int, float],
        adaptation_rate: float = 0.1,
        min_threshold: float = 1.0,
        max_threshold: float = 100.0
    ):
        """
        Initialize adaptive threshold manager.
        
        Args:
            initial_thresholds: Initial threshold values
            adaptation_rate: Rate at which thresholds adapt (0-1)
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.thresholds = initial_thresholds.copy()
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Track threshold history
        self.threshold_history = {k: [v] for k, v in initial_thresholds.items()}
    
    def update_thresholds(
        self,
        new_thresholds: Dict[int, float],
        class_sample_counts: Optional[Dict[int, int]] = None
    ) -> Dict[int, float]:
        """
        Update thresholds based on new statistics.
        
        Args:
            new_thresholds: New computed thresholds
            class_sample_counts: Optional sample counts for weighting
            
        Returns:
            Updated thresholds
        """
        updated = {}
        
        for class_idx, new_thresh in new_thresholds.items():
            if class_idx in self.thresholds:
                # Exponential moving average
                old_thresh = self.thresholds[class_idx]
                adapted = old_thresh + self.adaptation_rate * (new_thresh - old_thresh)
            else:
                adapted = new_thresh
            
            # Clamp to bounds
            adapted = max(self.min_threshold, min(self.max_threshold, adapted))
            
            updated[class_idx] = adapted
            
            # Track history
            if class_idx not in self.threshold_history:
                self.threshold_history[class_idx] = []
            self.threshold_history[class_idx].append(adapted)
        
        self.thresholds = updated
        
        return updated
    
    def get_threshold_for_class(self, class_idx: int) -> float:
        """Get current threshold for a class."""
        return self.thresholds.get(class_idx, 25.0)
    
    def get_threshold_history(self, class_idx: int) -> list:
        """Get threshold history for a class."""
        return self.threshold_history.get(class_idx, [])

def calibrate_thresholds_using_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    mahalanobis,
    target_fpr: float = 0.05,
    device: str = 'cuda'
) -> Dict[int, float]:
    """
    Calibrate thresholds to achieve a target false positive rate.
    
    Args:
        model: Model to extract features from
        val_loader: Validation data loader
        mahalanobis: MahalanobisDistance object
        target_fpr: Target false positive rate (default: 0.05)
        device: Device for computation
        
    Returns:
        Dictionary of calibrated thresholds
    """
    logger.info(f"Calibrating thresholds for target FPR: {target_fpr}")
    
    # Collect all distances for in-distribution samples
    all_distances = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            features = pooled_output
            
            for feat, label in zip(features, labels):
                class_idx = label.item()
                distance = mahalanobis.compute_distance(feat.unsqueeze(0), class_idx).item()
                all_distances.append(distance)
                all_labels.append(class_idx)
    
    # For each class, find threshold that gives target FPR
    thresholds = {}
    
    for class_idx in set(all_labels):
        class_distances = [d for d, l in zip(all_distances, all_labels) if l == class_idx]
        
        if len(class_distances) >= 10:
            # Sort distances
            class_distances.sort()
            
            # Find threshold at target FPR percentile
            # For FPR=0.05, we want 95% of in-dist samples below threshold
            percentile = (1 - target_fpr) * 100
            threshold = np.percentile(class_distances, percentile)
            thresholds[class_idx] = float(threshold)
        else:
            logger.warning(f"Class {class_idx} has insufficient samples for calibration")
            thresholds[class_idx] = 25.0
    
    logger.info(f"Calibrated {len(thresholds)} class thresholds")
    
    return thresholds

if __name__ == "__main__":
    """Test dynamic threshold computation."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    num_classes = 3
    feature_dim = 1536
    batch_size = 32
    
    # Dummy model that returns fixed features
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return type('Output', (), {'last_hidden_state': torch.randn(batch_size, 1, feature_dim)})()
    
    model = DummyModel()
    
    # Dummy Mahalanobis
    class DummyMahalanobis:
        def __init__(self, num_classes):
            self.num_classes = num_classes
        
        def compute_distance(self, features, class_idx):
            return torch.randn(features.shape[0])
    
    mahalanobis = DummyMahalanobis(num_classes)
    
    # Dummy validation loader
    dummy_data = torch.randn(100, 3, 224, 224)
    dummy_labels = torch.randint(0, num_classes, (100,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    val_loader = DataLoader(dataset, batch_size=16)
    
    # Compute thresholds
    thresholds = DynamicOODThreshold.compute_thresholds(
        mahalanobis, model, val_loader, feature_dim, device='cpu'
    )
    
    print(f"Computed thresholds: {thresholds}")
    
    # Validate
    validator = DynamicOODThreshold()
    metrics = validator.validate_thresholds(thresholds, val_loader, model, mahalanobis, device='cpu')
    print(f"Validation metrics: {metrics}")