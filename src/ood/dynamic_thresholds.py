#!/usr/bin/env python3
"""
Dynamic OOD Threshold Computation for AADS-ULoRA v5.5
Computes per-class OOD thresholds using Mahalanobis distance statistics with statistical confidence.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from torch.utils.data import DataLoader
import logging
import scipy.stats as stats

logger = logging.getLogger(__name__)

class DynamicOODThreshold:
    """
    Compute dynamic OOD thresholds per class using Mahalanobis distance.
    
    The threshold for each class is computed using confidence intervals:
        threshold = upper_confidence_bound + k * std_distance
    
    where k is a configurable factor and the upper confidence bound provides
    statistical conservatism to prevent false positives.
    """
    
    def __init__(
        self,
        threshold_factor: float = 2.0,
        min_val_samples_per_class: int = 30,
        fallback_threshold: float = 25.0,
        confidence_level: float = 0.95,
        max_fallback_threshold: float = 50.0,
        min_fallback_threshold: float = 10.0,
        use_confidence_intervals: bool = True
    ):
        """
        Initialize dynamic threshold calculator.
        
        Args:
            threshold_factor: Multiplier for std (k-sigma, default: 2.0 for ~95% coverage)
            min_val_samples_per_class: Minimum validation samples per class (default: 30)
            fallback_threshold: Default threshold if insufficient data (default: 25.0)
            confidence_level: Confidence level for statistical bounds (default: 0.95)
            max_fallback_threshold: Maximum allowed fallback threshold (default: 50.0)
            min_fallback_threshold: Minimum allowed fallback threshold (default: 10.0)
            use_confidence_intervals: Whether to use confidence intervals (default: True)
        """
        self.threshold_factor = threshold_factor
        self.min_val_samples_per_class = min_val_samples_per_class
        self.fallback_threshold = fallback_threshold
        self.confidence_level = confidence_level
        self.max_fallback_threshold = max_fallback_threshold
        self.min_fallback_threshold = min_fallback_threshold
        self.use_confidence_intervals = use_confidence_intervals
        
        # Will be populated after computing thresholds
        self.thresholds = {}
        self.class_stats = {}
        self.confidence_intervals = {}
        
        # Validate configuration
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration parameters and log warnings if needed."""
        if self.min_val_samples_per_class < 30:
            logger.warning(
                f"Minimum validation samples per class is {self.min_val_samples_per_class}, "
                f"which is below recommended 30 for reliable statistics. "
                f"Consider increasing to at least 30 for robust threshold computation."
            )
        
        if not (0.9 <= self.confidence_level <= 0.99):
            logger.warning(
                f"Confidence level {self.confidence_level} is outside typical range (0.9-0.99). "
                f"Standard values: 0.95 (95% confidence) or 0.99 (99% confidence)."
            )
        
        if self.threshold_factor < 1.0 or self.threshold_factor > 3.0:
            logger.warning(
                f"Threshold factor (k-sigma) {self.threshold_factor} is outside typical range (1.0-3.0). "
                f"Recommended: 2.0 for 95% confidence, 2.576 for 99% confidence."
            )
        
        if self.fallback_threshold < self.min_fallback_threshold or self.fallback_threshold > self.max_fallback_threshold:
            logger.warning(
                f"Fallback threshold {self.fallback_threshold} is outside bounds "
                f"[{self.min_fallback_threshold}, {self.max_fallback_threshold}]."
            )
    
    @classmethod
    def compute_thresholds(
        cls,
        mahalanobis,
        model: torch.nn.Module,
        val_loader: DataLoader,
        feature_dim: int,
        device: str = 'cuda',
        **kwargs
    ) -> Dict[int, float]:
        """
        Compute dynamic OOD thresholds for all classes with confidence intervals.
        
        Args:
            mahalanobis: MahalanobisDistance object
            model: Model to extract features from
            val_loader: Validation data loader
            feature_dim: Feature dimension
            device: Device for computation
            **kwargs: Additional arguments for DynamicOODThreshold constructor
            
        Returns:
            Dictionary mapping class index to threshold
        """
        logger.info("Computing dynamic OOD thresholds with statistical confidence intervals...")
        
        # Initialize threshold computer with configuration
        threshold_computer = cls(**kwargs)
        
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
        
        # Compute thresholds per class with confidence intervals
        thresholds = {}
        
        for class_idx, distances in distances_per_class.items():
            sample_count = len(distances)
            
            if sample_count >= threshold_computer.min_val_samples_per_class:
                distances_array = np.array(distances)
                
                # Compute basic statistics
                mean_dist = np.mean(distances_array)
                std_dist = np.std(distances_array)
                
                # Compute confidence interval for the mean
                if threshold_computer.use_confidence_intervals:
                    ci_lower, ci_upper = threshold_computer._compute_confidence_interval(
                        distances_array, 
                        threshold_computer.confidence_level
                    )
                    # Use conservative upper bound for threshold
                    base_threshold = ci_upper
                else:
                    base_threshold = mean_dist
                    ci_lower, ci_upper = mean_dist, mean_dist
                
                # Final threshold: conservative base + variability margin
                threshold = base_threshold + threshold_computer.threshold_factor * std_dist
                
                # Store statistics
                thresholds[class_idx] = float(threshold)
                threshold_computer.class_stats[class_idx] = {
                    'mean': float(mean_dist),
                    'std': float(std_dist),
                    'n': sample_count,
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
                    'threshold': float(threshold)
                }
                
                logger.info(
                    f"Class {class_idx}: n={sample_count}, "
                    f"mean={mean_dist:.4f}, std={std_dist:.4f}, "
                    f"ci=[{ci_lower:.4f}, {ci_upper:.4f}], "
                    f"threshold={threshold:.4f}"
                )
            else:
                # Handle insufficient samples
                threshold = threshold_computer._handle_insufficient_samples(
                    class_idx, 
                    sample_count
                )
                
                thresholds[class_idx] = threshold
                
                logger.warning(
                    f"Class {class_idx} has insufficient validation samples "
                    f"({sample_count} < {threshold_computer.min_val_samples_per_class}), "
                    f"using fallback threshold: {threshold:.4f}"
                )
        
        logger.info(f"Computed dynamic thresholds for {len(thresholds)} classes")
        
        return thresholds
    
    def _compute_confidence_interval(self, data: np.ndarray, confidence: float) -> Tuple[float, float]:
        """
        Compute confidence interval for the mean of the data.
        
        Uses t-distribution for small samples (n < 30) and normal distribution
        for larger samples (n >= 30) based on Central Limit Theorem.
        
        Args:
            data: Array of data points
            confidence: Confidence level (e.g., 0.95 for 95% confidence)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        if n < 2:
            logger.warning(f"Sample size {n} is too small for confidence interval computation")
            mean = float(np.mean(data))
            return mean, mean
        
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        
        # Determine degrees of freedom and distribution
        if n < 30:
            # Use t-distribution for small samples
            df = n - 1
            interval = stats.t.interval(confidence, df, loc=mean, scale=sem)
        else:
            # Use normal distribution for large samples (CLT)
            interval = stats.norm.interval(confidence, loc=mean, scale=sem)
        
        return float(interval[0]), float(interval[1])
    
    def _handle_insufficient_samples(self, class_idx: int, sample_count: int) -> float:
        """
        Handle cases where there are insufficient validation samples.
        
        Implements a progressive fallback strategy based on sample count:
        - 0 samples: Global fallback
        - 1-4 samples: Very conservative fallback (1.5x base)
        - 5-9 samples: Moderately conservative fallback (1.2x base)
        - 10-29 samples: Standard fallback (1.0x base)
        
        Args:
            class_idx: Class index
            sample_count: Number of available samples
            
        Returns:
            Fallback threshold value
        """
        if sample_count == 0:
            # No samples at all - use global fallback
            threshold = self.fallback_threshold
        elif sample_count < 5:
            # Very few samples - use very conservative fallback
            threshold = min(self.fallback_threshold * 1.5, self.max_fallback_threshold)
        elif sample_count < 10:
            # Some samples but still insufficient - use moderately conservative fallback
            threshold = min(self.fallback_threshold * 1.2, self.max_fallback_threshold)
        elif sample_count < 30:
            # Borderline case - use standard fallback
            threshold = self.fallback_threshold
        else:
            # Should not reach here, but use standard fallback
            threshold = self.fallback_threshold
        
        # Ensure threshold is within bounds
        threshold = max(self.min_fallback_threshold, min(threshold, self.max_fallback_threshold))
        
        logger.debug(
            f"Class {class_idx}: insufficient samples ({sample_count}), "
            f"using fallback threshold: {threshold:.4f}"
        )
        
        return threshold
    
    def compute_thresholds_from_distances(
        self,
        distances_per_class: Dict[int, list]
    ) -> Dict[int, float]:
        """
        Compute thresholds from pre-collected distances with confidence intervals.
        
        Args:
            distances_per_class: Dictionary mapping class index to list of distances
            
        Returns:
            Dictionary mapping class index to threshold
        """
        thresholds = {}
        
        for class_idx, distances in distances_per_class.items():
            sample_count = len(distances)
            
            if sample_count >= self.min_val_samples_per_class:
                distances_array = np.array(distances)
                mean_dist = np.mean(distances_array)
                std_dist = np.std(distances_array)
                
                # Compute confidence interval
                if self.use_confidence_intervals:
                    ci_lower, ci_upper = self._compute_confidence_interval(
                        distances_array, self.confidence_level
                    )
                    base_threshold = ci_upper
                else:
                    base_threshold = mean_dist
                
                # Conservative threshold using upper confidence bound
                threshold = base_threshold + self.threshold_factor * std_dist
                thresholds[class_idx] = float(threshold)
            else:
                # Handle insufficient samples
                threshold = self._handle_insufficient_samples(class_idx, sample_count)
                thresholds[class_idx] = threshold
        
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
                    
                    total_in_dist += 1
                    
                    if is_ood:
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
            'total_in_dist_samples': total_in_dist,
            'num_classes_tested': len(thresholds)
        }
        
        logger.info(
            f"Threshold validation - FPR: {false_positive_rate:.4f}, "
            f"TNR: {true_negative_rate:.4f}"
        )
        
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
                'num_classes': 0,
                'confidence_level': self.confidence_level
            }
        
        threshold_values = list(thresholds.values())
        
        return {
            'mean_threshold': float(np.mean(threshold_values)),
            'std_threshold': float(np.std(threshold_values)),
            'min_threshold': float(np.min(threshold_values)),
            'max_threshold': float(np.max(threshold_values)),
            'num_classes': len(threshold_values),
            'confidence_level': self.confidence_level
        }
    
    def get_class_stats(self, class_idx: int) -> Optional[Dict]:
        """Get detailed statistics for a specific class."""
        return self.class_stats.get(class_idx)
    
    def get_all_class_stats(self) -> Dict[int, Dict]:
        """Get detailed statistics for all classes."""
        return self.class_stats.copy()

class AdaptiveThresholdManager:
    """
    Manages adaptive threshold updates over time with statistical tracking.
    """
    
    def __init__(
        self,
        initial_thresholds: Dict[int, float],
        adaptation_rate: float = 0.1,
        min_threshold: float = 1.0,
        max_threshold: float = 100.0,
        confidence_level: float = 0.95
    ):
        """
        Initialize adaptive threshold manager.
        
        Args:
            initial_thresholds: Initial threshold values
            adaptation_rate: Rate at which thresholds adapt (0-1)
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            confidence_level: Confidence level for statistical bounds
        """
        self.thresholds = initial_thresholds.copy()
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.confidence_level = confidence_level
        
        # Track threshold history
        self.threshold_history = {k: [v] for k, v in initial_thresholds.items()}
        self.update_counts = {k: 1 for k in initial_thresholds.keys()}
    
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
                
                # Update count
                self.update_counts[class_idx] = self.update_counts.get(class_idx, 0) + 1
            else:
                adapted = new_thresh
                self.update_counts[class_idx] = 1
            
            # Clamp to bounds
            adapted = max(self.min_threshold, min(self.max_threshold, adapted))
            
            updated[class_idx] = adapted
            
            # Track history
            if class_idx not in self.threshold_history:
                self.threshold_history[class_idx] = []
            self.threshold_history[class_idx].append(adapted)
        
        self.thresholds = updated
        
        logger.debug(
            f"Updated thresholds for {len(updated)} classes. "
            f"Avg adaptation rate: {self.adaptation_rate:.3f}"
        )
        
        return updated
    
    def get_threshold_for_class(self, class_idx: int) -> float:
        """Get current threshold for a class."""
        return self.thresholds.get(class_idx, 25.0)
    
    def get_threshold_history(self, class_idx: int) -> list:
        """Get threshold history for a class."""
        return self.threshold_history.get(class_idx, [])
    
    def get_update_count(self, class_idx: int) -> int:
        """Get number of times threshold for class has been updated."""
        return self.update_counts.get(class_idx, 0)

def calibrate_thresholds_using_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    mahalanobis,
    target_fpr: float = 0.05,
    device: str = 'cuda',
    confidence_level: float = 0.95,
    min_samples: int = 30
) -> Dict[int, float]:
    """
    Calibrate thresholds to achieve a target false positive rate.
    
    Args:
        model: Model to extract features from
        val_loader: Validation data loader
        mahalanobis: MahalanobisDistance object
        target_fpr: Target false positive rate (default: 0.05)
        device: Device for computation
        confidence_level: Confidence level for statistical bounds
        min_samples: Minimum samples required for calibration (default: 30)
        
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
        
        if len(class_distances) >= min_samples:
            # Sort distances
            class_distances.sort()
            
            # Find threshold at target FPR percentile
            # For FPR=0.05, we want 95% of in-dist samples below threshold
            percentile = (1 - target_fpr) * 100
            threshold = np.percentile(class_distances, percentile)
            
            # Compute confidence interval for the threshold
            if len(class_distances) >= 2:
                ci_lower, ci_upper = DynamicOODThreshold._compute_confidence_interval(
                    np.array(class_distances), confidence_level
                )
                # Use conservative upper bound
                threshold = max(threshold, ci_upper)
            
            thresholds[class_idx] = float(threshold)
        else:
            logger.warning(
                f"Class {class_idx} has insufficient samples for calibration "
                f"({len(class_distances)} < {min_samples}), using fallback"
            )
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
