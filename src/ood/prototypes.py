#!/usr/bin/env python3
"""
Prototype Computation for OOD Detection
Computes class prototypes and statistics from training data.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
import logging
from src.utils.model_utils import extract_pooled_output
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrototypeConfig:
    """Configuration for prototype computation."""
    feature_dim: int
    device: str = 'cuda'
    use_moving_average: bool = False
    update_rate: float = 0.1
    min_samples: int = 5
    max_prototypes: int = 1000
    cache_size: int = 100


class PrototypeComputer:
    """
    Compute class prototypes and statistics for OOD detection.
    
    Args:
        config: PrototypeConfig object
    """
    
    def __init__(self, config: PrototypeConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.feature_dim = config.feature_dim
        self.prototypes = None
        self.class_stds = {}
        self.class_counts = {}
        self.class_means = {}
        
        # Cache for frequently accessed prototypes
        self.prototype_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def compute_prototypes(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        class_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Compute class prototypes from training data using vectorized operations.
        
        Args:
            model: Model to extract features from
            data_loader: Data loader for training data
            class_to_idx: Mapping from class names to indices
            
        Returns:
            Tuple of (prototypes, class_stds)
        """
        logger.info("Computing class prototypes with vectorized operations...")
        
        # Initialize accumulators
        num_classes = len(class_to_idx)
        features_per_class = {idx: [] for idx in range(num_classes)}
        
        # Extract features in batches
        model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                pooled_output = extract_pooled_output(model, images)
                features = pooled_output
                
                # Store features by class
                for feat, label in zip(features, labels):
                    class_idx = label.item()
                    if class_idx in features_per_class:
                        features_per_class[class_idx].append(feat.cpu())
        
        # Compute means and stds using vectorized operations
        prototypes = torch.zeros(num_classes, self.feature_dim, device=self.device)
        class_stds = {}
        
        for class_idx, feat_list in features_per_class.items():
            if len(feat_list) >= 2:  # Need at least 2 samples for std
                feats = torch.stack(feat_list)
                
                # Vectorized mean and std computation
                mean = feats.mean(dim=0)
                std = feats.std(dim=0)
                
                prototypes[class_idx] = mean
                class_stds[class_idx] = std
                
                # Store for later use
                self.class_means[class_idx] = mean
                self.class_counts[class_idx] = len(feat_list)
            else:
                logger.warning(f"Insufficient samples for class {class_idx}")
                # Use zero vector as placeholder
                prototypes[class_idx] = torch.zeros(self.feature_dim, device=self.device)
                class_stds[class_idx] = torch.ones(self.feature_dim, device=self.device) * 1e-6
                self.class_means[class_idx] = torch.zeros(self.feature_dim, device=self.device)
                self.class_counts[class_idx] = len(feat_list)
        
        logger.info(f"Computed prototypes for {len(class_stds)} classes")
        
        self.prototypes = prototypes
        self.class_stds = class_stds
        
        return prototypes, class_stds
    
    def compute_prototypes_from_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Compute prototypes from pre-extracted features using vectorized operations.
        Implements caching to avoid recomputing for the same (features, labels) combination.
        
        Args:
            features: Tensor of shape (num_samples, feature_dim)
            labels: Tensor of shape (num_samples,) with class indices
            
        Returns:
            Tuple of (prototypes, class_stds)
        """
        logger.info("Computing prototypes from pre-extracted features with vectorized operations...")
        
        # Generate cache key based on features and labels
        cache_key = self._generate_cache_key(features, labels)
        
        # Check cache
        if cache_key in self.prototype_cache:
            self.cache_hits += 1
            return self.prototype_cache[cache_key]
        
        self.cache_misses += 1
        
        features_per_class = {}
        
        # Vectorized grouping by class
        unique_classes = torch.unique(labels)
        
        for class_idx in unique_classes:
            mask = labels == class_idx
            class_features = features[mask]
            
            if class_features.size(0) >= 2:
                mean = class_features.mean(dim=0)
                std = class_features.std(dim=0)
                
                features_per_class[class_idx.item()] = {
                    'mean': mean,
                    'std': std,
                    'count': class_features.size(0)
                }
            else:
                logger.warning(f"Insufficient samples for class {class_idx.item()}")
                features_per_class[class_idx.item()] = {
                    'mean': torch.zeros(self.feature_dim, device=self.device),
                    'std': torch.ones(self.feature_dim, device=self.device) * 1e-6,
                    'count': class_features.size(0)
                }
        
        # Create prototypes tensor
        num_classes = len(features_per_class)
        prototypes = torch.zeros(num_classes, self.feature_dim, device=self.device)
        class_stds = {}
        
        for class_idx, stats in features_per_class.items():
            prototypes[class_idx] = stats['mean']
            class_stds[class_idx] = stats['std']
            
            # Store for later use
            self.class_means[class_idx] = stats['mean']
            self.class_counts[class_idx] = stats['count']
        
        logger.info(f"Computed prototypes for {len(class_stds)} classes")
        
        result = (prototypes, class_stds)
        
        # Cache the result
        self.prototype_cache[cache_key] = result
        if len(self.prototype_cache) > self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.prototype_cache))
            del self.prototype_cache[oldest_key]
        
        self.prototypes = prototypes
        self.class_stds = class_stds
        
        return result
    
    def _generate_cache_key(self, features: torch.Tensor, labels: torch.Tensor) -> str:
        """Generate a cache key from features and labels."""
        # Use hash of features and labels as key
        features_hash = hash(features.cpu().numpy().tobytes())
        labels_hash = hash(labels.cpu().numpy().tobytes())
        return f"{features_hash}_{labels_hash}"
    
    def get_prototype_quality(
        self,
        validation_loader: DataLoader,
        model: torch.nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate quality of computed prototypes using vectorized operations.
        
        Args:
            validation_loader: Validation data loader
            model: Model to extract features from
            
        Returns:
            Dictionary with quality metrics
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not computed yet")
        
        logger.info("Evaluating prototype quality with vectorized operations...")
        
        model.eval()
        with torch.no_grad():
            total_distance = 0.0
            total_samples = 0
            
            for images, labels in validation_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                pooled_output = extract_pooled_output(model, images)
                features = pooled_output
                
                for feat, label in zip(features, labels):
                    class_idx = label.item()
                    if class_idx in self.class_stds:
                        prototype = self.prototypes[class_idx]
                        std = self.class_stds[class_idx]
                        
                        # Vectorized normalized distance computation
                        diff = feat - prototype
                        distance = torch.norm(diff / std).item()
                        total_distance += distance
                        total_samples += 1
        
        avg_distance = total_distance / total_samples if total_samples > 0 else 0.0
        
        return {
            'avg_normalized_distance': avg_distance,
            'num_samples_evaluated': total_samples
        }
    
    def get_prototype_for_class(
        self,
        class_idx: int
    ) -> torch.Tensor:
        """
        Get prototype for a specific class with caching.
        
        Args:
            class_idx: Class index
            
        Returns:
            Prototype tensor
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not computed yet")
        
        # Check cache first
        if class_idx in self.prototype_cache:
            self.cache_hits += 1
            return self.prototype_cache[class_idx]
        
        self.cache_misses += 1
        prototype = self.prototypes[class_idx]
        
        # Cache the result
        self.prototype_cache[class_idx] = prototype
        
        # Limit cache size
        if len(self.prototype_cache) > self.config.cache_size:
            oldest_key = next(iter(self.prototype_cache))
            del self.prototype_cache[oldest_key]
        
        return prototype
    
    def get_prototype_cache_stats(self) -> Dict[str, int]:
        """
        Get prototype cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.prototype_cache)
        }
    
    def clear_prototype_cache(self):
        """Clear prototype cache."""
        self.prototype_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


def compute_prototypes(
    model: torch.nn.Module,
    data_loader: DataLoader,
    feature_dim: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Compute class prototypes from training data.
    
    Args:
        model: Model to extract features from
        data_loader: Data loader for training data
        feature_dim: Feature dimension
        device: Device for computation
        
    Returns:
        Tuple of (prototypes, class_stds)
    """
    logger.info("Computing class prototypes using PrototypeComputer...")
    
    # Get class mapping from data loader
    # Try to get class_to_idx from dataset
    if hasattr(data_loader.dataset, 'class_to_idx'):
        class_to_idx = data_loader.dataset.class_to_idx
    else:
        # Infer from labels
        logger.warning("Dataset has no class_to_idx, inferring from labels")
        all_labels = []
        for _, labels in data_loader:
            all_labels.extend(labels.cpu().numpy())
        unique_classes = sorted(set(all_labels))
        class_to_idx = {str(idx): idx for idx in unique_classes}
    
    # Create prototype computer
    config = PrototypeConfig(feature_dim=feature_dim, device=device)
    computer = PrototypeComputer(config=config)
    
    # Compute prototypes
    prototypes, class_stds = computer.compute_prototypes(model, data_loader, class_to_idx)
    
    return prototypes, class_stds


def update_prototypes_moving_average(
    old_prototype: torch.Tensor,
    new_sample: torch.Tensor,
    update_rate: float = 0.1
) -> torch.Tensor:
    """
    Update prototype using moving average.
    
    Args:
        old_prototype: Existing prototype
        new_sample: New sample to incorporate
        update_rate: Rate of update (0-1)
        
    Returns:
        Updated prototype
    """
    return old_prototype * (1 - update_rate) + new_sample * update_rate


def find_nearest_prototype(
    feature: torch.Tensor,
    prototypes: torch.Tensor
) -> Tuple[int, float]:
    """
    Find nearest prototype to a feature vector.
    
    Args:
        feature: Feature vector
        prototypes: Prototype tensor
        
    Returns:
        Tuple of (class_idx, distance)
    """
    distances = torch.cdist(feature.unsqueeze(0), prototypes)
    nearest_idx = distances.argmin().item()
    nearest_distance = distances.min().item()
    
    return nearest_idx, nearest_distance


def compute_prototype_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor
) -> float:
    """
    Compute classification accuracy using prototypes.
    
    Args:
        features: Feature vectors
        labels: True labels
        prototypes: Prototype tensor
        
    Returns:
        Accuracy score
    """
    correct = 0
    total = len(features)
    
    for feat, label in zip(features, labels):
        nearest_idx, _ = find_nearest_prototype(feat, prototypes)
        if nearest_idx == label.item():
            correct += 1
    
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    """Example usage of PrototypeComputer."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=1536, help='Feature dimension')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create prototype computer
    pc = PrototypeComputer(
        feature_dim=args.feature_dim,
        device=args.device
    )
    
    logger.info(f"PrototypeComputer initialized on {pc.device}")
    logger.info(f"Feature dimension: {pc.feature_dim}")