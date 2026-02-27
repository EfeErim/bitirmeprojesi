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
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.update_rate <= 0 or self.update_rate > 1:
            raise ValueError(f"update_rate must be in (0, 1], got {self.update_rate}")
        if self.min_samples <= 0:
            raise ValueError(f"min_samples must be > 0, got {self.min_samples}")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {self.feature_dim}")
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be > 0, got {self.cache_size}")


class PrototypeLookupResult:
    """Tensor-like wrapper that can also be unpacked as (prototype, class_std)."""

    def __init__(self, prototype: torch.Tensor, class_std: Optional[torch.Tensor]):
        self.prototype = prototype
        self.class_std = class_std

    def __iter__(self):
        yield self.prototype
        yield self.class_std

    def __getattr__(self, name):
        return getattr(self.prototype, name)

    def __getitem__(self, item):
        return self.prototype[item]

    def __repr__(self) -> str:
        return repr(self.prototype)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        norm_args = tuple(a.prototype if isinstance(a, PrototypeLookupResult) else a for a in args)
        norm_kwargs = {
            key: (value.prototype if isinstance(value, PrototypeLookupResult) else value)
            for key, value in kwargs.items()
        }
        return func(*norm_args, **norm_kwargs)


class PrototypeComputer:
    """
    Compute class prototypes and statistics for OOD detection.

    Backwards-compatible constructor: accepts either a `PrototypeConfig` via
    the `config` parameter, or legacy keyword args `feature_dim` and `device`.
    """

    def __init__(self, config: PrototypeConfig = None, feature_dim: int = None, device: str = 'cuda',
                 min_samples: int = None, use_moving_average: bool = None, update_rate: float = None):
        # Support either config object or legacy kwargs
        if config is None:
            if feature_dim is None:
                raise TypeError("PrototypeComputer requires either `config` or `feature_dim`")
            # Build config from kwargs if provided
            config_kwargs = {
                'feature_dim': feature_dim,
                'device': device
            }
            if min_samples is not None:
                config_kwargs['min_samples'] = min_samples
            if use_moving_average is not None:
                config_kwargs['use_moving_average'] = use_moving_average
            if update_rate is not None:
                config_kwargs['update_rate'] = update_rate
            config = PrototypeConfig(**config_kwargs)
        else:
            # Update config with kwargs if provided
            if min_samples is not None:
                config.min_samples = min_samples
            if use_moving_average is not None:
                config.use_moving_average = use_moving_average
            if update_rate is not None:
                config.update_rate = update_rate

        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.feature_dim = config.feature_dim
        
        # Initialize prototypes as dictionary for sparse class indices
        self.prototypes = {}
        self.prototype_counts = {}
        self.class_stds = {}
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
            Tuple of (prototypes tensor, class_stds dict)
        """
        logger.info("Computing prototypes from pre-extracted features with vectorized operations...")
        
        # Generate cache key based on features and labels
        cache_key = self._generate_cache_key(features, labels)
        
        # Check cache
        if cache_key in self.prototype_cache:
            self.cache_hits += 1
            return self.prototype_cache[cache_key]
        
        self.cache_misses += 1
        
        # Handle empty input
        if labels.numel() == 0 or features.numel() == 0:
            logger.warning("Empty features or labels provided to compute_prototypes_from_features")
            return torch.zeros(0, self.feature_dim, device=self.device), {}

        # Vectorized grouping by class
        unique_classes = torch.unique(labels)

        # Ensure prototype tensor can index by class index even if labels are non-contiguous
        max_class_idx = int(unique_classes.max().item())
        num_classes = max_class_idx + 1

        # Create prototypes tensor and stats dicts
        prototypes = torch.zeros(num_classes, self.feature_dim, device=self.device)
        class_stds = {}

        for class_idx in unique_classes:
            mask = labels == class_idx
            class_features = features[mask]
            idx = int(class_idx.item())

            if class_features.size(0) >= self.config.min_samples:
                mean = class_features.mean(dim=0)
                std = class_features.std(dim=0)

                prototypes[idx] = mean.to(self.device)
                class_stds[idx] = std.to(self.device)
                
                # Store for later use
                self.class_means[idx] = mean.to(self.device)
                self.prototype_counts[idx] = class_features.size(0)
                self.prototypes[idx] = mean.to(self.device)
            else:
                logger.warning(f"Insufficient samples ({class_features.size(0)}) for class {idx} (min: {self.config.min_samples})")
                prototypes[idx] = torch.zeros(self.feature_dim, device=self.device)
                class_stds[idx] = torch.ones(self.feature_dim, device=self.device) * 1e-6
                
                self.class_means[idx] = torch.zeros(self.feature_dim, device=self.device)
                self.prototype_counts[idx] = class_features.size(0)
                self.prototypes[idx] = torch.zeros(self.feature_dim, device=self.device)
        
        logger.info(f"Computed prototypes for {len(class_stds)} classes")
        
        self.class_stds = class_stds
        result = (prototypes, class_stds)
        
        # Cache the result
        self.prototype_cache[cache_key] = result
        
        if len(self.prototype_cache) > self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.prototype_cache))
            del self.prototype_cache[oldest_key]
        
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
    ) -> PrototypeLookupResult:
        """
        Get prototype for a specific class with caching.
        
        Args:
            class_idx: Class index
            
        Returns:
            Prototype tensor
        """
        if class_idx not in self.prototypes:
            self.cache_misses += 1
            raise ValueError(f"No prototype computed for class {class_idx}")

        self.cache_hits += 1
        return PrototypeLookupResult(
            prototype=self.prototypes[class_idx],
            class_std=self.class_stds.get(class_idx)
        )
    
    def get_prototype_cache_stats(self) -> Dict[str, int]:
        """
        Get prototype cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        # Provide both modern and legacy key names to preserve API compatibility
        stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.prototype_cache),
            # legacy keys expected by tests/older callers
            'hits': self.cache_hits,
            'misses': self.cache_misses,
        }

        return stats
    
    def clear_prototype_cache(self):
        """Clear prototype cache."""
        self.prototype_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def update_prototype(self, class_idx: int, new_features: torch.Tensor) -> torch.Tensor:
        """
        Update a prototype using moving average with new features.
        
        Args:
            class_idx: Class index
            new_features: New feature vectors of shape (num_samples, feature_dim)
            
        Returns:
            Updated prototype tensor
        """
        if new_features.numel() == 0:
            raise ValueError("No features provided for prototype update")
        
        # Compute mean of new features
        new_mean = new_features.mean(dim=0)
        
        # Get current prototype (or initialize if doesn't exist)
        if class_idx in self.prototypes:
            old_proto = self.prototypes[class_idx]
            old_count = self.prototype_counts.get(class_idx, 1)
        else:
            old_proto = torch.zeros(self.feature_dim, device=self.device)
            old_count = 0
        
        # Compute moving average update
        update_rate = self.config.update_rate
        updated = old_proto * (1 - update_rate) + new_mean.to(self.device) * update_rate
        
        # Update stored prototype
        self.prototypes[class_idx] = updated
        self.prototype_counts[class_idx] = old_count + new_features.size(0)
        
        return updated
    
    def compute_distances(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances from features to prototypes.
        
        Args:
            features: Feature tensor of shape (num_samples, feature_dim)
            prototypes: Prototype tensor of shape (num_classes, feature_dim)
            
        Returns:
            Distance tensor of shape (num_samples, num_classes)
        """
        if features.numel() == 0 or prototypes.numel() == 0:
            return torch.zeros(features.size(0), prototypes.size(0), device=self.device)
        
        # Use Euclidean distance via cdist
        distances = torch.cdist(features, prototypes)
        return distances
    
    def save_prototypes(self, path: str) -> None:
        """
        Save prototypes and related data to file.
        
        Args:
            path: Path to save file
        """
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        state = {
            'prototypes': self.prototypes,
            'prototypes_dict': {k: v.cpu() if torch.is_tensor(v) else v 
                                 for k, v in self.prototypes.items()},
            'prototype_counts': self.prototype_counts,
            'class_stds': {k: v.cpu() if torch.is_tensor(v) else v 
                          for k, v in self.class_stds.items()},
            'class_means': {k: v.cpu() if torch.is_tensor(v) else v 
                           for k, v in self.class_means.items()},
            'config': {
                'feature_dim': self.config.feature_dim,
                'device': str(self.device),
                'use_moving_average': self.config.use_moving_average,
                'update_rate': self.config.update_rate,
                'min_samples': self.config.min_samples,
            }
        }
        torch.save(state, path)
        logger.info(f"Saved prototypes to {path}")
    
    def load_prototypes(self, path: str) -> None:
        """
        Load prototypes and related data from file.
        
        Args:
            path: Path to load file from
        """
        state = torch.load(path, weights_only=False)
        
        # Restore prototypes - handle both dict and tensor formats
        proto_dict = state.get('prototypes_dict', {})
        if proto_dict:
            self.prototypes = {k: v.to(self.device) for k, v in proto_dict.items()}
        
        self.prototype_counts = state.get('prototype_counts', {})
        
        class_stds = state.get('class_stds', {})
        self.class_stds = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in class_stds.items()}
        
        class_means = state.get('class_means', {})
        self.class_means = {k: v.to(self.device) if torch.is_tensor(v) else v 
                           for k, v in class_means.items()}
        
        logger.info(f"Loaded prototypes from {path}")


def compute_prototypes(
    features_or_model,
    labels_or_data_loader=None,
    feature_dim: int = None,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Compute class prototypes from training data.
    Supports two signatures:
    1. compute_prototypes(features, labels) - direct feature computation
    2. compute_prototypes(model, data_loader, feature_dim, device) - legacy API
    
    Args:
        features_or_model: Either feature tensor or model
        labels_or_data_loader: Either label tensor or data_loader
        feature_dim: Feature dimension (only used with model API)
        device: Device for computation
        
    Returns:
        Tuple of (prototypes, class_stds)
    """
    # Check if this is the new API (features, labels) or old API (model, data_loader, feature_dim)
    if torch.is_tensor(features_or_model):
        # New API: compute_prototypes(features, labels)
        features = features_or_model
        labels = labels_or_data_loader
        
        logger.info("Computing class prototypes from features...")
        
        # Create prototype computer with default feature_dim from features
        config = PrototypeConfig(feature_dim=features.shape[1], device=device)
        computer = PrototypeComputer(config=config)
        
        # Compute prototypes
        prototypes, class_stds = computer.compute_prototypes_from_features(features, labels)
        
        return prototypes, class_stds
    else:
        # Old API: compute_prototypes(model, data_loader, feature_dim, device)
        model = features_or_model
        data_loader = labels_or_data_loader
        
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
    old_prototypes: torch.Tensor,
    new_features: torch.Tensor = None,
    new_labels: torch.Tensor = None,
    update_rate: float = 0.1,
    new_sample: torch.Tensor = None
) -> torch.Tensor:
    """
    Update prototypes using moving average.
    Supports two signatures:
    1. update_prototypes_moving_average(old_proto, new_sample, update_rate) - single sample
    2. update_prototypes_moving_average(old_prototypes, new_features, new_labels, update_rate) - batch by class
    
    Args:
        old_prototypes: Existing prototypes tensor or single prototype
        new_features: New feature vectors (for batch mode) or single sample (for legacy mode)
        new_labels: Labels for new features (batch mode) or None (legacy)
        update_rate: Rate of update (0-1)
        new_sample: (legacy) new sample tensor
        
    Returns:
        Updated prototype(s) tensor
    """
    # Handle legacy single sample case: update_prototypes_moving_average(old, new, rate)
    if new_labels is None and isinstance(update_rate, (int, float)):
        # Legacy API: (old_proto, new_sample, update_rate)
        # In this case, new_features is actually the new_sample
        return old_prototypes * (1 - update_rate) + new_features * update_rate
    
    # Batch update by class
    if new_labels is not None:
        updated = old_prototypes.clone()
        
        # Group features by class
        unique_classes = torch.unique(new_labels)
        for class_idx in unique_classes:
            mask = new_labels == class_idx
            class_features = new_features[mask]
            class_idx = int(class_idx.item())
            
            if class_features.numel() > 0 and class_idx < len(old_prototypes):
                # Compute mean of new features for this class
                new_mean = class_features.mean(dim=0)
                # Update prototype using moving average
                updated[class_idx] = old_prototypes[class_idx] * (1 - update_rate) + new_mean * update_rate
        
        return updated
    
    # Default: treat new_features as single sample (backwards compatibility)
    return old_prototypes * (1 - update_rate) + new_features * update_rate


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
