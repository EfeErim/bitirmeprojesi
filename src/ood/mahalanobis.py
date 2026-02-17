#!/usr/bin/env python3
"""
Mahalanobis Distance Calculator for OOD Detection
Computes Mahalanobis distance between features and class prototypes.
"""

import torch
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class MahalanobisDistance:
    """
    Compute Mahalanobis distance between features and class prototypes.
    
    Args:
        prototypes: Tensor of shape (num_classes, feature_dim)
        class_stds: Dictionary mapping class index to std tensor
    """
    
    def __init__(
        self,
        prototypes: torch.Tensor,
        class_stds: Dict[int, torch.Tensor],
        device: str = 'cpu'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        
        self.prototypes = prototypes.to(self.device)
        self.class_stds = {k: v.to(self.device) for k, v in class_stds.items()}
        self.num_classes = prototypes.shape[0]
        self.feature_dim = prototypes.shape[1]
        
        # Precompute inverse covariance matrices
        self.inv_covariances = self._compute_inv_covariances()
    
    def _compute_inv_covariances(self) -> Dict[int, torch.Tensor]:
        """Compute inverse covariance matrices for each class with numerical stability."""
        inv_covariances = {}

        for class_idx in range(self.num_classes):
            if class_idx in self.class_stds:
                std = self.class_stds[class_idx]
                # Create diagonal covariance matrix from std
                cov = torch.diag(std ** 2)

                # Adaptive regularization based on covariance trace
                # Ensures stability even for near-zero variance features
                trace = torch.trace(cov)
                reg_param = max(1e-6, (trace / self.feature_dim) * 1e-4)
                cov += torch.eye(self.feature_dim, device=cov.device) * reg_param

                # Robust matrix inversion with error handling
                try:
                    # Check condition number before inversion
                    cond_num = torch.cond(cov)
                    if cond_num > 1e10:
                        logger.warning(f"Class {class_idx}: Ill-conditioned covariance (cond={cond_num:.2e}), using pseudo-inverse")
                        inv_cov = torch.linalg.pinv(cov)
                    else:
                        inv_cov = torch.linalg.inv(cov)

                    # Check for NaN/Inf in result
                    if torch.isnan(inv_cov).any() or torch.isinf(inv_cov).any():
                        logger.warning(f"Class {class_idx}: NaN/Inf detected in inverse, using pseudo-inverse")
                        inv_cov = torch.linalg.pinv(cov)

                except RuntimeError as e:
                    logger.warning(f"Class {class_idx}: Matrix inversion failed ({str(e)}), using pseudo-inverse")
                    inv_cov = torch.linalg.pinv(cov)

                inv_covariances[class_idx] = inv_cov
            else:
                # Use identity matrix if no std available
                inv_covariances[class_idx] = torch.eye(self.feature_dim, device=self.prototypes.device)

        return inv_covariances
    
    def compute_distance(
        self,
        features: torch.Tensor,
        class_idx: int
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance between features and class prototype (device-safe).

        Args:
            features: Tensor of shape (batch_size, feature_dim)
            class_idx: Class index to compute distance to

        Returns:
            Tensor of shape (batch_size,) with Mahalanobis distances
        """
        if class_idx >= self.num_classes or class_idx not in self.inv_covariances:
            raise ValueError(f"Invalid class index: {class_idx}")

        # Ensure device consistency
        prototype = self.prototypes[class_idx].to(features.device)
        inv_cov = self.inv_covariances[class_idx].to(features.device)

        # Compute (x - μ)
        diff = features - prototype

        # Compute (x - μ)^T * inv_cov * (x - μ) per sample
        distances = (diff @ inv_cov * diff).sum(dim=1)

        return distances
    
    def compute_all_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distances to all class prototypes.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            
        Returns:
            Tensor of shape (batch_size, num_classes) with distances
        """
        distances = torch.zeros(features.shape[0], self.num_classes, device=features.device)
        
        for class_idx in range(self.num_classes):
            distances[:, class_idx] = self.compute_distance(features, class_idx)
        
        return distances
    
    def get_nearest_class(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get nearest class for each feature based on Mahalanobis distance.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            
        Returns:
            Tuple of (nearest_class_indices, distances)
        """
        all_distances = self.compute_all_distances(features)
        nearest_indices = torch.argmin(all_distances, dim=1)
        nearest_distances = torch.min(all_distances, dim=1).values
        
        return nearest_indices, nearest_distances

class MahalanobisDistanceBatch:
    """
    Batch Mahalanobis distance calculator for efficiency.
    """
    
    def __init__(
        self,
        prototypes: torch.Tensor,
        class_stds: Dict[int, torch.Tensor],
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.prototypes = prototypes.to(self.device)
        self.class_stds = {k: v.to(self.device) for k, v in class_stds.items()}
        self.num_classes = prototypes.shape[0]
        self.feature_dim = prototypes.shape[1]
        
        # Precompute inverse covariance matrices
        self.inv_covariances = self._compute_inv_covariances()
    
    def _compute_inv_covariances(self) -> Dict[int, torch.Tensor]:
        """Compute inverse covariance matrices for each class with numerical stability."""
        inv_covariances = {}

        for class_idx in range(self.num_classes):
            if class_idx in self.class_stds:
                std = self.class_stds[class_idx]
                cov = torch.diag(std ** 2)

                # Adaptive regularization based on covariance trace
                trace = torch.trace(cov)
                reg_param = max(1e-6, (trace / self.feature_dim) * 1e-4)
                cov += torch.eye(self.feature_dim, device=cov.device) * reg_param

                # Robust matrix inversion with error handling
                try:
                    cond_num = torch.cond(cov)
                    if cond_num > 1e10:
                        logger.warning(f"Class {class_idx}: Ill-conditioned covariance (cond={cond_num:.2e}), using pseudo-inverse")
                        inv_cov = torch.linalg.pinv(cov)
                    else:
                        inv_cov = torch.linalg.inv(cov)

                    if torch.isnan(inv_cov).any() or torch.isinf(inv_cov).any():
                        logger.warning(f"Class {class_idx}: NaN/Inf detected in inverse, using pseudo-inverse")
                        inv_cov = torch.linalg.pinv(cov)

                except RuntimeError as e:
                    logger.warning(f"Class {class_idx}: Matrix inversion failed ({str(e)}), using pseudo-inverse")
                    inv_cov = torch.linalg.pinv(cov)

                inv_covariances[class_idx] = inv_cov
            else:
                inv_covariances[class_idx] = torch.eye(self.feature_dim, device=self.device)

        return inv_covariances
    
    def compute_batch_distances(
        self,
        features: torch.Tensor,
        class_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distances for a batch of features.
        
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            class_indices: Optional list of class indices to compute distances for
            
        Returns:
            Tensor of shape (batch_size, num_classes) with distances
        """
        if class_indices is None:
            class_indices = list(range(self.num_classes))
        
        batch_size = features.shape[0]
        num_classes = len(class_indices)
        
        distances = torch.zeros(batch_size, num_classes, device=self.device)
        
        for i, class_idx in enumerate(class_indices):
            if class_idx < self.num_classes and class_idx in self.inv_covariances:
                prototype = self.prototypes[class_idx]
                inv_cov = self.inv_covariances[class_idx]
                
                # Compute (x - μ)
                diff = features - prototype
                
                # Compute (x - μ)^T * inv_cov * (x - μ) per sample
                # Correct: element-wise multiplication and sum over feature dimension
                distances[:, i] = (diff @ inv_cov * diff).sum(dim=1)
        
        return distances

if __name__ == "__main__":
    """Test Mahalanobis distance calculator."""
    import torch
    
    # Create dummy prototypes and stds
    prototypes = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    class_stds = {
        0: torch.tensor([0.1, 0.1, 0.1]),
        1: torch.tensor([0.2, 0.2, 0.2]),
        2: torch.tensor([0.3, 0.3, 0.3])
    }
    
    # Create calculator
    mahalanobis = MahalanobisDistance(prototypes, class_stds)
    
    # Test with sample features
    features = torch.tensor([
        [1.1, 0.1, 0.1],  # Close to class 0
        [0.1, 1.1, 0.1],  # Close to class 1
        [0.1, 0.1, 1.1],  # Close to class 2
        [0.5, 0.5, 0.5]   # Far from all classes
    ])
    
    distances = mahalanobis.compute_all_distances(features)
    print(f"Distances:\n{distances}")
    
    nearest, nearest_dist = mahalanobis.get_nearest_class(features)
    print(f"Nearest classes: {nearest}")
    print(f"Nearest distances: {nearest_dist}")