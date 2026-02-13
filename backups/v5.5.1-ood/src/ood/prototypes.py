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

logger = logging.getLogger(__name__)

class PrototypeComputer:
    """
    Compute class prototypes and statistics for OOD detection.
    
    Args:
        feature_dim: Dimensionality of feature vectors
        device: Device for computation
    """
    
    def __init__(
        self,
        feature_dim: int,
        device: str = 'cuda'
    ):
        self.feature_dim = feature_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.prototypes = None
        self.class_stds = {}
        self.class_counts = {}
    
    def compute_prototypes(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        class_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Compute class prototypes from training data.
        
        Args:
            model: Model to extract features from
            data_loader: Data loader for training data
            class_to_idx: Mapping from class names to indices
            
        Returns:
            Tuple of (prototypes, class_stds)
        """
        logger.info("Computing class prototypes...")
        
        # Initialize accumulators
        features_per_class = {idx: [] for idx in range(len(class_to_idx))}
        
        # Extract features
        model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features
                outputs = model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                features = pooled_output
                
                # Store features by class
                for feat, label in zip(features, labels):
                    class_idx = label.item()
                    if class_idx in features_per_class:
                        features_per_class[class_idx].append(feat.cpu())
        
        # Compute means and stds
        prototypes = torch.zeros(len(class_to_idx), self.feature_dim, device=self.device)
        class_stds = {}
        
        for class_idx, feat_list in features_per_class.items():
            if len(feat_list) >= 2:  # Need at least 2 samples for std
                feats = torch.stack(feat_list)
                mean = feats.mean(dim=0)
                std = feats.std(dim=0)
                
                prototypes[class_idx] = mean
                class_stds[class_idx] = std
            else:
                logger.warning(f"Insufficient samples for class {class_idx}")
                # Use zero vector as placeholder
                prototypes[class_idx] = torch.zeros(self.feature_dim, device=self.device)
                class_stds[class_idx] = torch.ones(self.feature_dim, device=self.device) * 1e-6
        
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
        Compute prototypes from pre-extracted features.
        
        Args:
            features: Tensor of shape (num_samples, feature_dim)
            labels: Tensor of shape (num_samples,) with class indices
            
        Returns:
            Tuple of (prototypes, class_stds)
        """
        logger.info("Computing prototypes from pre-extracted features...")
        
        features_per_class = {}
        
        for feat, label in zip(features, labels):
            class_idx = label.item()
            if class_idx not in features_per_class:
                features_per_class[class_idx] = []
            features_per_class[class_idx].append(feat.cpu())
        
        # Compute means and stds
        num_classes = len(features_per_class)
        prototypes = torch.zeros(num_classes, self.feature_dim, device=self.device)
        class_stds = {}
        
        for class_idx, feat_list in features_per_class.items():
            if len(feat_list) >= 2:
                feats = torch.stack(feat_list)
                mean = feats.mean(dim=0)
                std = feats.std(dim=0)
                
                prototypes[class_idx] = mean
                class_stds[class_idx] = std
            else:
                logger.warning(f"Insufficient samples for class {class_idx}")
                prototypes[class_idx] = torch.zeros(self.feature_dim, device=self.device)
                class_stds[class_idx] = torch.ones(self.feature_dim, device=self.device) * 1e-6
        
        logger.info(f"Computed prototypes for {len(class_stds)} classes")
        
        self.prototypes = prototypes
        self.class_stds = class_stds
        
        return prototypes, class_stds
    
    def get_prototype_quality(
        self,
        validation_loader: DataLoader,
        model: torch.nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate quality of computed prototypes.
        
        Args:
            validation_loader: Validation data loader
            model: Model to extract features from
            
        Returns:
            Dictionary with quality metrics
        """
        if self.prototypes is None:
            raise ValueError("Prototypes not computed yet")
        
        logger.info("Evaluating prototype quality...")
        
        model.eval()
        with torch.no_grad():
            total_distance = 0.0
            total_samples = 0
            
            for images, labels in validation_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                features = pooled_output
                
                for feat, label in zip(features, labels):
                    class_idx = label.item()
                    if class_idx in self.class_stds:
                        prototype = self.prototypes[class_idx]
                        std = self.class_stds[class_idx]
                        
                        # Compute normalized distance
                        diff = feat - prototype
                        distance = torch.norm(diff / std).item()
                        total_distance += distance
                        total_samples += 1
        
        avg_distance = total_distance / total_samples if total_samples > 0 else 0.0
        
        return {
            'avg_normalized_distance': avg_distance,
            'num_samples_evaluated': total_samples
        }

class PrototypeValidator:
    """
    Validate and analyze computed prototypes.
    """
    
    def __init__(self, prototypes: torch.Tensor, class_stds: Dict[int, torch.Tensor]):
        self.prototypes = prototypes
        self.class_stds = class_stds
        self.num_classes = prototypes.shape[0]
        self.feature_dim = prototypes.shape[1]
    
    def compute_inter_class_distances(self) -> torch.Tensor:
        """
        Compute pairwise distances between class prototypes.
        
        Returns:
            Tensor of shape (num_classes, num_classes) with distances
        """
        distances = torch.zeros(self.num_classes, self.num_classes)
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    diff = self.prototypes[i] - self.prototypes[j]
                    distances[i, j] = torch.norm(diff).item()
        
        return distances
    
    def compute_class_diversity_score(self) -> float:
        """
        Compute diversity score based on inter-class distances.
        
        Returns:
            Diversity score (higher is better)
        """
        if self.num_classes < 2:
            return 0.0
        
        distances = self.compute_inter_class_distances()
        # Average distance to other classes
        avg_distances = torch.sum(distances, dim=1) / (self.num_classes - 1)
        diversity_score = torch.mean(avg_distances).item()
        
        return diversity_score
    
    def find_similar_classes(self, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Find pairs of classes with similar prototypes.
        
        Args:
            threshold: Distance threshold for similarity
            
        Returns:
            List of (class1, class2, distance) tuples
        """
        similar_pairs = []
        
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                distance = torch.norm(self.prototypes[i] - self.prototypes[j]).item()
                if distance < threshold:
                    similar_pairs.append((i, j, distance))
        
        return similar_pairs

def compute_class_prototypes(
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
        feature_dim: Dimensionality of feature vectors
        device: Device for computation
        
    Returns:
        Tuple of (prototypes, class_stds)
    """
    prototype_computer = PrototypeComputer(feature_dim, device)
    return prototype_computer.compute_prototypes(model, data_loader, {})

if __name__ == "__main__":
    """Test prototype computation."""
    import torch
    from torchvision import models
    
    # Create dummy model
    model = models.resnet18()
    
    # Create dummy data loader
    class DummyDataset(Dataset):
        def __init__(self):
            self.data = torch.randn(100, 3, 224, 224)
            self.labels = torch.randint(0, 3, (100,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = DummyDataset()
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Compute prototypes
    prototypes, stds = compute_class_prototypes(model, data_loader, model.fc.in_features)
    
    print(f"Computed {prototypes.shape[0]} prototypes")
    print(f"Prototype shape: {prototypes.shape}")
    print(f"Class stds: {stds}")