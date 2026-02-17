#!/usr/bin/env python3
"""
Model utilities: shared helpers for extracting pooled features, model management, and inspection.
"""
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import json
import fnmatch
from pathlib import Path


def extract_pooled_output(model: Any, images: torch.Tensor) -> torch.Tensor:
    """Run `model(images)` and return the pooled CLS token embedding.

    This helper handles common model output formats (transformers-style
    outputs with `last_hidden_state`, tuples, or raw tensors).
    """
    outputs = model(images)

    # HuggingFace-style output with attribute
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state[:, 0, :]

    # Tuple/list outputs (e.g., (last_hidden_state, ...))
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        first = outputs[0]
        if hasattr(first, 'last_hidden_state'):
            return first.last_hidden_state[:, 0, :]
        if isinstance(first, torch.Tensor) and first.ndim == 3:
            return first[:, 0, :]

    # Tensor output
    if isinstance(outputs, torch.Tensor):
        if outputs.ndim == 3:
            return outputs[:, 0, :]
        if outputs.ndim == 2:
            # Already pooled
            return outputs

    raise ValueError("Unsupported model output format for pooled extraction")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)  # Convert to MB


def freeze_layers(model: nn.Module, layers: List[int] = None, pattern: str = None):
    """Freeze specific layers in a model.
    
    Args:
        model: The model to modify
        layers: List of layer indices to freeze (for Sequential models)
        pattern: Pattern to match layer names (e.g., "*conv*")
    """
    if layers is not None:
        for idx in layers:
            if hasattr(model, str(idx)):
                for param in getattr(model, str(idx)).parameters():
                    param.requires_grad = False
    
    if pattern is not None:
        for name, module in model.named_modules():
            if fnmatch.fnmatch(name, pattern):
                for param in module.parameters():
                    param.requires_grad = False


def unfreeze_layers(model: nn.Module, layers: List[int] = None, pattern: str = None):
    """Unfreeze specific layers in a model."""
    if layers is not None:
        for idx in layers:
            if hasattr(model, str(idx)):
                for param in getattr(model, str(idx)).parameters():
                    param.requires_grad = True
    
    if pattern is not None:
        for name, module in model.named_modules():
            if fnmatch.fnmatch(name, pattern):
                for param in module.parameters():
                    param.requires_grad = True


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, path: str = None) -> Tuple[int, float]:
    """Load model checkpoint from file."""
    if path is None:
        raise ValueError("Path must be provided")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    return epoch, loss


def save_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0, loss: float = 0.0, path: str = None, **kwargs):
    """Save model checkpoint to file."""
    if path is None:
        raise ValueError("Path must be provided")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)


class ModelLoader:
    """Class for loading models and checkpoints."""
    
    def __init__(self):
        pass
    
    def load_model(self, model: nn.Module, path: str = None, strict: bool = True) -> nn.Module:
        """Load model from file or return the provided model."""
        if path is not None:
            checkpoint = torch.load(path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            else:
                model.load_state_dict(checkpoint, strict=strict)
        return model
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint from file."""
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint


class ModelSaver:
    """Class for saving models and checkpoints."""
    
    def __init__(self):
        pass
    
    def save_model(self, model: nn.Module, path: str, compress: bool = False):
        """Save model to file."""
        if compress:
            # For compression, we could use torch.save with zip format
            torch.save(model.state_dict(), path, _use_new_zipfile_serialization=True)
        else:
            torch.save(model.state_dict(), path)
    
    def save_checkpoint(self, checkpoint: Dict, path: str):
        """Save checkpoint to file."""
        torch.save(checkpoint, path)


class ModelInspector:
    """Class for inspecting model properties."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def get_summary(self) -> Dict:
        """Get a summary of the model architecture and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        layers = []
        
        for name, module in self.model.named_modules():
            # Include all modules that have parameters or are important layers
            if list(module.parameters()) or isinstance(module, (nn.ReLU, nn.Dropout, nn.BatchNorm2d)):
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params,
                    'trainable': trainable
                })
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': get_model_size(self.model),
            'layers': layers
        }
    
    def find_dead_neurons(self, threshold: float = 1e-6) -> Dict[str, List[int]]:
        """Find neurons that always output zero (or near-zero) activations."""
        dead_neurons = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Check if weights are essentially zero
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    # Find rows (neurons) with very small norm
                    norms = weight.norm(dim=1) if weight.dim() > 1 else weight.abs()
                    dead_indices = (norms < threshold).nonzero(as_tuple=True)[0].tolist()
                    if dead_indices:
                        dead_neurons[name] = dead_indices
        
        return dead_neurons
    
    def check_gradient_flow(self) -> Dict:
        """Check gradient flow through the model."""
        gradient_info = {
            'avg_gradients': {},
            'gradient_flow_health': 'healthy'
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                gradient_info['avg_gradients'][name] = grad_mean
                
                # Check for vanishing gradients
                if grad_mean < 1e-7:
                    gradient_info['gradient_flow_health'] = 'vanishing'
                # Check for exploding gradients
                elif grad_mean > 100:
                    gradient_info['gradient_flow_health'] = 'exploding'
        
        return gradient_info
