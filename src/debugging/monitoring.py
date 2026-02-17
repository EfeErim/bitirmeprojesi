#!/usr/bin/env python3
"""
Training Monitoring and Debugging System for AADS-ULoRA v5.5
Implements real-time monitoring, alerting, and debugging utilities.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import smtplib
from email.message import EmailMessage

class TrainingMonitor:
    """Comprehensive training monitoring and debugging"""
    
    def __init__(self, config: Dict = None):
        # Configuration
        self.config = config or {}
        self.log_dir = Path(self.config.get('log_dir', './logs'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        
        # State tracking
        self.best_metrics = {}
        self.early_stop_counter = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.epoch_metrics = {}
        self.batch_metrics = {}
        
        # Setup logging
        self._setup_logging()
        
        # Alerting config
        self.alert_config = self.config.get('alerts', {})
        
    def _setup_logging(self):
        """Configure structured logging"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main logger
        self.logger = logging.getLogger('AADS-Monitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
    def track_metrics(self, metrics: Dict[str, float], phase: str = 'train'):
        """Track and validate training metrics"""
        # Check for NaN/Inf
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                self.trigger_alert(
                    f'Invalid metric value: {name}={value}',
                    level='critical'
                )
                
        # Update best metrics
        for metric in ['accuracy', 'auroc']:
            if metric in metrics:
                key = f'best_{metric}'
                if metrics[metric] > self.best_metrics.get(key, 0):
                    self.best_metrics[key] = metrics[metric]
                    self.logger.info(f'New best {metric}: {metrics[metric]:.4f}')
        
        # Early stopping check
        if 'val_loss' in metrics:
            self._check_early_stopping(metrics['val_loss'])
        
    def _check_early_stopping(self, val_loss: float):
        """Implement early stopping logic"""
        if val_loss < self.best_metrics.get('best_loss', float('inf')):
            self.best_metrics['best_loss'] = val_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            
            if self.early_stop_counter >= self.config.get('patience', 10):
                self.trigger_alert('Early stopping triggered', level='warning')
                raise RuntimeError('Early stopping triggered')
        
    def record_epoch(self, epoch: int, train_loss: float = None, val_loss: float = None,
                     train_acc: float = None, val_acc: float = None, **kwargs):
        """Record epoch-level metrics."""
        self.current_epoch = epoch
        self.epoch_metrics[epoch] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            **kwargs
        }
        
        # Track best metrics
        if val_loss is not None:
            if 'best_val_loss' not in self.best_metrics or val_loss < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = val_loss
        if val_acc is not None:
            if 'best_val_acc' not in self.best_metrics or val_acc > self.best_metrics['best_val_acc']:
                self.best_metrics['best_val_acc'] = val_acc
    
    def record_batch(self, batch: int, loss: float = None, learning_rate: float = None,
                     gradient_norm: float = None, **kwargs):
        """Record batch-level metrics."""
        self.current_batch = batch
        self.batch_metrics = getattr(self, 'batch_metrics', {})
        self.batch_metrics[batch] = {
            'loss': loss,
            'learning_rate': learning_rate,
            'gradient_norm': gradient_norm,
            **kwargs
        }
        
        # Log batch metrics
        if loss is not None:
            self.logger.info(f'Batch {batch}: loss={loss:.4f}')
    
    def monitor_gradients(self, model: torch.nn.Module):
        """Track gradient statistics"""
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms.append(norm)
                
                # Check for vanishing/exploding gradients
                if norm < 1e-6:
                    self.trigger_alert(
                        f'Vanishing gradients in {name}: {norm:.2e}',
                        level='warning'
                    )
                elif norm > 1e3:
                    self.trigger_alert(
                        f'Exploding gradients in {name}: {norm:.2e}',
                        level='critical'
                    )
        
        return {
            'grad_norm_mean': np.mean(grad_norms),
            'grad_norm_std': np.std(grad_norms),
            'grad_norm_max': np.max(grad_norms)
        }
        
    def trigger_alert(self, message: str, level: str = 'info'):
        """Send alerts through configured channels"""
        # Log locally
        getattr(self.logger, level)(message)
        
        # Email alerts for critical issues
        if level == 'critical' and 'email' in self.alert_config:
            self._send_email_alert(message)
        
    def _send_email_alert(self, message: str):
        """Send email alert"""
        msg = EmailMessage()
        msg.set_content(f"[AADS Alert] {message}")
        msg['Subject'] = 'AADS Training Alert'
        msg['From'] = self.alert_config['email']['from']
        msg['To'] = self.alert_config['email']['to']
        
        with smtplib.SMTP(self.alert_config['email']['smtp_server'],
                        self.alert_config['email']['smtp_port']) as server:
            server.send_message(msg)
            
    def save_checkpoint(self, state: Dict, is_best: bool = False):
        """Save training checkpoint"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{self.epoch}.pth'
        torch.save(state, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'model_best.pth'
            torch.save(state, best_path)
            
    def log_hardware_stats(self):
        """Log GPU/CPU utilization"""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            self.logger.info(f'GPU memory used: {gpu_mem:.2f}GB')
            
    def start_epoch(self):
        """Begin epoch tracking"""
        self.epoch_start_time = datetime.now()
        self.epoch += 1
        
    def end_epoch(self):
        """Log epoch duration"""
        duration = (datetime.now() - self.epoch_start_time).total_seconds()
        self.logger.info(f'Epoch {self.epoch} completed in {duration:.2f}s')

class Debugger:
    """Advanced debugging utilities"""
    pass


class DebugMonitor:
    """Main debug monitor singleton."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the monitor."""
        self.is_running = False
        self.metrics = {}
    
    def start(self):
        """Start monitoring."""
        self.is_running = True
    
    def stop(self):
        """Stop monitoring."""
        self.is_running = False
    
    def record_metric(self, name: str, value: float):
        """Record a metric."""
        self.metrics[name] = value
    
    def get_metrics(self):
        """Get all metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()


class ModelDebugger:
    """Model-specific debugging utilities."""
    
    @staticmethod
    def check_gradients(model: torch.nn.Module) -> Dict:
        """Check gradient statistics."""
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms.append(norm)
        
        if grad_norms:
            return {
                'grad_norm_mean': float(np.mean(grad_norms)),
                'grad_norm_std': float(np.std(grad_norms)),
                'grad_norm_max': float(np.max(grad_norms))
            }
        return {}
    
    @staticmethod
    def check_weights(model: torch.nn.Module) -> Dict:
        """Check weight statistics."""
        weight_norms = []
        for name, param in model.named_parameters():
            if param.data is not None:
                norm = param.data.norm(2).item()
                weight_norms.append(norm)
        
        if weight_norms:
            return {
                'weight_norm_mean': float(np.mean(weight_norms)),
                'weight_norm_std': float(np.std(weight_norms)),
                'weight_norm_max': float(np.max(weight_norms))
            }
        return {}
    
    @staticmethod
    def validate_forward_pass(model: torch.nn.Module, input_tensor: torch.Tensor) -> bool:
        """Validate that forward pass works without errors."""
        try:
            with torch.no_grad():
                _ = model(input_tensor)
            return True
        except Exception:
            return False
    
    @staticmethod
    def detect_nan_inf(model: torch.nn.Module) -> Dict[str, List[str]]:
        """Detect NaN/Inf in model parameters."""
        issues = {'nan': [], 'inf': []}
        
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                issues['nan'].append(name)
            if torch.isinf(param.data).any():
                issues['inf'].append(name)
        
        return issues
    
    def get_debug_report(self, model: torch.nn.Module) -> Dict:
        """Generate comprehensive debug report."""
        report = {
            'gradients': self.check_gradients(model),
            'weights': self.check_weights(model),
            'nan_inf': self.detect_nan_inf(model)
        }
        return report


class GradientTracker:
    """Track gradient flow through the model."""
    
    def __init__(self):
        self.gradients = {}
        self.hooks = []
    
    def track_gradients(self, model: torch.nn.Module):
        """Start tracking gradients."""
        self.gradients = {}
        
        def get_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach().cpu()
            return hook
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(get_gradient(name))
                self.hooks.append(hook)
    
    def gradient_flow_analysis(self) -> Dict:
        """Analyze gradient flow."""
        if not self.gradients:
            return {}
        
        stats = {}
        for name, grad in self.gradients.items():
            if grad is not None:
                stats[name] = {
                    'mean': grad.abs().mean().item(),
                    'std': grad.abs().std().item(),
                    'max': grad.abs().max().item()
                }
        
        return stats
    
    def reset_tracker(self):
        """Reset tracking state."""
        self.gradients.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ActivationTracker:
    """Track activations through the model."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def track_activations(self, model: torch.nn.Module):
        """Start tracking activations."""
        self.activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu()
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def activation_statistics(self) -> Dict:
        """Compute activation statistics."""
        stats = {}
        for name, act in self.activations.items():
            if act is not None:
                stats[name] = {
                    'mean': act.abs().mean().item(),
                    'std': act.abs().std().item(),
                    'sparsity': (act == 0).float().mean().item()
                }
        return stats
    
    def dead_activation_detection(self, threshold: float = 1e-6) -> List[str]:
        """Detect dead neurons (near-zero activations)."""
        dead = []
        for name, act in self.activations.items():
            if act is not None and act.abs().mean().item() < threshold:
                dead.append(name)
        return dead
    
    def reset_tracker(self):
        """Reset tracking state."""
        self.activations.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class DebugLogger:
    """Logging utilities for debugging."""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.logger = logging.getLogger('debug')
        
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Log a metric value."""
        msg = f"Metric: {name}={value}"
        if step is not None:
            msg += f" (step={step})"
        self.logger.info(msg)
    
    def log_message(self, message: str, level: str = 'info'):
        """Log a message."""
        getattr(self.logger, level)(message)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int = None):
        """Log histogram data (simplified)."""
        msg = f"Histogram: {name} - mean={np.mean(values):.4f}, std={np.std(values):.4f}"
        if step is not None:
            msg += f" (step={step})"
        self.logger.info(msg)
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model Graph: input_shape={input_shape}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        # Simplified implementation
        with open(filepath, 'w') as f:
            f.write("Debug metrics export\n")
    
    @staticmethod
    def visualize_activations(model: torch.nn.Module, input_tensor: torch.Tensor):
        """Hook to visualize layer activations"""
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        handles = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                handles.append(layer.register_forward_hook(get_activation(name)))
        
        # Run forward pass
        with torch.no_grad():
            model(input_tensor)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return activations
    
    @staticmethod
    def gradient_analysis(model: torch.nn.Module) -> Dict:
        """Analyze gradient flow"""
        stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                stats[name] = {
                    'mean': param.grad.data.mean().item(),
                    'std': param.grad.data.std().item(),
                    'min': param.grad.data.min().item(),
                    'max': param.grad.data.max().item()
                }
        return stats

if __name__ == "__main__":
    # Example usage
    monitor = TrainingMonitor({
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints',
        'patience': 5,
        'alerts': {
            'email': {
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'from': 'alerts@aads.com',
                'to': 'admin@aads.com'
            }
        }
    })
    
    # Simulate training loop
    for epoch in range(10):
        monitor.start_epoch()
        
        # Track metrics
        monitor.track_metrics({
            'accuracy': np.random.uniform(0.8, 0.95),
            'loss': np.random.uniform(0.1, 0.5),
            'val_loss': np.random.uniform(0.2, 0.6)
        })
        
        monitor.end_epoch()