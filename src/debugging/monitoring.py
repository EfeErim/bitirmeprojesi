#!/usr/bin/env python3
"""
Training Monitoring and Debugging System for AADS-ULoRA v5.5
Implements real-time monitoring, alerting, and debugging utilities.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import smtplib
from email.message import EmailMessage

class TrainingMonitor:
    """Comprehensive training monitoring and debugging"""
    
    def __init__(self, config: Dict):
        # Configuration
        self.config = config
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        
        # State tracking
        self.best_metrics = {}
        self.early_stop_counter = 0
        self.epoch = 0
        
        # Setup logging
        self._setup_logging()
        
        # Alerting config
        self.alert_config = config.get('alerts', {})
        
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