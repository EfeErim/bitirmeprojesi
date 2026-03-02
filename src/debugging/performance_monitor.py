#!/usr/bin/env python3
"""
Comprehensive Performance Monitoring for Colab Deployment
Provides GPU monitoring, training metrics, memory profiling, and real-time reporting.
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings

import torch
import numpy as np
from tqdm.auto import tqdm

from src.dataset.error_handling import get_error_handler, get_resource_monitor
from src.core.config_manager import ConfigurationManager
from src.debugging.monitoring_types import GPUMetrics, TrainingMetrics, MemoryProfile, DriveIOMetrics
from src.debugging.collectors import GPUMonitor, MemoryProfiler, DriveIOMonitor

logger = logging.getLogger(__name__)
error_handler = get_error_handler()
resource_monitor = get_resource_monitor()


class PerformanceMonitor:
    """
    Main performance monitoring orchestrator.
    Integrates GPU, memory, training metrics, and I/O monitoring.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        log_dir: str = "./logs",
        enable_gpu: bool = True,
        enable_memory: bool = True,
        enable_drive: bool = True,
        enable_realtime: bool = True,
        update_interval: float = 1.0
    ):
        self.config = config or {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.gpu_monitor = GPUMonitor(update_interval) if enable_gpu else None
        self.memory_profiler = MemoryProfiler() if enable_memory else None
        self.drive_monitor = DriveIOMonitor() if enable_drive else None
        
        # State
        self._running = False
        self._lock = threading.RLock()
        self._training_start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._batch_start_time: Optional[float] = None
        
        # Metrics storage
        self.training_history: List[TrainingMetrics] = []
        self.gpu_history: List[GPUMetrics] = []
        self.memory_history: List[MemoryProfile] = []
        self.drive_history: List[DriveIOMetrics] = []
        
        # Counters
        self.total_samples = 0
        self.current_epoch = 0
        self.current_batch = 0
        
        # Real-time display
        self.enable_realtime = enable_realtime
        self.progress_bar: Optional[tqdm] = None
        
        # Logging
        self._setup_logging()
        self.metrics_file = self.log_dir / 'performance_metrics.jsonl'
        
        logger.info("PerformanceMonitor initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / 'performance_monitor.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def start_training(self, total_epochs: int, total_batches: Optional[int] = None, total_samples: Optional[int] = None):
        """Start monitoring training session."""
        self._training_start_time = time.time()
        self.total_samples = total_samples or 0
        
        # Start background monitors
        if self.gpu_monitor:
            self.gpu_monitor.start()
        if self.drive_monitor:
            self.drive_monitor.start()
        
        # Initialize progress bar
        if self.enable_realtime and total_batches:
            self.progress_bar = tqdm(
                total=total_batches * total_epochs,
                desc="Training",
                unit="batch",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        logger.info(f"Started training monitoring: {total_epochs} epochs, {total_batches} batches/epoch")
    
    def start_epoch(self, epoch: int):
        """Mark epoch start."""
        self._epoch_start_time = time.time()
        self.current_epoch = epoch
        logger.info(f"Started epoch {epoch}")
    
    def end_epoch(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """Mark epoch end and record metrics."""
        if self._epoch_start_time:
            epoch_time = time.time() - self._epoch_start_time
        else:
            epoch_time = None
        
        logger.info(f"Completed epoch {epoch} in {epoch_time:.2f}s" if epoch_time else f"Completed epoch {epoch}")
    
    def record_batch(
        self,
        batch: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        batch_size: int = 1
    ):
        """Record batch-level metrics."""
        if self._batch_start_time:
            batch_time = time.time() - self._batch_start_time
        else:
            batch_time = None
        
        # Calculate throughput
        throughput = batch_size / batch_time if batch_time and batch_time > 0 else None
        
        # Get memory snapshot
        memory_allocated = None
        memory_reserved = None
        if self.memory_profiler and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        
        metrics = TrainingMetrics(
            timestamp=time.time(),
            epoch=self.current_epoch,
            batch=batch,
            samples_processed=batch_size,
            batch_time=batch_time or 0.0,
            throughput=throughput,
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved
        )
        
        with self._lock:
            self.training_history.append(metrics)
            self.current_batch = batch
            self.total_samples += batch_size
        
        # Update memory profiler
        if self.memory_profiler:
            profile = self.memory_profiler.snapshot(batch)
            with self._lock:
                self.memory_history.append(profile)
        
        # Update progress bar
        if self.progress_bar and batch_time:
            self.progress_bar.update(1)
            postfix = {
                'loss': f'{loss:.4f}' if loss is not None else 'N/A',
                'lr': f'{learning_rate:.2e}' if learning_rate is not None else 'N/A',
                'throughput': f'{throughput:.1f}/s' if throughput else 'N/A'
            }
            self.progress_bar.set_postfix(postfix)
        
        # Log to file
        self._log_metrics(metrics)
        
        # Update batch start time
        self._batch_start_time = time.time()
    
    def start_batch(self):
        """Mark batch start for timing."""
        self._batch_start_time = time.time()
    
    def end_training(self):
        """Stop monitoring and generate final report."""
        logger.info("Training completed, generating performance report...")
        
        # Stop background monitors
        if self.gpu_monitor:
            self.gpu_monitor.stop()
        if self.drive_monitor:
            self.drive_monitor.stop()
        
        # Close progress bar
        if self.progress_bar:
            self.progress_bar.close()
        
        # Generate final report
        report = self.generate_report()
        self._save_report(report)
        
        logger.info("Performance monitoring complete")
        return report
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to file."""
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training': self._get_training_summary(),
            'gpu': self._get_gpu_summary(),
            'memory': self._get_memory_summary(),
            'drive_io': self._get_drive_summary(),
            'bottlenecks': self._identify_bottlenecks(),
            'optimizations': self._suggest_optimizations()
        }
        return report
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training performance summary."""
        if not self.training_history:
            return {}
        
        batch_times = [t.batch_time for t in self.training_history if t.batch_time > 0]
        throughputs = [t.throughput for t in self.training_history if t.throughput]
        losses = [t.loss for t in self.training_history if t.loss is not None]
        
        total_time = time.time() - self._training_start_time if self._training_start_time else 0
        
        return {
            'total_samples': self.total_samples,
            'total_time_seconds': total_time,
            'avg_throughput_samples_sec': float(np.mean(throughputs)) if throughputs else None,
            'avg_batch_time_seconds': float(np.mean(batch_times)) if batch_times else None,
            'final_loss': losses[-1] if losses else None,
            'loss_improvement': losses[0] - losses[-1] if len(losses) > 1 else None,
            'epochs_completed': self.current_epoch + 1,
            'batches_completed': len(self.training_history)
        }
    
    def _get_gpu_summary(self) -> Dict[str, Any]:
        """Get GPU performance summary."""
        if self.gpu_monitor:
            return self.gpu_monitor.get_summary()
        return {'available': torch.cuda.is_available()}
    
    def _get_memory_summary(self) -> Dict[str, Any]:
        """Get memory profiling summary."""
        if self.memory_profiler:
            return self.memory_profiler.get_statistics()
        return {}
    
    def _get_drive_summary(self) -> Dict[str, Any]:
        """Get Drive I/O summary."""
        if self.drive_monitor:
            return self.drive_monitor.get_summary()
        return {}
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check GPU utilization
        if self.gpu_monitor:
            gpu_summary = self.gpu_monitor.get_summary()
            avg_util = gpu_summary.get('avg_utilization')
            if avg_util is not None and avg_util < 50:
                bottlenecks.append({
                    'type': 'low_gpu_utilization',
                    'severity': 'high' if avg_util < 30 else 'medium',
                    'value': avg_util,
                    'suggestion': 'Increase batch size or reduce data loading overhead'
                })
        
        # Check memory leaks
        if self.memory_profiler:
            mem_summary = self.memory_profiler.get_statistics()
            if mem_summary.get('leak_suspected', False):
                bottlenecks.append({
                    'type': 'memory_leak',
                    'severity': 'high',
                    'suggestion': 'Check for tensors not being freed, use torch.cuda.empty_cache()'
                })
        
        # Check batch time consistency
        if len(self.training_history) > 10:
            batch_times = [t.batch_time for t in self.training_history[-10:] if t.batch_time > 0]
            if batch_times:
                cv = np.std(batch_times) / np.mean(batch_times) if np.mean(batch_times) > 0 else 0
                if cv > 0.2:  # High coefficient of variation
                    bottlenecks.append({
                        'type': 'inconsistent_batch_times',
                        'severity': 'medium',
                        'value': cv,
                        'suggestion': 'Check data loading variability, consider prefetching'
                    })
        
        return bottlenecks
    
    def _suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Suggest performance optimizations based on metrics."""
        suggestions = []
        
        # GPU utilization optimization
        if self.gpu_monitor:
            gpu_summary = self.gpu_monitor.get_summary()
            avg_util = gpu_summary.get('avg_utilization', 0)
            if avg_util < 70:
                suggestions.append({
                    'area': 'gpu_utilization',
                    'priority': 'high',
                    'suggestion': 'Increase batch size or enable gradient accumulation',
                    'expected_improvement': '10-30% faster training'
                })
        
        # Memory optimization
        if self.memory_profiler:
            mem_summary = self.memory_profiler.get_statistics()
            frag_ratio = mem_summary.get('fragmentation_ratio', 1.0)
            if frag_ratio > 1.5:
                suggestions.append({
                    'area': 'memory_fragmentation',
                    'priority': 'medium',
                    'suggestion': 'Call torch.cuda.empty_cache() periodically',
                    'expected_improvement': '5-15% memory savings'
                })
        
        # I/O optimization
        if self.drive_monitor:
            drive_summary = self.drive_monitor.get_summary()
            read_speed = drive_summary.get('avg_read_speed_mb_s', 0)
            if read_speed and read_speed < 50:  # Less than 50 MB/s
                suggestions.append({
                    'area': 'drive_io',
                    'priority': 'high',
                    'suggestion': 'Use local SSD cache for frequently accessed data',
                    'expected_improvement': '2-5x faster data loading'
                })
        
        # Mixed precision
        if not torch.cuda.amp.is_available():
            suggestions.append({
                'area': 'mixed_precision',
                'priority': 'high',
                'suggestion': 'Enable AMP (Automatic Mixed Precision) for 2-3x speedup',
                'expected_improvement': '2-3x faster training, 50% memory reduction'
            })
        
        return suggestions
    
    def _save_report(self, report: Dict[str, Any]):
        """Save performance report to file."""
        report_file = self.log_dir / 'performance_report.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def export_metrics_csv(self, filepath: Optional[str] = None):
        """Export training metrics to CSV format."""
        if not self.training_history:
            logger.warning("No training metrics to export")
            return
        
        filepath = filepath or str(self.log_dir / 'training_metrics.csv')
        
        import csv
        with open(filepath, 'w', newline='') as f:
            if self.training_history:
                writer = csv.DictWriter(f, fieldnames=self.training_history[0].to_dict().keys())
                writer.writeheader()
                for metrics in self.training_history:
                    writer.writerow(metrics.to_dict())
        
        logger.info(f"Metrics exported to {filepath}")
    
    def get_live_stats(self) -> Dict[str, Any]:
        """Get current live statistics for real-time display."""
        stats = {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'total_samples': self.total_samples
        }
        
        # Add GPU stats
        if self.gpu_monitor:
            gpu_current = self.gpu_monitor.get_current()
            if gpu_current:
                stats['gpu_util'] = gpu_current.utilization
                stats['gpu_memory_gb'] = gpu_current.memory_used_gb
                stats['gpu_temp'] = gpu_current.temperature
        
        # Add memory stats
        if self.memory_profiler and self.memory_history:
            latest_mem = self.memory_history[-1]
            stats['memory_allocated_gb'] = latest_mem.allocated_gb
            stats['memory_peak_gb'] = latest_mem.peak_allocated_gb
        
        # Add throughput
        if self.training_history:
            recent = self.training_history[-10:]
            throughputs = [t.throughput for t in recent if t.throughput]
            if throughputs:
                stats['avg_throughput'] = float(np.mean(throughputs))
        
        return stats


class ColabPerformanceOptimizer:
    """
    Colab-specific performance optimizations.
    Provides recommendations and automatic optimizations.
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.recommendations: List[Dict[str, Any]] = []
        
    def analyze_and_optimize(self) -> List[Dict[str, Any]]:
        """Analyze performance and provide optimization recommendations."""
        self.recommendations = []
        
        # Get summaries
        gpu_summary = self.monitor._get_gpu_summary()
        memory_summary = self.monitor._get_memory_summary()
        drive_summary = self.monitor._get_drive_summary()
        training_summary = self.monitor._get_training_summary()
        
        # GPU-specific optimizations
        if gpu_summary.get('available'):
            avg_util = gpu_summary.get('avg_utilization', 0)
            if avg_util < 60:
                self.recommendations.append({
                    'category': 'gpu',
                    'priority': 1,
                    'title': 'Low GPU Utilization',
                    'description': f'GPU utilization is only {avg_util:.1f}%. Data loading may be the bottleneck.',
                    'actions': [
                        'Increase num_workers in DataLoader',
                        'Enable prefetching with prefetch_factor',
                        'Use local SSD cache for dataset',
                        'Reduce data augmentation complexity'
                    ]
                })
            
            mem_util = gpu_summary.get('avg_memory_util', 0)
            if mem_util and mem_util > 90:
                self.recommendations.append({
                    'category': 'memory',
                    'priority': 1,
                    'title': 'High GPU Memory Usage',
                    'description': f'GPU memory utilization is {mem_util:.1f}%. Risk of OOM.',
                    'actions': [
                        'Reduce batch size',
                        'Enable gradient checkpointing',
                        'Use gradient accumulation',
                        'Enable mixed precision training'
                    ]
                })
        
        # Drive I/O optimizations
        if drive_summary.get('mounted'):
            read_speed = drive_summary.get('avg_read_speed_mb_s', 0)
            if read_speed and read_speed < 100:
                self.recommendations.append({
                    'category': 'io',
                    'priority': 1,
                    'title': 'Slow Drive I/O',
                    'description': f'Drive read speed is only {read_speed:.1f} MB/s. This will bottleneck training.',
                    'actions': [
                        'Copy dataset to local SSD (/content)',
                        'Use LRU cache for dataset files',
                        'Pre-load dataset into memory if small enough',
                        'Use memory-mapped files'
                    ]
                })
        
        # Training efficiency
        avg_throughput = training_summary.get('avg_throughput_samples_sec', 0)
        if avg_throughput:
            # Estimate if throughput is reasonable based on GPU type
            expected_throughput = self._estimate_expected_throughput()
            if expected_throughput and avg_throughput < expected_throughput * 0.5:
                self.recommendations.append({
                    'category': 'efficiency',
                    'priority': 2,
                    'title': 'Low Training Throughput',
                    'description': f'Current throughput: {avg_throughput:.1f} samples/sec, expected: {expected_throughput:.1f}',
                    'actions': [
                        'Enable mixed precision (AMP)',
                        'Use torch.compile() if using PyTorch 2.0+',
                        'Optimize data augmentation pipeline',
                        'Reduce model complexity if possible'
                    ]
                })
        
        return self.recommendations
    
    def _estimate_expected_throughput(self) -> Optional[float]:
        """Estimate expected throughput based on GPU model."""
        if not torch.cuda.is_available():
            return None
        
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        # Rough estimates for common Colab GPUs (samples/sec for typical model)
        estimates = {
            't4': 50.0,
            'k80': 30.0,
            'p100': 60.0,
            'p4': 25.0,
            'v100': 100.0,
            'a100': 150.0,
            'l4': 80.0
        }
        
        for key, value in estimates.items():
            if key in gpu_name:
                return value
        
        # Default estimate
        return 50.0
    
    def apply_auto_optimizations(self) -> List[str]:
        """Apply automatic optimizations where safe."""
        applied = []
        
        # Set optimal thread settings
        torch.set_num_threads(4)  # Prevent CPU oversubscription
        applied.append("Set optimal CPU thread count")
        
        # Enable cuDNN benchmark if input sizes are consistent
        torch.backends.cudnn.benchmark = True
        applied.append("Enabled cuDNN benchmark")
        
        # Clear cache to start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            applied.append("Cleared GPU cache")
        
        logger.info(f"Applied {len(applied)} automatic optimizations")
        return applied


def create_performance_monitor(config_path: Optional[str] = None) -> PerformanceMonitor:
    """
    Factory function to create PerformanceMonitor with configuration.
    
    Args:
        config_path: Path to configuration file (JSON/YAML)
    
    Returns:
        Configured PerformanceMonitor instance
    """
    config = {}
    if config_path:
        try:
            path_obj = Path(config_path)
            if path_obj.is_dir():
                cfg_mgr = ConfigurationManager(config_dir=str(path_obj))
                merged = cfg_mgr.load_all_configs()
            else:
                environment = path_obj.stem if path_obj.stem != 'base' else None
                cfg_mgr = ConfigurationManager(config_dir=str(path_obj.parent), environment=environment)
                merged = cfg_mgr.load_all_configs()
            config = merged.get('performance_monitoring', {})
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Set defaults
    defaults = {
        'log_dir': './logs',
        'enable_gpu': True,
        'enable_memory': True,
        'enable_drive': True,
        'enable_realtime': True,
        'update_interval': 1.0
    }
    defaults.update(config)
    
    return PerformanceMonitor(**defaults)


# Example usage
if __name__ == "__main__":
    # Simple test
    monitor = create_performance_monitor()
    
    print("Performance Monitor Test")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    monitor.start_training(total_epochs=1, total_batches=10, total_samples=100)
    
    # Simulate training loop
    for epoch in range(1):
        monitor.start_epoch(epoch)
        for batch in range(10):
            monitor.start_batch()
            time.sleep(0.1)  # Simulate training
            # Simulate metrics
            loss = 1.0 - (epoch * 0.1 + batch * 0.01)
            lr = 1e-4
            grad_norm = 0.5 + np.random.randn() * 0.1
            monitor.record_batch(
                batch=batch,
                loss=loss,
                learning_rate=lr,
                gradient_norm=abs(grad_norm),
                batch_size=8
            )
        monitor.end_epoch(epoch, {'loss': loss})
    
    report = monitor.end_training()
    print("\nReport Summary:")
    print(json.dumps(report, indent=2))
    
    # Export CSV
    monitor.export_metrics_csv()
    
    print("\nTest complete. Check logs/ directory for outputs.")
