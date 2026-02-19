#!/usr/bin/env python3
"""
v5.5 Performance Metrics Tracking Module

Tracks all v5.5 specification performance targets:
- Phase 1 DoRA: ≥95% accuracy
- Phase 2 SD-LoRA: ≥90% retention on original diseases
- Phase 3 CONEC-LoRA: ≥85% protectedclass retention
- OOD Detection: ≥92% AUROC on validation out-of-distribution samples
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class V55PerformanceMetrics:
    """Tracks v5.5 performance targets and actual performance."""
    
    # v5.5 Specification Targets
    TARGETS = {
        'phase1_accuracy': 0.95,        # DoRA: ≥95% accuracy
        'phase2_retention': 0.90,       # SD-LoRA: ≥90% retention
        'phase3_protected_retention': 0.85,  # CONEC: ≥85% protected class retention
        'ood_auroc': 0.92,              # OOD detection: ≥92% AUROC
    }
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize metrics tracker.
        
        Args:
            output_dir: Directory to save metrics reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.metrics_file = self.output_dir / 'v55_performance_metrics.json'
        
        self.metrics = {
            'created_at': datetime.now().isoformat(),
            'v55_targets': self.TARGETS,
            'phases': {
                'phase1': {},
                'phase2': {},
                'phase3': {},
                'ood_detection': {}
            }
        }
        
        self.load_metrics()
        
    def load_metrics(self):
        """Load existing metrics from file if available."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    loaded = json.load(f)
                    # Preserve existing metrics
                    if 'phases' in loaded:
                        self.metrics['phases'].update(loaded['phases'])
                logger.info(f"Loaded existing metrics from {self.metrics_file}")
            except Exception as e:
                logger.warning(f"Could not load metrics: {e}")
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics to {self.metrics_file}")
    
    # Phase 1 Metrics
    def add_phase1_metrics(self, 
                          accuracy: float,
                          loss: float,
                          num_classes: int,
                          per_class_accuracy: Optional[Dict[int, float]] = None,
                          training_time_sec: Optional[float] = None):
        """Record Phase 1 (DoRA) metrics.
        
        Args:
            accuracy: Overall accuracy on validation set
            loss: Final validation loss
            num_classes: Number of classes
            per_class_accuracy: Per-class accuracy dictionary
            training_time_sec: Training time in seconds
        """
        self.metrics['phases']['phase1'] = {
            'accuracy': accuracy,
            'loss': loss,
            'num_classes': num_classes,
            'per_class_accuracy': per_class_accuracy or {},
            'training_time_sec': training_time_sec,
            'meets_target': accuracy >= self.TARGETS['phase1_accuracy'],
            'target': self.TARGETS['phase1_accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log result
        status = "✅ PASS" if accuracy >= self.TARGETS['phase1_accuracy'] else "❌ FAIL"
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1 (DoRA) - {status}")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {accuracy:.4f} (target ≥ {self.TARGETS['phase1_accuracy']:.2f})")
        logger.info(f"Loss: {loss:.4f}")
        logger.info(f"Classes: {num_classes}")
        if training_time_sec:
            logger.info(f"Training Time: {training_time_sec/60:.1f} minutes")
        
        self.save_metrics()
    
    # Phase 2 Metrics
    def add_phase2_metrics(self,
                          new_disease_accuracy: float,
                          old_diseases_retention: float,
                          num_old_classes: int,
                          num_new_classes: int,
                          loss: float,
                          per_class_retention: Optional[Dict[int, float]] = None,
                          training_time_sec: Optional[float] = None):
        """Record Phase 2 (SD-LoRA) metrics.
        
        Args:
            new_disease_accuracy: Accuracy on newly added disease classes
            old_diseases_retention: Retention (accuracy) on original disease classes
            num_old_classes: Number of original disease classes
            num_new_classes: Number of newly added classes
            loss: Final validation loss
            per_class_retention: Per-class retention dictionary
            training_time_sec: Training time in seconds
        """
        self.metrics['phases']['phase2'] = {
            'new_disease_accuracy': new_disease_accuracy,
            'old_diseases_retention': old_diseases_retention,
            'num_old_classes': num_old_classes,
            'num_new_classes': num_new_classes,
            'loss': loss,
            'per_class_retention': per_class_retention or {},
            'meets_target': old_diseases_retention >= self.TARGETS['phase2_retention'],
            'target': self.TARGETS['phase2_retention'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log result
        retention_status = "✅ PASS" if old_diseases_retention >= self.TARGETS['phase2_retention'] else "❌ FAIL"
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2 (SD-LoRA) - {retention_status}")
        logger.info(f"{'='*60}")
        logger.info(f"New Disease Accuracy: {new_disease_accuracy:.4f}")
        logger.info(f"Old Diseases Retention: {old_diseases_retention:.4f} (target ≥ {self.TARGETS['phase2_retention']:.2f})")
        logger.info(f"Old Classes: {num_old_classes}, New Classes: {num_new_classes}")
        logger.info(f"Loss: {loss:.4f}")
        if training_time_sec:
            logger.info(f"Training Time: {training_time_sec/60:.1f} minutes")
        
        self.save_metrics()
    
    # Phase 3 Metrics
    def add_phase3_metrics(self,
                          protected_class_retention: float,
                          overall_accuracy: float,
                          num_protected_classes: int,
                          loss: float,
                          per_class_retention: Optional[Dict[int, float]] = None,
                          training_time_sec: Optional[float] = None):
        """Record Phase 3 (CONEC-LoRA) metrics.
        
        Args:
            protected_class_retention: Retention on protected (original) crop classes
            overall_accuracy: Overall accuracy on all classes
            num_protected_classes: Number of protected classes
            loss: Final validation loss
            per_class_retention: Per-class retention dictionary
            training_time_sec: Training time in seconds
        """
        self.metrics['phases']['phase3'] = {
            'protected_class_retention': protected_class_retention,
            'overall_accuracy': overall_accuracy,
            'num_protected_classes': num_protected_classes,
            'loss': loss,
            'per_class_retention': per_class_retention or {},
            'meets_target': protected_class_retention >= self.TARGETS['phase3_protected_retention'],
            'target': self.TARGETS['phase3_protected_retention'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log result
        status = "✅ PASS" if protected_class_retention >= self.TARGETS['phase3_protected_retention'] else "❌ FAIL"
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 3 (CONEC-LoRA) - {status}")
        logger.info(f"{'='*60}")
        logger.info(f"Protected Class Retention: {protected_class_retention:.4f} (target ≥ {self.TARGETS['phase3_protected_retention']:.2f})")
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
        logger.info(f"Protected Classes: {num_protected_classes}")
        logger.info(f"Loss: {loss:.4f}")
        if training_time_sec:
            logger.info(f"Training Time: {training_time_sec/60:.1f} minutes")
        
        self.save_metrics()
    
    # OOD Detection Metrics
    def add_ood_metrics(self,
                       auroc: float,
                       threshold_per_class: Optional[Dict[int, float]] = None,
                       samples_evaluated: Optional[int] = None):
        """Record OOD Detection metrics.
        
        Args:
            auroc: Area Under ROC curve for OOD detection
            threshold_per_class: Per-class OOD thresholds computed
            samples_evaluated: Number of samples used for evaluation
        """
        self.metrics['phases']['ood_detection'] = {
            'auroc': auroc,
            'threshold_per_class': threshold_per_class or {},
            'samples_evaluated': samples_evaluated,
            'meets_target': auroc >= self.TARGETS['ood_auroc'],
            'target': self.TARGETS['ood_auroc'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log result
        status = "✅ PASS" if auroc >= self.TARGETS['ood_auroc'] else "❌ FAIL"
        logger.info(f"\n{'='*60}")
        logger.info(f"OOD DETECTION - {status}")
        logger.info(f"{'='*60}")
        logger.info(f"AUROC: {auroc:.4f} (target ≥ {self.TARGETS['ood_auroc']:.2f})")
        if samples_evaluated:
            logger.info(f"Samples Evaluated: {samples_evaluated}")
        logger.info(f"Per-class Thresholds: {len(threshold_per_class or {})}")
        
        self.save_metrics()
    
    def get_summary(self) -> Dict:
        """Get overall summary of all metrics."""
        summary = {
            'phase1_result': self.get_phase1_result(),
            'phase2_result': self.get_phase2_result(),
            'phase3_result': self.get_phase3_result(),
            'ood_result': self.get_ood_result(),
            'overall_pass': self.overall_pass(),
            'timestamp': datetime.now().isoformat()
        }
        return summary
    
    def get_phase1_result(self) -> str:
        if not self.metrics['phases']['phase1']:
            return "NOT_RUN"
        metrics = self.metrics['phases']['phase1']
        status = "PASS" if metrics.get('meets_target') else "FAIL"
        accuracy = metrics.get('accuracy', 0)
        return f"{status} ({accuracy:.4f})"
    
    def get_phase2_result(self) -> str:
        if not self.metrics['phases']['phase2']:
            return "NOT_RUN"
        metrics = self.metrics['phases']['phase2']
        status = "PASS" if metrics.get('meets_target') else "FAIL"
        retention = metrics.get('old_diseases_retention', 0)
        return f"{status} ({retention:.4f})"
    
    def get_phase3_result(self) -> str:
        if not self.metrics['phases']['phase3']:
            return "NOT_RUN"
        metrics = self.metrics['phases']['phase3']
        status = "PASS" if metrics.get('meets_target') else "FAIL"
        retention = metrics.get('protected_class_retention', 0)
        return f"{status} ({retention:.4f})"
    
    def get_ood_result(self) -> str:
        if not self.metrics['phases']['ood_detection']:
            return "NOT_RUN"
        metrics = self.metrics['phases']['ood_detection']
        status = "PASS" if metrics.get('meets_target') else "FAIL"
        auroc = metrics.get('auroc', 0)
        return f"{status} ({auroc:.4f})"
    
    def overall_pass(self) -> bool:
        """Check if all phases meet v5.5 targets."""
        results = [
            self.metrics['phases']['phase1'].get('meets_target', False),
            self.metrics['phases']['phase2'].get('meets_target', False),
            self.metrics['phases']['phase3'].get('meets_target', False),
            self.metrics['phases']['ood_detection'].get('meets_target', False),
        ]
        return all(results)
    
    def print_report(self):
        """Print comprehensive performance report."""
        print("\n" + "="*70)
        print(" "*20 + "v5.5 PERFORMANCE ATTESTATION REPORT")
        print("="*70)
        
        summary = self.get_summary()
        
        print(f"\n📊 PHASE RESULTS:")
        print(f"  Phase 1 (DoRA):          {self.get_phase1_result()}")
        print(f"  Phase 2 (SD-LoRA):       {self.get_phase2_result()}")
        print(f"  Phase 3 (CONEC-LoRA):    {self.get_phase3_result()}")
        print(f"  OOD Detection:           {self.get_ood_result()}")
        
        print(f"\n🎯 v5.5 SPECIFICATION TARGETS:")
        print(f"  Phase 1 Accuracy:        ≥ {self.TARGETS['phase1_accuracy']:.2f}")
        print(f"  Phase 2 Retention:       ≥ {self.TARGETS['phase2_retention']:.2f}")
        print(f"  Phase 3 Protected Ret:   ≥ {self.TARGETS['phase3_protected_retention']:.2f}")
        print(f"  OOD AUROC:              ≥ {self.TARGETS['ood_auroc']:.2f}")
        
        overall_status = "✅ ALL TARGETS MET" if summary['overall_pass'] else "❌ SOME TARGETS NOT MET"
        print(f"\n{'='*70}")
        print(f"  {overall_status}")
        print(f"{'='*70}\n")
        
        # Save report
        report_file = self.output_dir / 'v55_performance_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(" "*20 + "v5.5 PERFORMANCE ATTESTATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Phase 1 (DoRA):          {self.get_phase1_result()}\n")
            f.write(f"Phase 2 (SD-LoRA):       {self.get_phase2_result()}\n")
            f.write(f"Phase 3 (CONEC-LoRA):    {self.get_phase3_result()}\n")
            f.write(f"OOD Detection:           {self.get_ood_result()}\n\n")
            f.write(f"Overall: {overall_status}\n")
        
        logger.info(f"Report saved to {report_file}")


# Convenience function for use in notebooks
def create_metrics_tracker(output_dir: str = None) -> V55PerformanceMetrics:
    """Create a v5.5 performance metrics tracker."""
    return V55PerformanceMetrics(output_dir)
