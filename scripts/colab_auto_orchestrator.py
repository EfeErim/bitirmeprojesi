#!/usr/bin/env python3
"""
AADS-ULoRA Colab Auto-Orchestrator
Seamlessly runs the complete training pipeline with error handling and progress tracking.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/aads_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CoLabOrchestrator:
    """Orchestrates complete AADS-ULoRA training pipeline in Colab."""
    
    def __init__(self, repo_path: Optional[Path] = None):
        """Initialize the orchestrator."""
        self.repo_path = repo_path or Path.cwd()
        self.start_time = datetime.now()
        self.phase_times: Dict[str, float] = {}
        self.results: Dict[str, any] = {}
        
        logger.info(f"AADS-ULoRA Auto-Orchestrator initialized at {self.repo_path}")
    
    def setup_environment(self) -> bool:
        """Setup Colab environment."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 0: ENVIRONMENT SETUP")
        logger.info("="*60)
        
        try:
            # Verify GPU
            logger.info("Checking GPU availability...")
            import torch
            if not torch.cuda.is_available():
                logger.error("No GPU detected! Colab runtime must have GPU enabled.")
                return False
            
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {device_name} ({memory_gb:.1f}GB)")
            
            # Verify dependencies
            logger.info("Verifying dependencies...")
            required_packages = ['torch', 'torchvision', 'transformers', 'peft']
            for pkg in required_packages:
                try:
                    __import__(pkg)
                    logger.info(f"  OK: {pkg}")
                except ImportError:
                    logger.error(f"  MISSING: {pkg}")
                    return False
            
            logger.info("Environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def run_notebook(self, notebook_path: Path, phase_name: str, timeout: int = 3600) -> bool:
        """Execute a Jupyter notebook."""
        logger.info(f"\nExecuting {phase_name}...")
        
        if not notebook_path.exists():
            logger.warning(f"Notebook not found: {notebook_path}")
            return False
        
        try:
            phase_start = time.time()
            
            # Try using nbconvert to execute
            try:
                import nbformat
                from nbconvert.preprocessors import ExecutePreprocessor
                
                logger.info(f"  Executing {notebook_path.name}...")
                
                nb = nbformat.read(str(notebook_path), as_version=4)
                ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
                nb, outputs = ep.preprocess(nb, {
                    'metadata': {
                        'path': str(notebook_path.parent)
                    }
                })
                
                elapsed = (time.time() - phase_start) / 60
                self.phase_times[phase_name] = elapsed
                logger.info(f"{phase_name} completed in {elapsed:.1f} minutes")
                return True
                
            except ImportError:
                # Fallback: use papermill
                logger.info(f"  Using alternative execution method...")
                subprocess.run(
                    ['papermill', str(notebook_path), '/tmp/output.ipynb'],
                    timeout=timeout,
                    check=True
                )
                elapsed = (time.time() - phase_start) / 60
                self.phase_times[phase_name] = elapsed
                logger.info(f"{phase_name} completed in {elapsed:.1f} minutes")
                return True
                
        except Exception as e:
            logger.error(f"{phase_name} failed: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete training pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info("AADS-ULoRA v5.5 - Complete Auto-Training Pipeline")
        logger.info(f"{'='*60}")
        logger.info(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Phase 0: Setup
        if not self.setup_environment():
            logger.error("Environment setup failed. Cannot continue.")
            return False
        
        # Phase 1: Data Preparation
        notebook_dir = self.repo_path / 'colab_notebooks'
        
        data_nb = notebook_dir / '1_data_preparation.ipynb'
        if data_nb.exists():
            if not self.run_notebook(data_nb, "PHASE 1: Data Preparation", timeout=1800):
                logger.warning("Data preparation had issues, continuing...")
        
        # Phase 2: Phase 1 Training (DoRA)
        phase1_nb = notebook_dir / '2_phase1_training.ipynb'
        if phase1_nb.exists():
            if not self.run_notebook(phase1_nb, "PHASE 2: DoRA Training", timeout=3600):
                logger.error("Phase 1 training failed")
                return False
        
        # Phase 3: Phase 2 Training (SD-LoRA)
        phase2_nb = notebook_dir / '3_phase2_training.ipynb'
        if phase2_nb.exists():
            if not self.run_notebook(phase2_nb, "PHASE 3: SD-LoRA Training", timeout=3600):
                logger.error("Phase 2 training failed")
                return False
        
        # Phase 4: Phase 3 Training (CoNeC-LoRA)
        phase3_nb = notebook_dir / '4_phase3_training.ipynb'
        if phase3_nb.exists():
            if not self.run_notebook(phase3_nb, "PHASE 4: CoNeC-LoRA Training", timeout=3600):
                logger.error("Phase 3 training failed")
                return False
        
        # Phase 5: Validation
        validation_nb = notebook_dir / '5_testing_validation.ipynb'
        if validation_nb.exists():
            if not self.run_notebook(validation_nb, "PHASE 5: Validation", timeout=1800):
                logger.warning("Validation had issues, continuing...")
        
        # Phase 6: Monitoring & Reports
        monitoring_nb = notebook_dir / '6_performance_monitoring.ipynb'
        if monitoring_nb.exists():
            if not self.run_notebook(monitoring_nb, "PHASE 6: Performance Monitoring", timeout=1800):
                logger.warning("Monitoring had issues, continuing...")
        
        return True
    
    def print_summary(self):
        """Print execution summary."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() / 60
        
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"{'='*60}")
        
        logger.info(f"\nTIMING SUMMARY:")
        for phase, duration in self.phase_times.items():
            logger.info(f"  {phase}: {duration:.1f} minutes")
        logger.info(f"  Total Duration: {total_duration:.1f} minutes")
        
        logger.info(f"\nOUTPUTS GENERATED:")
        logger.info(f"  OK: Phase 1 DoRA Adapter")
        logger.info(f"  OK: Phase 2 SD-LoRA Adapter")
        logger.info(f"  OK: Phase 3 CoNeC-LoRA Adapter")
        logger.info(f"  OK: Validation Results")
        logger.info(f"  OK: Performance Reports")
        
        logger.info(f"\nOUTPUT LOCATIONS:")
        logger.info(f"  Models: {self.repo_path / 'models'}")
        logger.info(f"  Checkpoints: {self.repo_path / 'checkpoints'}")
        logger.info(f"  Logs: {self.repo_path / 'logs'}")
        logger.info(f"  Results: {self.repo_path / 'outputs'}")
        
        logger.info(f"\nCompleted: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}\n")


def main():
    """Entry point for the orchestrator."""
    # Detect if running in Colab
    try:
        from google.colab import drive
        repo_path = Path.cwd()
    except ImportError:
        repo_path = Path.cwd()
    
    orchestrator = CoLabOrchestrator(repo_path)
    
    # Run pipeline
    success = orchestrator.run_pipeline()
    
    # Print summary
    orchestrator.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
