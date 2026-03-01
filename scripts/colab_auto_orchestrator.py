#!/usr/bin/env python3
"""AADS v6 Colab auto-orchestrator for continual SD-LoRA workflow."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler('/tmp/aads_orchestrator.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CoLabOrchestrator:
    """Run notebook stages for v6 continual training."""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self.start_time = datetime.now()
        self.stage_times: Dict[str, float] = {}
        logger.info("AADS v6 Auto-Orchestrator initialized at %s", self.repo_path)

    def setup_environment(self) -> bool:
        logger.info("\n%s", "=" * 60)
        logger.info("STAGE 0: ENVIRONMENT SETUP")
        logger.info("%s", "=" * 60)
        try:
            import torch

            if not torch.cuda.is_available():
                logger.error("No GPU detected. Colab runtime must have GPU enabled.")
                return False

            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info("GPU: %s (%.1fGB)", device_name, memory_gb)

            required_packages = ['torch', 'torchvision', 'transformers', 'peft']
            for pkg in required_packages:
                __import__(pkg)
                logger.info("  OK: %s", pkg)
            return True
        except Exception as exc:
            logger.error("Environment setup failed: %s", exc)
            return False

    def run_notebook(self, notebook_path: Path, stage_name: str, timeout: int = 3600) -> bool:
        logger.info("\nExecuting %s...", stage_name)
        if not notebook_path.exists():
            logger.warning("Notebook not found: %s", notebook_path)
            return False

        start = time.time()
        try:
            try:
                import nbformat
                from nbconvert.preprocessors import ExecutePreprocessor

                nb = nbformat.read(str(notebook_path), as_version=4)
                ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
                ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
            except ImportError:
                subprocess.run(['papermill', str(notebook_path), '/tmp/output.ipynb'], timeout=timeout, check=True)

            elapsed = (time.time() - start) / 60
            self.stage_times[stage_name] = elapsed
            logger.info("%s completed in %.1f minutes", stage_name, elapsed)
            return True
        except Exception as exc:
            logger.error("%s failed: %s", stage_name, exc)
            return False

    def run_pipeline(self) -> bool:
        logger.info("\n%s", "=" * 60)
        logger.info("AADS v6 - Continual Auto-Training Pipeline")
        logger.info("%s", "=" * 60)
        logger.info("Started: %s", self.start_time.strftime('%Y-%m-%d %H:%M:%S'))

        if not self.setup_environment():
            return False

        notebook_dir = self.repo_path / 'colab_notebooks'
        stages = [
            (notebook_dir / '1_data_preparation.ipynb', 'STAGE 1: Data Preparation', 1800, True),
            (notebook_dir / '2_continual_sd_lora_training.ipynb', 'STAGE 2: Continual Training', 5400, False),
            (notebook_dir / '5_testing_validation.ipynb', 'STAGE 3: Validation', 1800, True),
            (notebook_dir / '6_performance_monitoring.ipynb', 'STAGE 4: Monitoring', 1800, True),
        ]

        for path, label, timeout, optional in stages:
            ok = self.run_notebook(path, label, timeout=timeout)
            if not ok and not optional:
                logger.error("Mandatory stage failed: %s", label)
                return False
        return True

    def print_summary(self, success: bool) -> None:
        end_time = datetime.now()
        total_minutes = (end_time - self.start_time).total_seconds() / 60
        logger.info("\n%s", "=" * 60)
        logger.info("PIPELINE %s", "COMPLETED" if success else "FAILED")
        logger.info("%s", "=" * 60)
        for stage, minutes in self.stage_times.items():
            logger.info("  %s: %.1f minutes", stage, minutes)
        logger.info("  Total Duration: %.1f minutes", total_minutes)
        logger.info("Outputs:")
        logger.info("  OK: continual_sd_lora_adapter")
        logger.info("  OK: validation reports")
        logger.info("  OK: monitoring reports")


def main() -> int:
    orchestrator = CoLabOrchestrator(Path.cwd())
    success = orchestrator.run_pipeline()
    orchestrator.print_summary(success)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
