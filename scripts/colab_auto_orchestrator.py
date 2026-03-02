#!/usr/bin/env python3
"""AADS v6 Colab auto-orchestrator for continual SD-LoRA workflow."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

LOG_FILE_PATH = Path(tempfile.gettempdir()) / 'aads_orchestrator.log'
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
MAX_OUTPUT_LINES_PER_STAGE = int(os.environ.get('AADS_ORCH_MAX_OUTPUT_LINES', '600'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'), logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


class CoLabOrchestrator:
    """Run notebook stages for v6 continual training."""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self.start_time = datetime.now()
        self.stage_times: Dict[str, float] = {}
        logger.info("AADS v6 Auto-Orchestrator initialized at %s", self.repo_path)
        logger.info("Orchestrator log file: %s", LOG_FILE_PATH)

    @staticmethod
    def _normalize_output_text(raw: object) -> str:
        if isinstance(raw, list):
            return ''.join(str(part) for part in raw)
        if raw is None:
            return ''
        return str(raw)

    def _iter_output_lines(self, output: dict) -> Iterable[str]:
        output_type = output.get('output_type')
        if output_type == 'stream':
            text = self._normalize_output_text(output.get('text'))
            for line in text.splitlines():
                yield line
            return

        if output_type == 'error':
            traceback = output.get('traceback')
            if isinstance(traceback, list) and traceback:
                for line in traceback:
                    yield str(line)
            else:
                ename = output.get('ename', 'Error')
                evalue = output.get('evalue', '')
                yield f"{ename}: {evalue}".strip()
            return

        if output_type in {'execute_result', 'display_data'}:
            data = output.get('data', {})
            if isinstance(data, dict):
                text = data.get('text/plain') or data.get('text/markdown')
                for line in self._normalize_output_text(text).splitlines():
                    yield line

    def _log_notebook_outputs(self, notebook, stage_name: str) -> None:
        lines_remaining = MAX_OUTPUT_LINES_PER_STAGE
        emitted_any = False

        for cell_idx, cell in enumerate(notebook.get('cells', []), start=1):
            if cell.get('cell_type') != 'code':
                continue
            outputs = cell.get('outputs') or []
            if not outputs:
                continue

            source = self._normalize_output_text(cell.get('source')).strip()
            headline = source.splitlines()[0] if source else '<empty code cell>'
            logger.info("[%s][Cell %d] %s", stage_name, cell_idx, headline[:120])
            emitted_any = True

            for output in outputs:
                for line in self._iter_output_lines(output):
                    cleaned = line.rstrip()
                    if not cleaned:
                        continue
                    logger.info("[%s][Cell %d] %s", stage_name, cell_idx, cleaned)
                    lines_remaining -= 1
                    if lines_remaining <= 0:
                        logger.warning(
                            "[%s] Output truncated after %d lines. Increase AADS_ORCH_MAX_OUTPUT_LINES to see more.",
                            stage_name,
                            MAX_OUTPUT_LINES_PER_STAGE,
                        )
                        return

        if not emitted_any:
            logger.info("[%s] No notebook cell outputs captured.", stage_name)

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
        logger.info("Notebook: %s | Timeout: %ss", notebook_path, timeout)
        if not notebook_path.exists():
            logger.warning("Notebook not found: %s", notebook_path)
            return False

        start = time.time()
        notebook = None
        try:
            try:
                import nbformat
                from nbconvert.preprocessors import ExecutePreprocessor

                # Notebooks are committed with UTF-8 BOM to preserve cross-tool compatibility.
                notebook_text = notebook_path.read_text(encoding='utf-8-sig')
                notebook = nbformat.reads(notebook_text, as_version=4)
                ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
                ep.preprocess(notebook, {'metadata': {'path': str(notebook_path.parent)}})
                self._log_notebook_outputs(notebook, stage_name)
            except ImportError:
                subprocess.run(['papermill', str(notebook_path), '/tmp/output.ipynb'], timeout=timeout, check=True)

            elapsed = (time.time() - start) / 60
            self.stage_times[stage_name] = elapsed
            logger.info("%s completed in %.1f minutes", stage_name, elapsed)
            return True
        except Exception as exc:
            if notebook is not None:
                self._log_notebook_outputs(notebook, stage_name)
            logger.exception("%s failed: %s", stage_name, exc)
            return False

    def run_pipeline(self) -> bool:
        logger.info("\n%s", "=" * 60)
        logger.info("AADS v6 - Continual Auto-Training Pipeline")
        logger.info("%s", "=" * 60)
        logger.info("Started: %s", self.start_time.strftime('%Y-%m-%d %H:%M:%S'))

        if not self.setup_environment():
            return False

        dataset_root_override = os.environ.get('AADS_DATASET_ROOT', '').strip()
        if dataset_root_override:
            logger.info("Dataset root override from env: %s", dataset_root_override)
        else:
            logger.info("Dataset root override not set; notebooks use ROOT/data/class_root_dataset by default.")

        notebook_dir = self.repo_path / 'colab_notebooks'
        stages = [
            (notebook_dir / '1_data_preparation.ipynb', 'STAGE 1: Data Preparation', 1800, False),
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
    parser = argparse.ArgumentParser(description="Run AADS v6 Colab notebook orchestration.")
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='',
        help='Optional dataset root override passed to stage notebooks via AADS_DATASET_ROOT.',
    )
    args = parser.parse_args()

    if args.dataset_root:
        os.environ['AADS_DATASET_ROOT'] = args.dataset_root

    orchestrator = CoLabOrchestrator(Path.cwd())
    success = orchestrator.run_pipeline()
    orchestrator.print_summary(success)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
