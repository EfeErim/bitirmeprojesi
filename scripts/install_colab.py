#!/usr/bin/env python3
"""Install and bootstrap AADS v6 Colab workspace."""

from __future__ import annotations

import importlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from src.core.colab_contract import StepGate, required_workspace_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ColabInstaller:
    workspace_dir: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        self._sync_paths()
        self.gpu_info = self.detect_gpu()
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _sync_paths(self) -> None:
        """Normalize mutable path attributes used by test and runtime paths."""
        self.workspace_dir = Path(self.workspace_dir)
        self.config_dir = self.workspace_dir / 'config'

    def _detect_cuda_from_nvidia_smi(self) -> str:
        """Best-effort CUDA version probe; returns a stable fallback for local/dev."""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = f"{result.stdout}\n{result.stderr}"
            match = re.search(r'CUDA Version:\s*([0-9]+\.[0-9]+)', output)
            if match:
                return match.group(1)
        except Exception:
            pass
        return '11.8'

    def detect_gpu(self) -> Dict[str, object]:
        """Collect basic GPU facts used for Colab defaults."""
        try:
            import torch  # local import to support test monkeypatching
        except Exception:
            return {
                'available': False,
                'type': 'None',
                'memory_gb': 0,
                'cuda_version': 'Unknown',
                'device_count': 0,
            }

        if not torch.cuda.is_available():
            return {
                'available': False,
                'type': 'None',
                'memory_gb': 0,
                'cuda_version': 'Unknown',
                'device_count': 0,
            }

        device_count = int(torch.cuda.device_count())
        device_name = str(torch.cuda.get_device_name(0)) if device_count > 0 else 'Unknown'
        props = torch.cuda.get_device_properties(0) if device_count > 0 else None
        total_memory = float(getattr(props, 'total_memory', 0.0))
        memory_gb = round(total_memory / (1024 ** 3), 1) if total_memory > 0 else 0.0
        cuda_version = str(getattr(torch.version, 'cuda', None) or self._detect_cuda_from_nvidia_smi())
        return {
            'available': True,
            'type': device_name,
            'memory_gb': memory_gb,
            'cuda_version': cuda_version,
            'device_count': device_count,
        }

    def get_pytorch_install_command(self, cuda_version: str) -> list[str]:
        """Return pip command for a CUDA-compatible PyTorch wheel index."""
        wheel = 'cpu'
        if cuda_version == '11.8':
            wheel = 'cu118'
        elif cuda_version == '11.7':
            wheel = 'cu117'
        elif cuda_version == '12.1':
            wheel = 'cu121'
        return [
            sys.executable,
            '-m',
            'pip',
            'install',
            'torch',
            'torchvision',
            'torchaudio',
            '--index-url',
            f'https://download.pytorch.org/whl/{wheel}',
        ]

    @staticmethod
    def _get_batch_size_for_gpu(memory_gb: float, base_batch_size: int) -> int:
        """Scale batch size with available GPU memory."""
        if memory_gb >= 24:
            return max(1, base_batch_size * 4)
        if memory_gb >= 16:
            return max(1, base_batch_size * 2)
        if memory_gb >= 8:
            return max(1, base_batch_size)
        return max(1, base_batch_size // 2)

    def install_dependencies(self) -> bool:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_colab.txt'], check=True)
            return True
        except Exception as exc:
            logger.error('Dependency install failed: %s', exc)
            return False

    def setup_workspace(self) -> bool:
        try:
            self._sync_paths()
            for p in required_workspace_paths(self.workspace_dir):
                p.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as exc:
            logger.error('Workspace setup failed: %s', exc)
            return False

    def create_colab_config(self) -> bool:
        self._sync_paths()
        if not isinstance(getattr(self, 'gpu_info', None), dict):
            self.gpu_info = self.detect_gpu()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        suggested_batch = self._get_batch_size_for_gpu(float(self.gpu_info.get('memory_gb', 0) or 0), base_batch_size=8)
        cfg = {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'title': 'AADS v6 Colab Configuration',
            'description': 'Google Colab-optimized configuration for AADS v6 continual runtime',
            'version': '6.0.0-colab',
            'last_updated': '2026-03-01',
            'colab': {
                'enabled': True,
                'environment': 'google_colab',
                'workspace_path': str(self.workspace_dir),
                'gpu_type': str(self.gpu_info.get('type', 'None')),
            },
            'training': {
                'continual': {
                    'backbone': {'model_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m'},
                    'quantization': {'mode': 'int8_hybrid', 'strict_backend': True, 'allow_cpu_fallback': False},
                    'adapter': {
                        'target_modules_strategy': 'all_linear_transformer',
                        'lora_r': 16,
                        'lora_alpha': 32,
                        'lora_dropout': 0.1,
                    },
                    'fusion': {'layers': [2, 5, 8, 11], 'output_dim': 768, 'dropout': 0.1, 'gating': 'softmax'},
                    'ood': {'threshold_factor': 2.0},
                    'learning_rate': 1e-4,
                    'weight_decay': 0.0,
                    'num_epochs': 3,
                    'batch_size': int(suggested_batch),
                    'device': 'cuda',
                    'strict_model_loading': True,
                }
            },
        }
        try:
            (self.config_dir / 'colab.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            return True
        except Exception as exc:
            logger.error('Failed to write config: %s', exc)
            return False

    def create_requirements_file(self) -> bool:
        """Create a Colab-local requirements file for notebook bootstrap."""
        try:
            self._sync_paths()
            target_dir = self.workspace_dir / 'colab_notebooks'
            target_dir.mkdir(parents=True, exist_ok=True)
            content = "\n".join(
                [
                    "torch>=2.2.0",
                    "transformers>=4.40.0",
                    "peft>=0.10.0",
                    "accelerate>=0.29.0",
                ]
            )
            (target_dir / 'requirements_colab.txt').write_text(content + "\n", encoding='utf-8')
            return True
        except Exception as exc:
            logger.error('Failed to create notebook requirements: %s', exc)
            return False

    def verify_runtime_imports(self) -> bool:
        gates = []
        modules = (
            'src.training.continual_sd_lora',
            'src.training.quantization',
            'src.adapter.independent_crop_adapter',
            'src.ood.continual_ood',
        )
        for name in modules:
            try:
                mod = importlib.import_module(name)
                module_path = Path(getattr(mod, '__file__', '')).resolve()
                gates.append(
                    StepGate(
                        step_id='RUNTIME_IMPORTS',
                        check_name=name,
                        passed=module_path.exists(),
                        expected='importable module',
                        actual=str(module_path),
                    )
                )
            except Exception as exc:
                gates.append(
                    StepGate(
                        step_id='RUNTIME_IMPORTS',
                        check_name=name,
                        passed=False,
                        expected='importable module',
                        actual=f'import error: {exc}',
                    )
                )

        ok = True
        for gate in gates:
            logger.info(gate.as_log_line())
            ok = ok and gate.passed
        return ok

    def run(self) -> bool:
        steps = [
            ('Workspace setup', self.setup_workspace),
            ('Config generation', self.create_colab_config),
            ('Runtime import checks', self.verify_runtime_imports),
        ]
        for name, fn in steps:
            logger.info('Running: %s', name)
            if not fn():
                logger.error('Step failed: %s', name)
                return False
        return True


def main() -> int:
    installer = ColabInstaller(workspace_dir=Path.cwd())
    ok = installer.run()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

