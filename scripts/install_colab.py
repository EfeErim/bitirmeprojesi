#!/usr/bin/env python3
"""
Automated Installation Script for Google Colab
Detects GPU, installs dependencies, and sets up the environment.
"""

import os
import sys
import subprocess
import json
import logging
import importlib
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.colab_contract import COLAB_WORKSPACE_PATH, required_workspace_paths, StepGate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ColabInstaller:
    """Handles Colab environment setup and dependency installation."""
    
    def __init__(self):
        self.gpu_info = None
        self.cuda_version = None
        self.pytorch_version = None
        self.workspace_dir = COLAB_WORKSPACE_PATH
        self.config_dir = self.workspace_dir / 'config'

    def _emit_gate(self, gate: StepGate) -> bool:
        if gate.passed:
            logger.info(gate.as_log_line())
            return True
        logger.error(gate.as_log_line())
        return False
        
    def detect_gpu(self) -> Dict:
        """Detect GPU type, memory, and CUDA version."""
        logger.info("🔍 Detecting GPU...")
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if not cuda_available:
                logger.warning("⚠️  CUDA is not available. Please enable GPU in Runtime → Change runtime type")
                return {
                    'available': False,
                    'type': 'None',
                    'memory_gb': 0,
                    'cuda_version': 'Unknown',
                    'device_count': 0
                }
            
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else 'Unknown'
            cuda_version = torch.version.cuda if hasattr(torch, 'version') else 'Unknown'
            
            # Get GPU memory
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_gb = total_memory / (1024**3)
            except Exception as e:
                logger.warning(f"Could not get GPU memory: {e}")
                memory_gb = 0
            
            gpu_info = {
                'available': True,
                'type': device_name,
                'memory_gb': round(memory_gb, 2),
                'cuda_version': cuda_version,
                'device_count': device_count
            }
            
            logger.info(f"✅ GPU detected: {device_name}")
            logger.info(f"   Memory: {gpu_info['memory_gb']} GB")
            logger.info(f"   CUDA: {cuda_version}")
            logger.info(f"   Devices: {device_count}")
            
            self.gpu_info = gpu_info
            return gpu_info
            
        except ImportError:
            logger.error("❌ PyTorch not installed. Will install appropriate version.")
            return {
                'available': False,
                'type': 'Unknown',
                'memory_gb': 0,
                'cuda_version': 'Unknown',
                'device_count': 0
            }
    
    def get_pytorch_install_command(self, cuda_version: str) -> str:
        """Determine PyTorch installation command based on CUDA version."""
        logger.info(f"Determining PyTorch version for CUDA {cuda_version}")
        
        # Map CUDA version to PyTorch version
        if cuda_version.startswith('11.8'):
            return 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
        elif cuda_version.startswith('11.7'):
            return 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117'
        elif cuda_version.startswith('12.1'):
            return 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
        elif cuda_version.startswith('12.2'):
            return 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122'
        else:
            # Default to CPU version if CUDA version not recognized
            logger.warning(f"CUDA version {cuda_version} not recognized. Installing CPU version.")
            return 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
    
    def install_dependencies(self) -> bool:
        """Install all required dependencies."""
        logger.info("📦 Installing dependencies...")
        
        # First, upgrade pip
        logger.info("Upgrading pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        
        # Install PyTorch if not available
        try:
            import torch
            logger.info(f"✅ PyTorch already installed: {torch.__version__}")
        except ImportError:
            logger.info("Installing PyTorch...")
            # Detect CUDA version from nvidia-smi if available
            cuda_version = self._detect_cuda_from_nvidia_smi()
            torch_cmd = self.get_pytorch_install_command(cuda_version)
            subprocess.run(torch_cmd, shell=True, check=True)
            logger.info("✅ PyTorch installed")
        
        # Install other dependencies
        requirements = [
            'transformers>=4.35.0',
            'peft>=0.6.0',
            'accelerate>=0.24.0',
            'datasets>=2.14.0',
            'evaluate>=0.4.0',
            'scikit-learn',
            'pandas',
            'numpy',
            'pillow',
            'tqdm',
            'psutil',
            'gdown',
            'fastapi',
            'uvicorn[standard]',
            'python-multipart',
            'pydantic>=2.0.0',
            'redis',
            'sqlalchemy',
            'requests',
            'tensorboard',
            'matplotlib',
            'seaborn',
            'jupyter',
            'ipywidgets'
        ]
        
        for req in requirements:
            logger.info(f"Installing {req}...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', req], check=True)
                logger.info(f"✅ {req} installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {req}: {e}")
                return False
        
        logger.info("✅ All dependencies installed")
        return True
    
    def _detect_cuda_from_nvidia_smi(self) -> str:
        """Detect CUDA version from nvidia-smi."""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Parse CUDA version from output
            for line in output.split('\n'):
                if 'CUDA Version' in line:
                    # Extract version like "12.2"
                    import re
                    match = re.search(r'CUDA Version: (\d+\.\d+)', line)
                    if match:
                        return match.group(1)
            
            # Default to 11.8 if not found
            logger.warning("Could not detect CUDA version from nvidia-smi. Defaulting to 11.8")
            return '11.8'
        except Exception as e:
            logger.warning(f"Could not run nvidia-smi: {e}. Defaulting to 11.8")
            return '11.8'
    
    def setup_workspace(self) -> bool:
        """Create workspace directory structure."""
        logger.info("📁 Setting up workspace...")
        
        try:
            # Create main workspace
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Workspace created: {self.workspace_dir}")
            
            # Create subdirectories
            dirs = [
                'data',
                'models',
                'checkpoints',
                'logs',
                'outputs',
                'cache',
                'config',
                'scripts',
                'src',
                'tests',
                'docs',
                'colab_notebooks'
            ]
            
            for d in dirs:
                (self.workspace_dir / d).mkdir(exist_ok=True)
                logger.debug(f"Created directory: {d}")
            
            logger.info("✅ Directory structure created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create workspace: {e}")
            return False
    
    def create_colab_config(self) -> bool:
        """Create Colab-specific configuration file."""
        logger.info("⚙️  Creating Colab configuration...")
        
        try:
            config_path = self.config_dir / 'colab.json'
            
            # Get GPU memory for batch size configuration
            memory_gb = self.gpu_info.get('memory_gb', 0) if self.gpu_info else 0
            
            config = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "AADS-ULoRA Colab Configuration",
                "description": "Google Colab-optimized configuration",
                "version": "5.5.3-colab",
                "last_updated": "2026-02-19",
                
                "colab": {
                    "enabled": True,
                    "environment": "google_colab",
                    "gpu_type": self.gpu_info.get('type', 'Unknown') if self.gpu_info else 'Unknown',
                    "gpu_memory_gb": memory_gb,
                    "drive_mount_path": "/content/drive/MyDrive/aads_ulora",
                    "workspace_path": str(self.workspace_dir),
                    "cache_dir": "/content/cache",
                    
                    "memory_optimization": {
                        "gradient_checkpointing": True,
                        "mixed_precision": True,
                        "memory_efficient_attention": True,
                        "clear_cache_frequency": 10,
                        "max_batch_size_gb4": 8,
                        "max_batch_size_gb8": 16,
                        "max_batch_size_gb16": 32,
                        "max_batch_size_gb24": 64,
                        "max_batch_size_gb32": 128
                    },
                    
                    "training": {
                        "gradient_accumulation_steps": 2,
                        "use_amp": True,
                        "pin_memory": True,
                        "num_workers": 2,
                        "prefetch_factor": 2,
                        "persistent_workers": False,
                        "checkpoint_interval": 5,
                        "early_stopping_patience": 10
                    }
                },
                
                "training": {
                    "phase1": {
                        "batch_size": self._get_batch_size_for_gpu(memory_gb, 8),
                        "gradient_accumulation_steps": 2
                    },
                    "phase2": {
                        "batch_size": self._get_batch_size_for_gpu(memory_gb, 4)
                    },
                    "phase3": {
                        "batch_size": self._get_batch_size_for_gpu(memory_gb, 16)
                    }
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Colab configuration created: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Colab config: {e}")
            return False
    
    def _get_batch_size_for_gpu(self, memory_gb: float, base_batch: int) -> int:
        """Adjust batch size based on GPU memory."""
        if memory_gb >= 32:
            return base_batch * 4
        elif memory_gb >= 24:
            return base_batch * 3
        elif memory_gb >= 16:
            return base_batch * 2
        elif memory_gb >= 8:
            return base_batch
        else:
            return max(base_batch // 2, 1)
    
    def create_requirements_file(self) -> bool:
        """Create requirements_colab.txt with pinned dependencies."""
        logger.info("📋 Creating requirements file...")
        
        try:
            req_path = self.workspace_dir / 'colab_notebooks' / 'requirements_colab.txt'
            req_path.parent.mkdir(exist_ok=True)
            
            requirements = """# AADS-ULoRA Colab Dependencies
# Generated automatically by install_colab.py

# Core ML
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.35.0
peft>=0.6.0
accelerate>=0.24.0
datasets>=2.14.0
evaluate>=0.4.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
pillow>=9.0.0
scikit-learn>=1.3.0

# Training & monitoring
tqdm>=4.65.0
psutil>=5.9.0
tensorboard>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0

# API & web
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Database
sqlalchemy>=2.0.0
redis>=4.6.0

# Utilities
requests>=2.31.0
gdown>=4.7.0
jupyter>=1.0.0
ipywidgets>=8.0.0

# Colab-specific
google-colab>=1.0.0
"""
            
            with open(req_path, 'w') as f:
                f.write(requirements)
            
            logger.info(f"✅ Requirements file created: {req_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create requirements file: {e}")
            return False

    def verify_bootstrap_ready(self) -> bool:
        """Run BOOTSTRAP_READY gate checks before training notebooks."""
        logger.info("🔎 Running BOOTSTRAP_READY preflight checks...")

        gates = [
            StepGate(
                step_id="BOOTSTRAP_READY",
                check_name="workspace_root_exists",
                passed=self.workspace_dir.exists(),
                expected=f"existing path: {self.workspace_dir}",
                actual=f"exists={self.workspace_dir.exists()}"
            ),
            StepGate(
                step_id="BOOTSTRAP_READY",
                check_name="config_file_exists",
                passed=(self.config_dir / 'colab.json').exists(),
                expected=f"file: {self.config_dir / 'colab.json'}",
                actual=f"exists={(self.config_dir / 'colab.json').exists()}"
            )
        ]

        for expected_path in required_workspace_paths(self.workspace_dir):
            gates.append(
                StepGate(
                    step_id="BOOTSTRAP_READY",
                    check_name=f"dir_{expected_path.name}",
                    passed=expected_path.exists() and expected_path.is_dir(),
                    expected=f"directory: {expected_path}",
                    actual=f"exists={expected_path.exists()} is_dir={expected_path.is_dir()}"
                )
            )

        # REPO_SYNC gate checks for critical imports
        modules_to_check = (
            'src.training.colab_phase1_training',
            'src.training.colab_phase2_sd_lora',
            'src.training.colab_phase3_conec_lora',
            'src.dataset.colab_datasets',
            'src.dataset.colab_dataloader',
        )
        for module_name in modules_to_check:
            try:
                module = importlib.import_module(module_name)
                module_path = Path(getattr(module, '__file__', '')).resolve() if getattr(module, '__file__', None) else None
                in_workspace = bool(module_path) and str(module_path).startswith(str(Path.cwd().resolve()))
                gates.append(
                    StepGate(
                        step_id='REPO_SYNC_OK',
                        check_name=f'import_{module_name}',
                        passed=module_path is not None and in_workspace,
                        expected=f'imported from workspace root {Path.cwd().resolve()}',
                        actual=str(module_path) if module_path else 'module has no __file__'
                    )
                )
            except Exception as import_error:
                gates.append(
                    StepGate(
                        step_id='REPO_SYNC_OK',
                        check_name=f'import_{module_name}',
                        passed=False,
                        expected='importable module',
                        actual=f'import error: {import_error}'
                    )
                )

        all_passed = True
        for gate in gates:
            all_passed = self._emit_gate(gate) and all_passed

        if all_passed:
            logger.info("✅ BOOTSTRAP_READY gate passed")
        else:
            logger.error("❌ BOOTSTRAP_READY gate failed")
        return all_passed
    
    def run(self) -> bool:
        """Run complete installation process."""
        logger.info("=" * 60)
        logger.info("🚀 Starting AADS-ULoRA Colab Installation")
        logger.info("=" * 60)
        
        steps = [
            ("GPU Detection", self.detect_gpu),
            ("Dependency Installation", self.install_dependencies),
            ("Workspace Setup", self.setup_workspace),
            ("Colab Config Creation", self.create_colab_config),
            ("Requirements File", self.create_requirements_file),
            ("Bootstrap Preflight Gate", self.verify_bootstrap_ready)
        ]
        
        success = True
        for step_name, step_func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Step: {step_name}")
            logger.info(f"{'='*60}")
            try:
                if not step_func():
                    success = False
                    logger.error(f"❌ Step '{step_name}' failed")
                    break
            except Exception as e:
                success = False
                logger.error(f"❌ Step '{step_name}' failed with exception: {e}")
                break
        
        if success:
            logger.info("\n" + "=" * 60)
            logger.info("✅ Installation completed successfully!")
            logger.info("=" * 60)
            logger.info("\nNext steps:")
            logger.info("1. Restart the runtime (Runtime → Restart runtime)")
            logger.info("2. Run the bootstrap notebook: colab_bootstrap.ipynb")
            logger.info("3. Start training with the training notebooks")
        else:
            logger.error("\n❌ Installation failed. Check logs above.")
        
        return success


def main():
    """Main entry point."""
    installer = ColabInstaller()
    success = installer.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()