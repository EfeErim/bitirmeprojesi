"""
Colab Environment Detection and Validation Tests
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.install_colab import ColabInstaller


class TestGPUDetection:
    """Test GPU detection functionality."""

    def test_detect_gpu_with_cuda(self):
        """Test GPU detection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_name', return_value='Tesla T4'):
                    with patch('torch.cuda.get_device_properties') as mock_props:
                        mock_props.return_value.total_memory = 15 * 1024**3  # 15GB
                        
                        installer = ColabInstaller()
                        gpu_info = installer.detect_gpu()
                        
                        assert gpu_info['available'] is True
                        assert gpu_info['type'] == 'Tesla T4'
                        assert gpu_info['memory_gb'] == 15.0
                        assert gpu_info['device_count'] == 1

    def test_detect_gpu_without_cuda(self):
        """Test GPU detection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            installer = ColabInstaller()
            gpu_info = installer.detect_gpu()
            
            assert gpu_info['available'] is False
            assert gpu_info['type'] == 'None'
            assert gpu_info['memory_gb'] == 0

    def test_detect_gpu_pytorch_not_installed(self):
        """Test GPU detection when PyTorch is not installed."""
        with patch.dict('sys.modules', {'torch': None}):
            installer = ColabInstaller()
            gpu_info = installer.detect_gpu()
            
            assert gpu_info['available'] is False
            assert gpu_info['cuda_version'] == 'Unknown'


class TestPyTorchInstallation:
    """Test PyTorch installation logic."""

    def test_get_pytorch_install_command_cuda118(self):
        """Test PyTorch install command for CUDA 11.8."""
        installer = ColabInstaller()
        cmd = installer.get_pytorch_install_command('11.8')
        assert any('cu118' in part for part in cmd)

    def test_get_pytorch_install_command_cuda117(self):
        """Test PyTorch install command for CUDA 11.7."""
        installer = ColabInstaller()
        cmd = installer.get_pytorch_install_command('11.7')
        assert any('cu117' in part for part in cmd)

    def test_get_pytorch_install_command_cuda121(self):
        """Test PyTorch install command for CUDA 12.1."""
        installer = ColabInstaller()
        cmd = installer.get_pytorch_install_command('12.1')
        assert any('cu121' in part for part in cmd)

    def test_get_pytorch_install_command_unknown(self):
        """Test PyTorch install command for unknown CUDA version."""
        installer = ColabInstaller()
        cmd = installer.get_pytorch_install_command('unknown')
        assert any('cpu' in part for part in cmd)

    def test_detect_cuda_from_nvidia_smi(self):
        """Test CUDA detection from nvidia-smi."""
        installer = ColabInstaller()
        
        mock_result = MagicMock()
        mock_result.stdout = "CUDA Version: 11.8"
        
        with patch('subprocess.run', return_value=mock_result):
            cuda_version = installer._detect_cuda_from_nvidia_smi()
            assert cuda_version == '11.8'

    def test_detect_cuda_from_nvidia_smi_fallback(self):
        """Test fallback when nvidia-smi fails."""
        installer = ColabInstaller()
        
        with patch('subprocess.run', side_effect=Exception("Command failed")):
            cuda_version = installer._detect_cuda_from_nvidia_smi()
            assert cuda_version == '11.8'  # Default fallback


class TestWorkspaceSetup:
    """Test workspace directory setup."""

    def test_setup_workspace(self, tmp_path):
        """Test workspace directory creation."""
        installer = ColabInstaller()
        installer.workspace_dir = tmp_path / "test_workspace"
        
        result = installer.setup_workspace()
        
        assert result is True
        assert installer.workspace_dir.exists()
        assert (installer.workspace_dir / 'data').exists()
        assert (installer.workspace_dir / 'models').exists()
        assert (installer.workspace_dir / 'checkpoints').exists()
        assert (installer.workspace_dir / 'logs').exists()
        assert (installer.workspace_dir / 'colab_notebooks').exists()


class TestConfigCreation:
    """Test Colab configuration creation."""

    def test_create_colab_config(self, tmp_path):
        """Test Colab config file creation."""
        installer = ColabInstaller()
        installer.workspace_dir = tmp_path / "test_workspace"
        installer.workspace_dir.mkdir()
        installer.config_dir = installer.workspace_dir / 'config'
        installer.config_dir.mkdir()
        
        installer.gpu_info = {
            'available': True,
            'type': 'Tesla T4',
            'memory_gb': 15.0,
            'cuda_version': '11.8',
            'device_count': 1
        }
        
        result = installer.create_colab_config()
        
        assert result is True
        config_path = installer.config_dir / 'colab.json'
        assert config_path.exists()
        
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert config['colab']['enabled'] is True
        assert config['colab']['gpu_type'] == 'Tesla T4'
        assert 'training' in config

    def test_batch_size_adjustment(self):
        """Test batch size adjustment based on GPU memory."""
        installer = ColabInstaller()
        
        # Test different GPU memory sizes
        assert installer._get_batch_size_for_gpu(32, 8) == 32  # 32GB -> 8*4
        assert installer._get_batch_size_for_gpu(16, 8) == 16  # 16GB -> 8*2
        assert installer._get_batch_size_for_gpu(8, 8) == 8    # 8GB -> 8*1
        assert installer._get_batch_size_for_gpu(4, 8) == 4    # 4GB -> 8/2


class TestRequirementsFile:
    """Test requirements file creation."""

    def test_create_requirements_file(self, tmp_path):
        """Test requirements file creation."""
        installer = ColabInstaller()
        installer.workspace_dir = tmp_path / "test_workspace"
        (installer.workspace_dir / 'colab_notebooks').mkdir(parents=True)
        
        result = installer.create_requirements_file()
        
        assert result is True
        req_path = installer.workspace_dir / 'colab_notebooks' / 'requirements_colab.txt'
        assert req_path.exists()
        
        content = req_path.read_text()
        assert 'torch' in content
        assert 'transformers' in content
        assert 'peft' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
