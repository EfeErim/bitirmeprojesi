#!/usr/bin/env python3
"""Collector implementations for performance monitoring."""

import logging
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.debugging.monitoring_types import GPUMetrics, MemoryProfile, DriveIOMetrics

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitors GPU metrics using NVIDIA tools."""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._history: deque = deque(maxlen=1000)
        self._current: Optional[GPUMetrics] = None
        self._nvidia_smi_available = self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> bool:
        try:
            subprocess.run(['nvidia-smi', '--version'], capture_output=True, check=True, timeout=2)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("nvidia-smi not available, GPU monitoring disabled")
            return False

    def start(self):
        if not self._running and self._nvidia_smi_available:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("GPU monitor started")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("GPU monitor stopped")

    def _monitor_loop(self):
        while self._running:
            try:
                metrics = self._collect_gpu_metrics()
                with self._lock:
                    self._current = metrics
                    self._history.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
            time.sleep(self.update_interval)

    def _collect_gpu_metrics(self) -> GPUMetrics:
        metrics = GPUMetrics(timestamp=time.time())

        if not self._nvidia_smi_available:
            return metrics

        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,fan.speed',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) >= 7:
                    metrics.utilization = float(values[0]) if values[0] != 'N/A' else None
                    metrics.memory_util = float(values[1]) if values[1] != 'N/A' else None
                    metrics.memory_used_gb = float(values[2]) / 1024
                    metrics.memory_total_gb = float(values[3]) / 1024
                    metrics.temperature = float(values[4]) if values[4] != 'N/A' else None
                    metrics.power_draw_w = float(values[5]) if values[5] != 'N/A' else None
                    metrics.fan_speed = float(values[6]) if values[6] != 'N/A' else None
        except Exception as e:
            logger.debug(f"Failed to query nvidia-smi: {e}")

        return metrics

    def get_current(self) -> Optional[GPUMetrics]:
        with self._lock:
            return self._current

    def get_history(self, last_n: int = 100) -> List[GPUMetrics]:
        with self._lock:
            return list(self._history)[-last_n:]

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {}

            utilizations = [m.utilization for m in self._history if m.utilization is not None]
            memory_utils = [m.memory_util for m in self._history if m.memory_util is not None]
            temperatures = [m.temperature for m in self._history if m.temperature is not None]

            return {
                'gpu_available': True,
                'nvidia_smi_available': self._nvidia_smi_available,
                'avg_utilization': np.mean(utilizations) if utilizations else None,
                'max_utilization': np.max(utilizations) if utilizations else None,
                'avg_memory_util': np.mean(memory_utils) if memory_utils else None,
                'max_memory_util': np.max(memory_utils) if memory_utils else None,
                'avg_temperature': np.mean(temperatures) if temperatures else None,
                'max_temperature': np.max(temperatures) if temperatures else None,
                'samples': len(self._history)
            }


class MemoryProfiler:
    """Profiles GPU memory usage and detects leaks."""

    def __init__(self, leak_threshold: float = 0.05, check_interval: int = 100):
        self.leak_threshold = leak_threshold
        self.check_interval = check_interval
        self._lock = threading.RLock()
        self._allocations: List[float] = []
        self._peak_allocated = 0.0
        self._peak_reserved = 0.0
        self._allocation_history: deque = deque(maxlen=1000)
        self._snapshots: List[Tuple[float, float, float]] = []

    def snapshot(self, step: int) -> MemoryProfile:
        if not torch.cuda.is_available():
            return MemoryProfile(
                timestamp=time.time(),
                allocated_gb=0.0,
                reserved_gb=0.0,
                peak_allocated_gb=0.0,
                peak_reserved_gb=0.0,
                fragmentation_ratio=0.0,
                allocation_events=0
            )

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        with self._lock:
            self._allocations.append(allocated)
            self._peak_allocated = max(self._peak_allocated, allocated)
            self._peak_reserved = max(self._peak_reserved, reserved)
            self._snapshots.append((time.time(), allocated, reserved))

            leak_suspected = self._detect_memory_leak(allocated, reserved)

            profile = MemoryProfile(
                timestamp=time.time(),
                allocated_gb=allocated,
                reserved_gb=reserved,
                peak_allocated_gb=self._peak_allocated,
                peak_reserved_gb=self._peak_reserved,
                fragmentation_ratio=reserved / allocated if allocated > 0 else 0.0,
                allocation_events=len(self._allocations),
                leak_suspected=leak_suspected
            )
            self._allocation_history.append(profile)

        return profile

    def _detect_memory_leak(self, allocated: float, reserved: float) -> bool:
        if len(self._snapshots) < 10:
            return False

        recent = self._snapshots[-10:]
        if len(recent) < 2:
            return False

        start_alloc = recent[0][1]
        end_alloc = recent[-1][1]
        time_diff = recent[-1][0] - recent[0][0]

        if time_diff < 1.0:
            return False

        growth_rate = (end_alloc - start_alloc) / time_diff

        if growth_rate > self.leak_threshold / 10:
            increasing = all(recent[i][1] <= recent[i + 1][1] for i in range(len(recent) - 1))
            if increasing and growth_rate > 0.01:
                logger.warning(f"Potential memory leak detected: {growth_rate * 1000:.1f} MB/s growth rate")
                return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            if not self._allocation_history:
                return {}

            allocations = [p.allocated_gb for p in self._allocation_history]
            reserved = [p.reserved_gb for p in self._allocation_history]
            recent_profiles = list(self._allocation_history)[-10:]
            leak_suspected = any(p.leak_suspected for p in recent_profiles)

            return {
                'avg_allocated_gb': float(np.mean(allocations)),
                'max_allocated_gb': float(np.max(allocations)),
                'avg_reserved_gb': float(np.mean(reserved)),
                'max_reserved_gb': float(np.max(reserved)),
                'peak_allocated_gb': self._peak_allocated,
                'peak_reserved_gb': self._peak_reserved,
                'current_allocated_gb': allocations[-1] if allocations else 0.0,
                'current_reserved_gb': reserved[-1] if reserved else 0.0,
                'fragmentation_ratio': reserved[-1] / allocations[-1] if allocations and allocations[-1] > 0 else 0.0,
                'samples': len(self._allocation_history),
                'leak_suspected': leak_suspected,
            }

    def reset(self):
        with self._lock:
            self._allocations.clear()
            self._snapshots.clear()
            self._allocation_history.clear()
            self._peak_allocated = 0.0
            self._peak_reserved = 0.0


class DriveIOMonitor:
    """Monitors Google Drive I/O performance (Colab-specific)."""

    def __init__(self, drive_mount_path: str = '/content/drive', update_interval: float = 2.0):
        self.drive_mount_path = Path(drive_mount_path)
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._history: deque = deque(maxlen=100)
        self._current: Optional[DriveIOMetrics] = None
        self._is_mounted = self.drive_mount_path.exists()

    def start(self):
        if not self._running and self._is_mounted:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("Drive I/O monitor started")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("Drive I/O monitor stopped")

    def _monitor_loop(self):
        while self._running:
            try:
                metrics = self._collect_drive_metrics()
                with self._lock:
                    self._current = metrics
                    self._history.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting Drive metrics: {e}")
            time.sleep(self.update_interval)

    def _collect_drive_metrics(self) -> DriveIOMetrics:
        metrics = DriveIOMetrics(timestamp=time.time())

        try:
            result = subprocess.run(['iostat', '-x', '1', '1'], capture_output=True, text=True, timeout=3)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '/dev/sd' in line or '/dev/nvme' in line:
                        parts = line.split()
                        if len(parts) >= 7:
                            try:
                                metrics.read_speed_mb_s = float(parts[5])
                                metrics.write_speed_mb_s = float(parts[6])
                            except (ValueError, IndexError):
                                pass
                        break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            metrics = self._estimate_drive_metrics()

        return metrics

    def _estimate_drive_metrics(self) -> DriveIOMetrics:
        metrics = DriveIOMetrics(timestamp=time.time())

        try:
            test_file = self.drive_mount_path / '.perf_test.tmp'
            test_data = b'x' * (1024 * 1024)

            start = time.time()
            with open(test_file, 'wb') as f:
                f.write(test_data)
            write_time = time.time() - start

            start = time.time()
            with open(test_file, 'rb') as f:
                _ = f.read()
            read_time = time.time() - start

            test_file.unlink(missing_ok=True)

            if write_time > 0:
                metrics.write_speed_mb_s = len(test_data) / (1024**2) / write_time
            if read_time > 0:
                metrics.read_speed_mb_s = len(test_data) / (1024**2) / read_time

            metrics.iops = int(1.0 / max(write_time, read_time)) if max(write_time, read_time) > 0 else None
            metrics.avg_latency_ms = max(write_time, read_time) * 1000

        except Exception as e:
            logger.debug(f"Drive benchmark failed: {e}")

        return metrics

    def get_current(self) -> Optional[DriveIOMetrics]:
        with self._lock:
            return self._current

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._history:
                return {'mounted': self._is_mounted}

            read_speeds = [m.read_speed_mb_s for m in self._history if m.read_speed_mb_s is not None]
            write_speeds = [m.write_speed_mb_s for m in self._history if m.write_speed_mb_s is not None]

            return {
                'mounted': self._is_mounted,
                'avg_read_speed_mb_s': float(np.mean(read_speeds)) if read_speeds else None,
                'avg_write_speed_mb_s': float(np.mean(write_speeds)) if write_speeds else None,
                'max_read_speed_mb_s': float(np.max(read_speeds)) if read_speeds else None,
                'max_write_speed_mb_s': float(np.max(write_speeds)) if write_speeds else None,
                'samples': len(self._history)
            }
