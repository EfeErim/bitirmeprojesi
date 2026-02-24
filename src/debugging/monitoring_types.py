#!/usr/bin/env python3
"""Shared monitoring dataclasses for performance monitoring."""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any


@dataclass
class GPUMetrics:
    timestamp: float
    utilization: Optional[float] = None
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None
    memory_util: Optional[float] = None
    temperature: Optional[float] = None
    power_draw_w: Optional[float] = None
    fan_speed: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    timestamp: float
    epoch: int
    batch: int
    samples_processed: int
    batch_time: float
    epoch_time: Optional[float] = None
    throughput: Optional[float] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_allocated_gb: Optional[float] = None
    memory_reserved_gb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryProfile:
    timestamp: float
    allocated_gb: float
    reserved_gb: float
    peak_allocated_gb: float
    peak_reserved_gb: float
    fragmentation_ratio: float
    allocation_events: int
    leak_suspected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriveIOMetrics:
    timestamp: float
    read_speed_mb_s: Optional[float] = None
    write_speed_mb_s: Optional[float] = None
    iops: Optional[int] = None
    avg_latency_ms: Optional[float] = None
    active_transfers: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
