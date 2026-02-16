"""
Centralized pipeline orchestration and management.
Provides unified interface for pipeline execution, monitoring, and lifecycle management.
"""

import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline execution states."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetadata:
    """Metadata for pipeline execution."""
    pipeline_id: str
    name: str
    version: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    state: PipelineState = PipelineState.CREATED
    config: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_id: str
    success: bool
    output: Dict[str, Any]
    metrics: Dict[str, Any]
    errors: Optional[List[str]] = None


class PipelineManager:
    """Centralized pipeline orchestration and management."""
    
    def __init__(self):
        self._pipelines: Dict[str, PipelineMetadata] = {}
        self._results: Dict[str, PipelineResult] = {}
        self._lock = asyncio.Lock()
        self._running_pipelines: Dict[str, asyncio.Task] = {}
        
    async def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        version: str,
        config: Dict[str, Any],
        tags: Dict[str, Any] = None
    ) -> PipelineMetadata:
        """Create a new pipeline instance."""
        async with self._lock:
            if pipeline_id in self._pipelines:
                raise ValueError(f"Pipeline {pipeline_id} already exists")
            
            metadata = PipelineMetadata(
                pipeline_id=pipeline_id,
                name=name,
                version=version,
                created_at=time.time(),
                config=config,
                tags=tags or {},
                state=PipelineState.CREATED
            )
            
            self._pipelines[pipeline_id] = metadata
            return metadata
    
    async def start_pipeline(
        self,
        pipeline_id: str,
        pipeline_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> PipelineMetadata:
        """Start pipeline execution."""
        async with self._lock:
            if pipeline_id not in self._pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            metadata = self._pipelines[pipeline_id]
            if metadata.state != PipelineState.CREATED:
                raise ValueError(f"Pipeline {pipeline_id} is already running")
            
            # Update state and start time
            metadata.started_at = time.time()
            metadata.state = PipelineState.RUNNING
            
            # Create and start task
            task = asyncio.create_task(
                self._execute_pipeline(pipeline_id, pipeline_func)
            )
            self._running_pipelines[pipeline_id] = task
            
            return metadata
    
    async def _execute_pipeline(
        self,
        pipeline_id: str,
        pipeline_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Internal pipeline execution."""
        try:
            metadata = self._pipelines[pipeline_id]
            config = metadata.config
            
            # Execute pipeline
            output = pipeline_func(config)
            
            # Create result
            result = PipelineResult(
                pipeline_id=pipeline_id,
                success=True,
                output=output,
                metrics={}
            )
            
            # Update metadata
            metadata.completed_at = time.time()
            metadata.state = PipelineState.COMPLETED
            
            # Store result
            self._results[pipeline_id] = result
            
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            
        except Exception as e:
            # Handle failure
            metadata.completed_at = time.time()
            metadata.state = PipelineState.FAILED
            
            result = PipelineResult(
                pipeline_id=pipeline_id,
                success=False,
                output={},
                metrics={},
                errors=[str(e)]
            )
            
            self._results[pipeline_id] = result
            
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
        
        finally:
            # Clean up
            async with self._lock:
                self._running_pipelines.pop(pipeline_id, None)
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel pipeline execution."""
        async with self._lock:
            if pipeline_id not in self._pipelines:
                return False
            
            metadata = self._pipelines[pipeline_id]
            if metadata.state not in [PipelineState.RUNNING, PipelineState.CREATED]:
                return False
            
            # Cancel running task
            task = self._running_pipelines.get(pipeline_id)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Update state
            metadata.state = PipelineState.CANCELLED
            metadata.completed_at = time.time()
            
            return True
    
    async def get_pipeline_status(self, pipeline_id: str) -> PipelineMetadata:
        """Get pipeline status."""
        async with self._lock:
            if pipeline_id not in self._pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            return self._pipelines[pipeline_id]
    
    async def get_pipeline_result(self, pipeline_id: str) -> PipelineResult:
        """Get pipeline result."""
        async with self._lock:
            if pipeline_id not in self._results:
                raise ValueError(f"Pipeline {pipeline_id} result not found")
            
            return self._results[pipeline_id]
    
    async def list_pipelines(self, state: PipelineState = None) -> List[PipelineMetadata]:
        """List all pipelines, optionally filtered by state."""
        async with self._lock:
            if state:
                return [
                    meta for meta in self._pipelines.values()
                    if meta.state == state
                ]
            return list(self._pipelines.values())
    
    async def cleanup_pipelines(self, older_than_seconds: int = 86400):
        """Clean up old completed pipelines."""
        async with self._lock:
            now = time.time()
            to_remove = []
            
            for pipeline_id, metadata in self._pipelines.items():
                if metadata.state in [PipelineState.COMPLETED, PipelineState.FAILED, PipelineState.CANCELLED]:
                    if metadata.completed_at and (now - metadata.completed_at) > older_than_seconds:
                        to_remove.append(pipeline_id)
            
            for pipeline_id in to_remove:
                self._pipelines.pop(pipeline_id, None)
                self._results.pop(pipeline_id, None)
                logger.info(f"Cleaned up pipeline {pipeline_id}")


# Global pipeline manager instance
pipeline_manager = PipelineManager()