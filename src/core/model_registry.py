"""
Model version tracking and caching system.
Provides centralized management of model versions, caching, and loading.
"""

import time
import os
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    model_id: str
    version: str
    name: str
    description: str
    created_at: float
    file_path: str
    size_bytes: int
    checksum: str
    tags: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ModelCacheEntry:
    """Cache entry for a loaded model."""
    model: Any
    metadata: ModelMetadata
    last_accessed: float
    access_count: int = 0


class ModelRegistry:
    """Centralized model version tracking and caching."""
    
    def __init__(
        self,
        model_dir: str = "models",
        cache_size: int = 10,
        cache_ttl: int = 3600
    ):
        self.model_dir = Path(model_dir)
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self._models: Dict[str, ModelMetadata] = {}
        self._cache: Dict[str, ModelCacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cache_cleanup_task = None
        
        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing models
        self._load_models()
        
        # Start cache cleanup task
        self._start_cache_cleanup()
    
    def _load_models(self):
        """Load model metadata from disk."""
        metadata_file = self.model_dir / "models.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    models_data = json.load(f)
                    
                for model_data in models_data:
                    metadata = ModelMetadata(**model_data)
                    self._models[metadata.model_id] = metadata
                
                logger.info(f"Loaded {len(models_data)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
    
    def _save_models(self):
        """Save model metadata to disk."""
        metadata_file = self.model_dir / "models.json"
        
        try:
            models_data = [
                {k: v for k, v in meta.__dict__.items() if k != 'model'}
                for meta in self._models.values()
            ]
            
            with open(metadata_file, 'w') as f:
                json.dump(models_data, f, indent=2, default=str)
                
            logger.info(f"Saved {len(models_data)} models to registry")
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    async def register_model(
        self,
        model_id: str,
        version: str,
        name: str,
        description: str,
        file_path: str,
        tags: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> ModelMetadata:
        """Register a new model version."""
        async with self._lock:
            if model_id in self._models:
                raise ValueError(f"Model {model_id} already registered")
            
            # Calculate checksum
            checksum = self._calculate_checksum(file_path)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                version=version,
                name=name,
                description=description,
                created_at=time.time(),
                file_path=file_path,
                size_bytes=file_size,
                checksum=checksum,
                tags=tags or {},
                dependencies=dependencies or []
            )
            
            # Add to registry
            self._models[model_id] = metadata
            self._save_models()
            
            logger.info(f"Registered model {model_id} version {version}")
            return metadata
    
    async def get_model(
        self,
        model_id: str,
        version: str = None
    ) -> Tuple[Any, ModelMetadata]:
        """Get a model instance, loading from cache or disk."""
        async with self._lock:
            # Find the right version
            metadata = self._find_model(model_id, version)
            if not metadata:
                raise ValueError(f"Model {model_id} version {version} not found")
            
            # Check cache first
            cache_key = f"{metadata.model_id}:{metadata.version}"
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry.model, metadata
            
            # Load from disk
            model = self._load_model_from_disk(metadata)
            
            # Add to cache
            self._add_to_cache(cache_key, model, metadata)
            
            return model, metadata
    
    def _find_model(self, model_id: str, version: str = None) -> Optional[ModelMetadata]:
        """Find the best matching model."""
        if model_id not in self._models:
            return None
        
        if not version:
            # Return latest version
            return max(
                (meta for meta in self._models.values() if meta.model_id == model_id),
                key=lambda m: m.created_at
            )
        
        # Find exact version
        for meta in self._models.values():
            if meta.model_id == model_id and meta.version == version:
                return meta
        
        return None
    
    def _load_model_from_disk(self, metadata: ModelMetadata) -> Any:
        """Load model from disk."""
        try:
            # For this example, assume models are pickled
            with open(metadata.file_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded model {metadata.model_id} from disk")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {metadata.model_id}: {e}")
            raise
    
    def _add_to_cache(self, key: str, model: Any, metadata: ModelMetadata):
        """Add model to cache."""
        # Evict if cache is full
        if len(self._cache) >= self.cache_size:
            self._evict_cache()
        
        # Add to cache
        entry = ModelCacheEntry(
            model=model,
            metadata=metadata,
            last_accessed=time.time()
        )
        self._cache[key] = entry
    
    def _evict_cache(self):
        """Evict least recently used cache entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_entry = min(self._cache.values(), key=lambda e: e.last_accessed)
        
        # Remove from cache
        for key, entry in self._cache.items():
            if entry == lru_entry:
                self._cache.pop(key)
                logger.info(f"Evicted model {entry.metadata.model_id} from cache")
                break
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        hash_algo = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_algo.update(chunk)
        
        return hash_algo.hexdigest()
    
    async def list_models(self, model_id: str = None) -> List[ModelMetadata]:
        """List all models or models for specific ID."""
        async with self._lock:
            if model_id:
                return [
                    meta for meta in self._models.values()
                    if meta.model_id == model_id
                ]
            return list(self._models.values())
    
    async def get_model_versions(self, model_id: str) -> List[str]:
        """Get all versions for a model."""
        async with self._lock:
            return [
                meta.version for meta in self._models.values()
                if meta.model_id == model_id
            ]
    
    async def get_latest_version(self, model_id: str) -> Optional[ModelMetadata]:
        """Get the latest version of a model."""
        async with self._lock:
            versions = [
                meta for meta in self._models.values()
                if meta.model_id == model_id
            ]
            
            if not versions:
                return None
            
            return max(versions, key=lambda m: m.created_at)
    
    async def remove_model(self, model_id: str, version: str = None) -> bool:
        """Remove a model version."""
        async with self._lock:
            metadata = self._find_model(model_id, version)
            if not metadata:
                return False
            
            # Remove from registry
            del self._models[metadata.model_id]
            self._save_models()
            
            # Remove from cache
            cache_key = f"{metadata.model_id}:{metadata.version}"
            self._cache.pop(cache_key, None)
            
            logger.info(f"Removed model {model_id} version {version}")
            return True
    
    def _start_cache_cleanup(self):
        """Start periodic cache cleanup."""
        async def cleanup():
            while True:
                await asyncio.sleep(self.cache_ttl)
                self._cleanup_cache()
        
        self._cache_cleanup_task = asyncio.create_task(cleanup())
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        now = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if (now - entry.last_accessed) > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._cache.pop(key)
            logger.info(f"Cleaned up expired cache entry: {key}")


# Global model registry instance
model_registry = ModelRegistry()