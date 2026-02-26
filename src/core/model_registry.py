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
import torch

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
        cache_ttl: int = 3600,
        allow_unsafe_pickle: Optional[bool] = None,
    ):
        self.model_dir = Path(model_dir)
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        if allow_unsafe_pickle is None:
            allow_unsafe_pickle = (
                str(os.getenv("AADS_ALLOW_UNSAFE_PICKLE", "0")).strip().lower()
                in {"1", "true", "yes", "on"}
            )
        self.allow_unsafe_pickle = bool(allow_unsafe_pickle)
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

    @staticmethod
    def _registry_key(model_id: str, version: str) -> str:
        """Return canonical key for a specific model version."""
        return f"{model_id}:{version}"
    
    def _load_models(self):
        """Load model metadata from disk."""
        metadata_file = self.model_dir / "models.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    models_data = json.load(f)
                    
                for model_data in models_data:
                    metadata = ModelMetadata(**model_data)
                    self._models[self._registry_key(metadata.model_id, metadata.version)] = metadata
                
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
            registry_key = self._registry_key(model_id, version)
            if registry_key in self._models:
                raise ValueError(f"Model {model_id} version {version} already registered")
            
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
            self._models[registry_key] = metadata
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
        if not version:
            versions = [
                meta for meta in self._models.values()
                if meta.model_id == model_id
            ]
            if not versions:
                return None
            return max(versions, key=lambda m: m.created_at)
        
        # Find exact version
        return self._models.get(self._registry_key(model_id, version))
    
    def _load_model_from_disk(self, metadata: ModelMetadata) -> Any:
        """Load model from disk."""
        try:
            file_path = Path(metadata.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Model file does not exist: {file_path}")

            current_checksum = self._calculate_checksum(str(file_path))
            if current_checksum != metadata.checksum:
                raise RuntimeError(
                    f"Checksum mismatch for model {metadata.model_id}:{metadata.version}. "
                    f"Expected {metadata.checksum}, got {current_checksum}."
                )

            suffix = file_path.suffix.lower()
            if suffix in {".pt", ".pth", ".bin"}:
                model = torch.load(str(file_path), map_location="cpu")
            elif suffix in {".pkl", ".pickle"}:
                if not self.allow_unsafe_pickle:
                    raise RuntimeError(
                        "Refusing to load pickle model without explicit opt-in. "
                        "Set AADS_ALLOW_UNSAFE_PICKLE=1 if you trust the artifact source."
                    )
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                # Try torch first for common serialized artifacts; only fallback to
                # pickle when explicitly allowed.
                try:
                    model = torch.load(str(file_path), map_location="cpu")
                except Exception:
                    if not self.allow_unsafe_pickle:
                        raise RuntimeError(
                            f"Unsupported model format '{file_path.suffix}' for safe loading. "
                            "Use a torch-serialized artifact or opt into unsafe pickle loading."
                        )
                    with open(file_path, 'rb') as f:
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
        lru_key, lru_entry = min(
            self._cache.items(),
            key=lambda item: item[1].last_accessed
        )
        self._cache.pop(lru_key, None)
        logger.info(f"Evicted model {lru_entry.metadata.model_id} from cache")
    
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
            self._models.pop(self._registry_key(metadata.model_id, metadata.version), None)
            self._save_models()
            
            # Remove from cache
            cache_key = f"{metadata.model_id}:{metadata.version}"
            self._cache.pop(cache_key, None)
            
            logger.info(f"Removed model {model_id} version {version}")
            return True
    
    def _start_cache_cleanup(self):
        """Start periodic cache cleanup."""
        if self._cache_cleanup_task and not self._cache_cleanup_task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Imported in synchronous contexts (CLI/tests) where no loop is running.
            # Defer background cleanup until an async caller explicitly enables it.
            logger.debug("No running event loop; cache cleanup task not started.")
            return

        async def cleanup():
            try:
                while True:
                    await asyncio.sleep(self.cache_ttl)
                    self._cleanup_cache()
            except asyncio.CancelledError:
                return
        
        self._cache_cleanup_task = loop.create_task(cleanup())

    def ensure_cache_cleanup_task(self):
        """Public hook for async contexts to start cleanup if needed."""
        self._start_cache_cleanup()

    async def shutdown(self):
        """Stop background tasks owned by this registry."""
        if self._cache_cleanup_task and not self._cache_cleanup_task.done():
            self._cache_cleanup_task.cancel()
            try:
                await self._cache_cleanup_task
            except asyncio.CancelledError:
                pass
        self._cache_cleanup_task = None
    
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
