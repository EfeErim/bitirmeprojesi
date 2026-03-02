#!/usr/bin/env python3
"""
Local Caching Strategy for Google Colab
LRU cache implementation for Google Drive with local SSD caching.
"""

import os
import time
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

from src.dataset.error_handling import (
    CacheError,
    get_error_handler,
    get_retry_handler,
    get_resource_monitor
)

logger = logging.getLogger(__name__)
error_handler = get_error_handler()
retry_handler = get_retry_handler()
resource_monitor = get_resource_monitor()


@dataclass
class CacheEntry:
    """Represents a single cache entry."""
    file_path: str
    original_path: str
    size_bytes: int
    last_accessed: float
    access_count: int
    checksum: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)


class ByteSizeLRUCache:
    """LRU cache implementation constrained by byte budget and entry count."""
    
    def __init__(self, max_size_bytes: int, max_entries: int = 1000):
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get an entry from the cache."""
        with self._lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.cache[key] = entry
                return entry
            return None
    
    def put(self, key: str, entry: CacheEntry):
        """Add an entry to the cache."""
        with self._lock:
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Check if we need to evict entries
            while (self.current_size + entry.size_bytes > self.max_size_bytes or 
                   len(self.cache) >= self.max_entries) and len(self.cache) > 0:
                self._evict_oldest()
            
            # Add new entry
            self.cache[key] = entry
            self.current_size += entry.size_bytes
    
    def _remove_entry(self, key: str):
        """Remove a specific entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size -= entry.size_bytes
            del self.cache[key]
    
    def _evict_oldest(self) -> str:
        """Evict the least recently used entry."""
        if not self.cache:
            return ""
        
        key = next(iter(self.cache))
        self._remove_entry(key)
        return key
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.current_size = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self.cache),
                "total_size_bytes": self.current_size,
                "max_size_bytes": self.max_size_bytes,
                "max_entries": self.max_entries,
                "hit_rate": self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.cache:
            return 0.0
        total_access = sum(e.access_count for e in self.cache.values())
        if total_access == 0:
            return 0.0
        # This is a simplified calculation; in production you'd track misses separately
        return min(1.0, total_access / (total_access + len(self.cache)))


class ColabCacheManager:
    """Manages local SSD caching for Google Drive files."""
    
    def __init__(
        self,
        cache_dir: str = "./.cache",
        max_cache_size_gb: float = 10.0,
        max_entries: int = 1000,
        cleanup_interval_hours: int = 24,
        ttl_days: int = 7
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert GB to bytes
        max_cache_size_bytes = int(max_cache_size_gb * 1024**3)
        
        self.lru_cache = ByteSizeLRUCache(max_cache_size_bytes, max_entries)
        self.cleanup_interval = cleanup_interval_hours * 3600
        self.ttl_seconds = ttl_days * 86400
        self._stop_cleanup = threading.Event()
        
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.last_cleanup = self._load_metadata()
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _load_metadata(self) -> float:
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Restore cache entries
                for key, entry_data in data.get("entries", {}).items():
                    entry = CacheEntry.from_dict(entry_data)
                    self.lru_cache.put(key, entry)
                
                last_cleanup = data.get("last_cleanup", time.time())
                logger.info(f"Loaded cache metadata: {len(data.get('entries', {}))} entries")
                return last_cleanup
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
        
        return time.time()
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            data = {
                "entries": {
                    key: entry.to_dict() 
                    for key, entry in self.lru_cache.cache.items()
                },
                "last_cleanup": time.time()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def get_cached_file(self, file_key: str) -> Optional[Path]:
        """Get a file from the local cache."""
        try:
            with self.lru_cache._lock:
                entry = self.lru_cache.get(file_key)
                if entry is None:
                    # Compatibility path: resolve user-facing file_key to internal cache key.
                    for cache_key, cache_entry in self.lru_cache.cache.items():
                        if cache_entry.metadata.get("file_key") == file_key:
                            entry = self.lru_cache.get(cache_key)
                            break
                if entry and entry.file_path and Path(entry.file_path).exists():
                    logger.debug(f"Cache hit for: {file_key}")
                    return Path(entry.file_path)
            
            return None
        except Exception as e:
            error_handler.handle_exception(
                CacheError(
                    message=f"Failed to get cached file: {str(e)}",
                    cache_key=file_key,
                    operation="get_cached_file",
                    cause=e
                )
            )
            return None
    
    def cache_file(
        self,
        source_path: Path,
        file_key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Cache a file locally."""
        try:
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Check system resources before caching
            if not resource_monitor.check_disk_space(required_gb=0.1):  # 100MB
                raise CacheError(
                    message="Insufficient disk space for caching",
                    cache_key=file_key,
                    cache_dir=str(self.cache_dir),
                    operation="cache_file",
                    cause=Exception("Low disk space")
                )
            
            # Generate cache key
            cache_key = self._generate_cache_key(source_path, file_key)
            
            # Check if already cached
            existing = self.get_cached_file(file_key)
            if existing:
                return existing
            
            # Copy file to cache with retry
            cache_path = self.cache_dir / f"{cache_key}_{source_path.name}"
            
            def copy_file():
                shutil.copy2(source_path, cache_path)
            
            retry_handler.execute_with_retry(copy_file)
            
            # Calculate checksum
            checksum = self._calculate_checksum(cache_path)
            
            # Create cache entry
            merged_metadata = dict(metadata or {})
            merged_metadata.setdefault("file_key", file_key)
            entry = CacheEntry(
                file_path=str(cache_path),
                original_path=str(source_path),
                size_bytes=cache_path.stat().st_size,
                last_accessed=time.time(),
                access_count=1,
                checksum=checksum,
                metadata=merged_metadata
            )
            
            # Add to LRU cache
            self.lru_cache.put(cache_key, entry)
            
            logger.info(f"Cached file: {source_path.name} -> {cache_path.name}")
            return cache_path
            
        except Exception as e:
            error_handler.handle_exception(
                CacheError(
                    message=f"Failed to cache file: {str(e)}",
                    cache_key=file_key,
                    cache_dir=str(self.cache_dir),
                    operation="cache_file",
                    cause=e
                )
            )
            raise
    
    def invalidate(self, file_key: str) -> bool:
        """Invalidate a cached file."""
        with self.lru_cache._lock:
            cache_key = file_key
            entry = self.lru_cache.cache.get(cache_key)
            if entry is None:
                for key, cached_entry in self.lru_cache.cache.items():
                    if cached_entry.metadata.get("file_key") == file_key:
                        cache_key = key
                        entry = cached_entry
                        break
            if entry:
                try:
                    cache_path = Path(entry.file_path)
                    if cache_path.exists():
                        try:
                            cache_path.unlink()
                        except Exception as unlink_error:
                            logger.warning(
                                f"Cache file could not be deleted during invalidation ({cache_path}): {unlink_error}"
                            )
                    self.lru_cache._remove_entry(cache_key)
                    self._save_metadata()
                    logger.info(f"Invalidated cache entry: {file_key}")
                    return True
                except Exception as e:
                    logger.error(f"Error invalidating cache entry: {str(e)}")
        
        return False
    
    def clear_cache(self, keep_recent: bool = True) -> int:
        """Clear the cache. If keep_recent is True, keep entries accessed in last 24h."""
        cleared_count = 0
        cutoff_time = time.time() - 86400 if keep_recent else 0
        
        with self.lru_cache._lock:
            keys_to_remove = []
            
            for key, entry in self.lru_cache.cache.items():
                if entry.last_accessed < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    entry = self.lru_cache.cache[key]
                    cache_path = Path(entry.file_path)
                    if cache_path.exists():
                        cache_path.unlink()
                    self.lru_cache._remove_entry(key)
                    cleared_count += 1
                except Exception as e:
                    logger.error(f"Error clearing cache entry {key}: {str(e)}")
        
        self._save_metadata()
        logger.info(f"Cleared {cleared_count} cache entries")
        return cleared_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.lru_cache.stats()
        stats.update({
            "cache_dir": str(self.cache_dir),
            "last_cleanup": self.last_cleanup,
            "cleanup_interval_hours": self.cleanup_interval / 3600,
            "ttl_days": self.ttl_seconds / 86400
        })
        
        # Add disk usage
        try:
            total_size = sum(
                f.stat().st_size for f in self.cache_dir.glob("*") 
                if f.is_file() and f.name != "cache_metadata.json"
            )
            stats["disk_usage_bytes"] = total_size
            stats["disk_usage_gb"] = total_size / (1024**3)
        except Exception as e:
            logger.error(f"Error calculating disk usage: {str(e)}")
        
        return stats
    
    def _background_cleanup(self):
        """Background thread for periodic cleanup."""
        while not self._stop_cleanup.is_set():
            # Wait with timeout so shutdown can interrupt sleep.
            if self._stop_cleanup.wait(timeout=3600):
                break
            
            try:
                current_time = time.time()
                if current_time - self.last_cleanup > self.cleanup_interval:
                    self._perform_cleanup()
            except Exception as e:
                logger.error(f"Error in background cleanup: {str(e)}")

    def shutdown(self):
        """Stop background cleanup thread and persist metadata."""
        self._stop_cleanup.set()
        if getattr(self, "_cleanup_thread", None) and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        try:
            self._save_metadata()
        except Exception:
            pass

    def __del__(self):
        """Best-effort cleanup for interpreter shutdown paths."""
        try:
            self.shutdown()
        except Exception:
            pass
    
    def _perform_cleanup(self):
        """Perform cache cleanup based on TTL and size constraints."""
        logger.info("Starting cache cleanup...")
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = []
        with self.lru_cache._lock:
            for key, entry in self.lru_cache.cache.items():
                if current_time - entry.last_accessed > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                try:
                    entry = self.lru_cache.cache[key]
                    cache_path = Path(entry.file_path)
                    if cache_path.exists():
                        try:
                            cache_path.unlink()
                        except Exception as unlink_error:
                            logger.warning(
                                f"Cache file could not be deleted during cleanup ({cache_path}): {unlink_error}"
                            )
                    self.lru_cache._remove_entry(key)
                except Exception as e:
                    logger.error(f"Error removing expired entry {key}: {str(e)}")
        
        # Save updated metadata
        self._save_metadata()
        self.last_cleanup = time.time()
        
        logger.info(f"Cache cleanup completed. Removed {len(expired_keys)} expired entries.")
    
    def _generate_cache_key(self, source_path: Path, file_key: str) -> str:
        """Generate a unique cache key."""
        combined = f"{source_path}:{file_key}"
        return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()[:16]
    
    def _calculate_checksum(self, file_path: Path, chunk_size: int = 1024*1024) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def prefetch_files(self, file_keys: List[str]) -> List[Path]:
        """Prefetch multiple files into cache."""
        results = []
        for file_key in file_keys:
            # This would typically be called with actual file paths
            # For now, return None as we need the source path
            results.append(None)
        return results
    
    def warm_cache(self, file_list: List[Tuple[Path, str, Dict[str, Any]]]) -> int:
        """Warm the cache with frequently used files."""
        cached_count = 0
        for source_path, file_key, metadata in file_list:
            try:
                self.cache_file(source_path, file_key, metadata)
                cached_count += 1
            except Exception as e:
                logger.error(f"Failed to cache {source_path}: {str(e)}")
        
        logger.info(f"Cache warming completed: {cached_count}/{len(file_list)} files cached")
        return cached_count


class PerformanceMonitor:
    """Monitor cache and I/O performance."""
    
    def __init__(self, cache_manager: ColabCacheManager):
        self.cache_manager = cache_manager
        self.metrics: Dict[str, List[float]] = {
            "cache_hits": [],
            "cache_misses": [],
            "io_times": [],
            "cache_sizes": []
        }
        self.start_time = time.time()
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics["cache_hits"].append(time.time())
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics["cache_misses"].append(time.time())
    
    def record_io_time(self, duration: float):
        """Record I/O operation time."""
        self.metrics["io_times"].append(duration)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        now = time.time()
        runtime = now - self.start_time
        
        cache_hits = len(self.metrics["cache_hits"])
        cache_misses = len(self.metrics["cache_misses"])
        total_requests = cache_hits + cache_misses
        
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        io_times = self.metrics["io_times"]
        avg_io_time = sum(io_times) / len(io_times) if io_times else 0.0
        
        cache_stats = self.cache_manager.get_stats()
        
        return {
            "runtime_seconds": runtime,
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": hit_rate,
            "avg_io_time_seconds": avg_io_time,
            "cache_stats": cache_stats,
            "throughput_ops_per_sec": total_requests / runtime if runtime > 0 else 0.0
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        for key in self.metrics:
            self.metrics[key].clear()
        self.start_time = time.time()


# Backwards-compatible alias retained for existing imports/tests.
LRUCache = ByteSizeLRUCache


def get_colab_cache_manager(
    cache_dir: str = "./.cache",
    max_cache_size_gb: float = 10.0
) -> ColabCacheManager:
    """Get a cache manager configured for Google Colab."""
    # Colab typically has ~68GB SSD, so use 10GB for cache by default
    # Adjust based on available space if needed
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return ColabCacheManager(
        cache_dir=cache_dir,
        max_cache_size_gb=max_cache_size_gb,
        max_entries=1000,
        cleanup_interval_hours=24,
        ttl_days=7
    )


if __name__ == "__main__":
    # Example usage
    cache_mgr = get_colab_cache_manager()
    
    # Test caching a file
    test_file = Path("./test.txt")
    if test_file.exists():
        cached = cache_mgr.cache_file(
            source_path=test_file,
            file_key="test_file",
            metadata={"description": "Test file"}
        )
        print(f"Cached at: {cached}")
    
    # Print stats
    stats = cache_mgr.get_stats()
    print(json.dumps(stats, indent=2))
