import time
import logging
from collections import OrderedDict
import threading
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TTLCache:
    """
    Thread-safe TTL cache with LRU eviction.
    
    Features:
    - Time-to-live expiration
    - Least-recently-used eviction when full
    - Automatic cleanup on access
    - Thread-safe operations
    - Cache statistics tracking
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize TTL cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()  # key -> (value, timestamp)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        # Iterate through cache to find expired entries
        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)
            else:
                # Since OrderedDict is ordered by insertion, we can break early
                break
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
            self.cache_evictions += 1
            logger.debug(f"Cache entry expired: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, returns None if expired or not found."""
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache:
                value, timestamp = self.cache[key]
                # Move to end to mark as recently used
                self.cache.move_to_end(key)
                self.cache_hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                self.cache_misses += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
    
    def set(self, key: str, value: Any):
        """Add item to cache with current timestamp."""
        with self.lock:
            self._cleanup_expired()
            
            # If key already exists, remove it first
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = (value, time.time())
            
            # Evict oldest if over max size
            if len(self.cache) > self.max_size:
                evicted_key, _ = self.cache.popitem(last=False)
                self.cache_evictions += 1
                logger.debug(f"Cache evicted oldest entry: {evicted_key}")
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_evictions = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_evictions': self.cache_evictions,
                'hit_rate': hit_rate,
                'current_size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl
            }
    
    def get_current_keys(self) -> List[str]:
        """Get list of current cache keys (for debugging)."""
        with self.lock:
            return list(self.cache.keys())

def test_ttl_cache():
    """Test the TTL cache implementation."""
    logger.info("Testing TTLCache implementation...")
    
    # Create cache with 2 second TTL
    cache = TTLCache(max_size=5, ttl=2)
    
    # Test basic set/get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1", "Basic get failed"
    
    # Test cache hit
    cache.set("key2", "value2")
    cache.get("key1")  # Should be a hit
    cache.get("key2")  # Should be a hit
    
    # Test cache miss
    assert cache.get("nonexistent") is None, "Cache miss failed"
    
    # Test expiration
    time.sleep(3)  # Wait for TTL to expire
    assert cache.get("key1") is None, "Expiration failed for key1"
    assert cache.get("key2") is None, "Expiration failed for key2"
    
    # Test LRU eviction
    cache.set("key3", "value3")
    cache.set("key4", "value4")
    cache.set("key5", "value5")
    cache.set("key6", "value6")  # Size: 4
    cache.set("key7", "value7")  # Size: 5
    cache.set("key8", "value8")  # Size: 6, should evict key3 (oldest)
    
    assert cache.get("key3") is None, "LRU eviction failed"
    assert cache.get("key4") == "value4", "LRU eviction affected wrong key"
    
    # Test statistics
    stats = cache.get_stats()
    assert stats['cache_hits'] > 0, "Hits not tracked"
    assert stats['cache_misses'] > 0, "Misses not tracked"
    assert stats['cache_evictions'] > 0, "Evictions not tracked"
    
    # Test thread safety (basic)
    import threading
    
    def writer_thread():
        for i in range(10):
            cache.set(f"thread_key_{i}", f"thread_value_{i}")
            time.sleep(0.1)
    
    def reader_thread():
        for i in range(10):
            cache.get(f"thread_key_{i}")
            time.sleep(0.1)
    
    threads = []
    for _ in range(3):
        t = threading.Thread(target=writer_thread)
        threads.append(t)
        t.start()
    
    for _ in range(2):
        t = threading.Thread(target=reader_thread)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    logger.info("All TTLCache tests passed!")
    logger.info(f"Final cache stats: {cache.get_stats()}")

if __name__ == "__main__":
    test_ttl_cache()