import time
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.independent_multi_crop_pipeline import TTLCache

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    cache.set("key6", "value6")  # This should evict key3 (oldest)
    
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