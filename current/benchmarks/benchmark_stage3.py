#!/usr/bin/env python3
"""
Benchmark script for Stage 3 production optimizations.
Measures performance improvements from caching, compression, rate limiting, etc.
"""
import time
import json
import base64
import requests
import statistics
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample test image (1x1 red pixel)
SAMPLE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

class Stage3Benchmark:
    """Benchmark Stage 3 optimizations."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: Dict[str, List[float]] = {}
    
    def run_benchmark(self, num_requests: int = 20, use_batch: bool = False) -> Dict[str, Any]:
        """Run benchmark tests."""
        logger.info(f"Starting Stage 3 benchmark ({num_requests} requests)...")
        
        # Test single diagnosis endpoint
        logger.info("Testing /v1/diagnose endpoint...")
        times = self._test_endpoint("/v1/diagnose", num_requests, use_batch)
        self.results["diagnose"] = times
        
        # Test batch endpoint if enabled
        if use_batch:
            logger.info("Testing /v1/diagnose/batch endpoint...")
            batch_times = self._test_batch_endpoint(num_requests // 2)
            self.results["diagnose_batch"] = batch_times
        
        # Test caching (second request should be faster)
        logger.info("Testing cache effectiveness...")
        cache_times = self._test_caching()
        self.results["caching"] = cache_times
        
        # Generate report
        return self._generate_report()
    
    def _test_endpoint(self, endpoint: str, num_requests: int, use_batch: bool = False) -> List[float]:
        """Test single endpoint."""
        times = []
        url = f"{self.base_url}{endpoint}"
        
        for i in range(num_requests):
            payload = {
                "image": SAMPLE_IMAGE_B64,
                "crop_hint": "tomato"
            }
            
            start = time.time()
            try:
                response = requests.post(url, json=payload, timeout=30)
                elapsed = time.time() - start
                times.append(elapsed)
                
                if response.status_code != 200:
                    logger.warning(f"Request {i+1} failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Request {i+1} error: {e}")
                times.append(0)
            
            time.sleep(0.1)  # Small delay to avoid overwhelming
        
        return times
    
    def _test_batch_endpoint(self, num_requests: int) -> List[float]:
        """Test batch endpoint."""
        times = []
        url = f"{self.base_url}/v1/diagnose/batch"
        
        for i in range(num_requests):
            payload = {
                "images": [SAMPLE_IMAGE_B64] * 3,  # 3 images per batch
                "crop_hint": "tomato"
            }
            
            start = time.time()
            try:
                response = requests.post(url, json=payload, timeout=30)
                elapsed = time.time() - start
                times.append(elapsed)
                
                if response.status_code != 200:
                    logger.warning(f"Batch request {i+1} failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Batch request {i+1} error: {e}")
                times.append(0)
            
            time.sleep(0.2)
        
        return times
    
    def _test_caching(self) -> List[float]:
        """Test cache effectiveness by making two identical requests."""
        url = f"{self.base_url}/v1/diagnose"
        payload = {"image": SAMPLE_IMAGE_B64, "crop_hint": "tomato"}
        
        # First request (cache miss)
        start = time.time()
        try:
            response = requests.post(url, json=payload, timeout=30)
            first_time = time.time() - start
        except Exception as e:
            logger.error(f"Cache test first request error: {e}")
            first_time = 0
        
        # Second request (should be cache hit)
        time.sleep(0.5)
        start = time.time()
        try:
            response = requests.post(url, json=payload, timeout=30)
            second_time = time.time() - start
        except Exception as e:
            logger.error(f"Cache test second request error: {e}")
            second_time = 0
        
        return [first_time, second_time]
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate benchmark report."""
        report = {
            "version": "v5.5.3-performance",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {},
            "details": {}
        }
        
        for endpoint, times in self.results.items():
            if not times or all(t == 0 for t in times):
                continue
            
            valid_times = [t for t in times if t > 0]
            
            if valid_times:
                report["details"][endpoint] = {
                    "samples": len(valid_times),
                    "avg_seconds": statistics.mean(valid_times),
                    "min_seconds": min(valid_times),
                    "max_seconds": max(valid_times),
                    "median_seconds": statistics.median(valid_times),
                    "stdev_seconds": statistics.stdev(valid_times) if len(valid_times) > 1 else 0
                }
        
        # Calculate improvements
        if "diagnose" in report["details"] and "caching" in report["details"]:
            cache_times = self.results["caching"]
            if len(cache_times) >= 2 and cache_times[0] > 0 and cache_times[1] > 0:
                cache_improvement = ((cache_times[0] - cache_times[1]) / cache_times[0]) * 100
                report["summary"]["cache_improvement_percent"] = round(cache_improvement, 1)
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted report."""
        print("\n" + "="*60)
        print("STAGE 3 PRODUCTION OPTIMIZATIONS BENCHMARK REPORT")
        print("="*60)
        print(f"Version: {report['version']}")
        print(f"Timestamp: {report['timestamp']}")
        print("\n" + "-"*60)
        print("PERFORMANCE METRICS")
        print("-"*60)
        
        for endpoint, metrics in report["details"].items():
            print(f"\n{endpoint.upper()}:")
            print(f"  Samples: {metrics['samples']}")
            print(f"  Average: {metrics['avg_seconds']:.4f}s")
            print(f"  Median:  {metrics['median_seconds']:.4f}s")
            print(f"  Min:     {metrics['min_seconds']:.4f}s")
            print(f"  Max:     {metrics['max_seconds']:.4f}s")
        
        if "cache_improvement_percent" in report["summary"]:
            print(f"\nCache Improvement: {report['summary']['cache_improvement_percent']}%")
        
        print("\n" + "="*60)


def main():
    """Run benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Stage 3 optimizations")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests per test")
    parser.add_argument("--batch", action="store_true", help="Include batch endpoint tests")
    parser.add_argument("--output", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    # Check if API is reachable
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"API not healthy at {args.url}")
            return
    except Exception as e:
        logger.error(f"Cannot connect to API at {args.url}: {e}")
        logger.info("Make sure the API is running: python api/main.py")
        return
    
    # Run benchmark
    benchmark = Stage3Benchmark(args.url)
    report = benchmark.run_benchmark(args.requests, args.batch)
    benchmark.print_report(report)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()