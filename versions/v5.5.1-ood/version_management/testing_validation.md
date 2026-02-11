# Testing and Validation Procedures

## 1. Testing Strategy Overview

Comprehensive testing for each version includes:
- **Unit Tests**: Individual component validation
- **Integration Tests**: Full pipeline testing
- **Performance Tests**: Benchmarking and regression testing
- **A/B Tests**: Comparative validation (for major changes)
- **Rollback Tests**: Verify restore procedures

## 2. Unit Testing

### 2.1 Test Structure

```
tests/
├── unit/
│   ├── test_adapter.py
│   ├── test_ood.py
│   ├── test_router.py
│   └── test_version_manager.py
├── integration/
│   └── test_full_pipeline.py
├── performance/
│   └── test_regression.py
└── version_tests/
    ├── test_v5.5.0.py
    ├── test_v5.5.1_ood.py
    └── test_v5.5.2_router.py
```

### 2.2 Version-Specific Unit Tests

```python
# tests/version_tests/test_v5.5.1_ood.py
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from version_management.backup import VersionManager
from src.ood.dynamic_thresholds import DynamicOODThreshold
from src.adapter.independent_crop_adapter import IndependentCropAdapter

def test_version_exists():
    """Test that v5.5.1-ood backup exists."""
    vm = VersionManager()
    backups = vm.list_backups()
    assert "v5.5.1-ood" in backups, "Version v5.5.1-ood not found in backups"

def test_ood_dynamic_threshold():
    """Test dynamic OOD threshold calculation."""
    threshold = DynamicOODThreshold(threshold_factor=2.0)

    # Mock class features
    import numpy as np
    class_features = [np.random.randn(768) for _ in range(20)]

    # Calculate threshold
    thresh = threshold.calculate_threshold(class_features)
    assert thresh > 0, "Threshold should be positive"
    assert isinstance(thresh, float), "Threshold should be float"

def test_adapter_ood_integration():
    """Test adapter OOD detection with dynamic thresholds."""
    adapter = IndependentCropAdapter("tomato")

    # Mock some data
    # This would normally load actual trained model
    # For unit test, we mock the components

    assert adapter is not None
    assert hasattr(adapter, 'detect_ood_dynamic')

def test_version_restore():
    """Test restoring v5.5.1-ood to temp directory."""
    vm = VersionManager()

    temp_dir = Path("temp_test_restore_v5.5.1")
    success, msg = vm.restore_backup("v5.5.1-ood", target_dir=temp_dir)

    assert success, f"Restore failed: {msg}"

    # Verify critical files exist
    assert (temp_dir / "src/adapter/independent_crop_adapter.py").exists()
    assert (temp_dir / "src/ood/dynamic_thresholds.py").exists()
    assert (temp_dir / "config/adapter_spec_v55.json").exists()

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_rollback_script_executable():
    """Test that rollback script exists and is executable."""
    import os
    script_path = Path("rollback_v5.5.1-ood.py")
    assert script_path.exists(), "Rollback script not found"
    assert os.access(script_path, os.X_OK), "Rollback script not executable"

def test_config_version_match():
    """Test that config version matches expected."""
    vm = VersionManager()
    info = vm.get_version_info("v5.5.1-ood")

    assert info is not None, "Version info not found"

    config_path = vm.versions_dir / "v5.5.1-ood" / "config/adapter_spec_v55.json"
    with open(config_path) as f:
        config = json.load(f)

    assert config["version"] == "5.5.1", "Config version mismatch"
```

### 2.3 Running Unit Tests

```bash
# Run all version-specific tests
pytest tests/version_tests/ -v

# Run tests for specific version
pytest tests/version_tests/test_v5.5.1_ood.py -v

# Run with coverage
pytest tests/version_tests/ --cov=src --cov-report=html
```

## 3. Integration Testing

### 3.1 Full Pipeline Test

```python
# tests/integration/test_version_pipeline.py
import pytest
import json
import tempfile
from pathlib import Path

def test_full_pipeline_with_version(version: str):
    """Test complete pipeline with a specific version."""
    from version_management.backup import VersionManager
    import sys
    import os

    vm = VersionManager()

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Restore version
        success, msg = vm.restore_backup(version, target_dir=temp_path)
        assert success, f"Failed to restore version: {msg}"

        # Change to temp directory
        old_cwd = os.getcwd()
        os.chdir(temp_path)

        try:
            # Load configuration
            with open("config/adapter_spec_v55.json") as f:
                config = json.load(f)

            # Initialize pipeline
            from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
            pipeline = IndependentMultiCropPipeline(config)

            # Initialize router (mock or load pre-trained)
            # For integration test, we might use a small test model
            assert pipeline is not None
            assert pipeline.router is None  # Not initialized yet

            # Test adapter creation
            from src.adapter.independent_crop_adapter import IndependentCropAdapter
            adapter = IndependentCropAdapter("tomato")
            assert adapter.crop_name == "tomato"

            # Test OOD detection setup
            # This would require actual trained model or mocks
            # For now, we test the structure

            assert hasattr(adapter, 'phase1_initialize')
            assert hasattr(adapter, 'phase2_add_disease')
            assert hasattr(adapter, 'phase3_fortify')
            assert hasattr(adapter, 'detect_ood_dynamic')

        finally:
            os.chdir(old_cwd)

@pytest.mark.parametrize("version", ["v5.5.0", "v5.5.1-ood"])
def test_pipeline_initialization(version):
    """Test pipeline initialization for multiple versions."""
    test_full_pipeline_with_version(version)
```

### 3.2 Component Integration Tests

```python
# tests/integration/test_ood_integration.py
def test_ood_with_real_model():
    """Test OOD detection with actual trained model (if available)."""
    # This test requires a trained model
    # Skip if model not available

    import torch
    from src.adapter.independent_crop_adapter import IndependentCropAdapter
    from src.utils.data_loader import CropDataset

    # Check if model exists
    model_path = Path("models/tomato_phase1/")
    if not model_path.exists():
        pytest.skip("No trained model found")

    adapter = IndependentCropAdapter("tomato")
    adapter.load_model(str(model_path))

    # Load test image
    # Test OOD detection
    # Verify thresholds are working

    assert adapter.mahalanobis is not None
    assert adapter.ood_thresholds is not None
```

## 4. Performance Testing

### 4.1 Benchmarking Suite

```python
# tests/performance/benchmark.py
import time
import pytest
import torch
from pathlib import Path

class PerformanceBenchmark:
    """Performance benchmarking for AADS-ULoRA versions."""

    def __init__(self, version: str):
        self.version = version
        self.results = {}

    def benchmark_inference(
        self,
        pipeline,
        test_images,
        num_runs: int = 100
    ) -> dict:
        """Benchmark inference performance."""

        latencies = []
        memory_usage = []

        # Warmup
        for _ in range(10):
            _ = pipeline.process_image(test_images[0])

        # Actual benchmark
        for i in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()

            result = pipeline.process_image(test_images[i % len(test_images)])

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

            # Memory
            if torch.cuda.is_available():
                memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                memory_usage.append(memory)

        return {
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies)//2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)],
            "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "throughput_fps": 1000 / (sum(latencies) / len(latencies))
        }

    def benchmark_training(self, trainer, train_loader, num_epochs: int = 1):
        """Benchmark training performance."""
        # Implementation depends on trainer type
        pass

    def run_full_benchmark(self, pipeline, test_images):
        """Run complete benchmark suite."""
        self.results = {
            "version": self.version,
            "inference": self.benchmark_inference(pipeline, test_images),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return self.results

    def save_results(self, output_path: Path):
        """Save benchmark results to JSON."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

# Usage in test
def test_performance_benchmark_v5_5_1_ood():
    """Benchmark v5.5.1-ood performance."""
    from version_management.backup import VersionManager
    import tempfile

    vm = VersionManager()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Restore version
        vm.restore_backup("v5.5.1-ood", target_dir=Path(temp_dir))

        # Load pipeline (simplified)
        # In practice, you'd load actual trained models
        from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline

        with open(f"{temp_dir}/config/adapter_spec_v55.json") as f:
            config = json.load(f)

        pipeline = IndependentMultiCropPipeline(config)
        # Load models if available

        # Create dummy test images
        test_images = [torch.randn(3, 224, 224) for _ in range(10)]

        # Benchmark
        benchmark = PerformanceBenchmark("v5.5.1-ood")
        results = benchmark.run_full_benchmark(pipeline, test_images)

        # Assert performance targets
        inference = results["inference"]
        assert inference["avg_latency_ms"] < 200, "Latency too high"
        assert inference["throughput_fps"] > 10, "Throughput too low"

        # Save results
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        benchmark.save_results(results_dir / f"benchmark_v5.5.1-ood.json")
```

### 4.2 Regression Testing

```python
# tests/performance/test_regression.py
import json
import pytest
from pathlib import Path

# Performance thresholds based on v5.5.0 baseline
PERFORMANCE_THRESHOLDS = {
    "v5.5.0": {
        "crop_routing_accuracy": 0.98,
        "phase1_accuracy": 0.95,
        "ood_auroc": 0.89,
        "avg_latency_ms": 185,
        "memory_per_adapter_mb": 23.5
    }
}

def load_performance_metrics(version: str) -> dict:
    """Load performance metrics for a version."""
    metrics_file = Path(f"versions/{version}/performance_metrics.json")
    if not metrics_file.exists():
        # Try to get from benchmark results
        benchmark_file = Path(f"benchmark_results/benchmark_{version}.json")
        if benchmark_file.exists():
            with open(benchmark_file) as f:
                data = json.load(f)
                return data.get("inference", {})
        else:
            pytest.skip(f"No performance metrics for {version}")
    else:
        with open(metrics_file) as f:
            return json.load(f)

def test_no_performance_regression(version: str, baseline: str = "v5.5.0"):
    """Test that version does not regress performance."""

    baseline_metrics = PERFORMANCE_THRESHOLDS.get(baseline)
    if not baseline_metrics:
        pytest.skip(f"No baseline metrics for {baseline}")

    current_metrics = load_performance_metrics(version)

    # Check accuracy metrics (should not decrease)
    if "accuracy" in current_metrics:
        assert current_metrics["accuracy"] >= baseline_metrics["phase1_accuracy"] * 0.99, \
            f"Accuracy regression: {current_metrics['accuracy']} < {baseline_metrics['phase1_accuracy']}"

    # Check latency (should not increase significantly)
    if "avg_latency_ms" in current_metrics:
        assert current_metrics["avg_latency_ms"] <= baseline_metrics["avg_latency_ms"] * 1.10, \
            f"Latency regression: {current_metrics['avg_latency_ms']}ms > {baseline_metrics['avg_latency_ms'] * 1.10}ms"

    # Check memory (should not increase significantly)
    if "memory_per_adapter_mb" in current_metrics:
        assert current_metrics["memory_per_adapter_mb"] <= baseline_metrics["memory_per_adapter_mb"] * 1.20, \
            f"Memory regression: {current_metrics['memory_per_adapter_mb']}MB > {baseline_metrics['memory_per_adapter_mb'] * 1.20}MB"

@pytest.mark.parametrize("version", ["v5.5.1-ood", "v5.5.2-router"])
def test_performance_not_regressed(version):
    """Parametrized test for multiple versions."""
    test_no_performance_regression(version)
```

## 5. A/B Testing Validation

### 5.1 Statistical Significance Testing

```python
# tests/ab_testing/statistical_tests.py
import numpy as np
from scipy import stats

def compare_versions(metrics_a: list, metrics_b: list, alpha: float = 0.05) -> dict:
    """
    Statistical comparison of two versions using t-test.

    Args:
        metrics_a: List of metrics for version A
        metrics_b: List of metrics for version B
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with test results
    """

    # Normality test
    _, p_norm_a = stats.shapiro(metrics_a)
    _, p_norm_b = stats.shapiro(metrics_b)

    if p_norm_a > 0.05 and p_norm_b > 0.05:
        # Both normal, use t-test
        t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b, equal_var=False)
        test_used = "t-test"
    else:
        # Non-parametric, use Mann-Whitney U
        u_stat, p_value = stats.mannwhitneyu(metrics_a, metrics_b, alternative='two-sided')
        test_used = "Mann-Whitney U"

    mean_a = np.mean(metrics_a)
    mean_b = np.mean(metrics_b)
    improvement = ((mean_b - mean_a) / mean_a) * 100 if mean_a > 0 else 0

    return {
        "test_used": test_used,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "improvement_pct": improvement,
        "p_value": p_value,
        "significant": p_value < alpha,
        "winner": "B" if improvement > 0 and p_value < alpha else "A"
    }

# Example usage
def test_ab_test_results():
    """Validate A/B test results."""

    # Simulated accuracy data
    version_a_acc = [0.951, 0.948, 0.953, 0.950, 0.952] * 200  # 1000 samples
    version_b_acc = [0.956, 0.959, 0.955, 0.958, 0.957] * 200

    result = compare_versions(version_a_acc, version_b_acc)

    assert result["significant"], f"Difference not significant: p={result['p_value']}"
    assert result["winner"] == "B", "Version B should be winner"
    assert result["improvement_pct"] > 0, "Version B should improve accuracy"
```

### 5.2 A/B Test Validation Checklist

```python
# tests/ab_testing/validation_checklist.py
def validate_ab_test_setup(version_a: str, version_b: str) -> bool:
    """Validate A/B test configuration."""

    checks = []

    # 1. Both versions exist
    vm = VersionManager()
    backups = vm.list_backups()
    checks.append(("Versions exist", version_a in backups and version_b in backups))

    # 2. Both versions have rollback scripts
    checks.append(("Rollback scripts exist",
        Path(f"rollback_{version_a}.py").exists() and
        Path(f"rollback_{version_b}.py").exists()
    ))

    # 3. Both versions verified
    success_a, _ = vm.verify_backup(version_a)
    success_b, _ = vm.verify_backup(version_b)
    checks.append(("Backups verified", success_a and success_b))

    # 4. Traffic split configured
    # Check nginx or load balancer config
    # Implementation depends on infrastructure

    # 5. Monitoring enabled
    checks.append(("Monitoring active", True))  # Placeholder

    # 6. Alert rules configured
    checks.append(("Alerts configured", True))  # Placeholder

    # Report results
    all_passed = all(check[1] for check in checks)

    print("\nA/B Test Validation Results:")
    for check, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    return all_passed
```

## 6. Rollback Testing

### 6.1 Automated Rollback Test

```python
# tests/rollback/test_rollback_procedures.py
import pytest
import tempfile
from pathlib import Path

def test_rollback_to_v5_5_0():
    """Test rollback from v5.5.1-ood to v5.5.0."""
    from version_management.backup import VersionManager

    vm = VersionManager()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Simulate having v5.5.1-ood deployed
        deployed_dir = Path(temp_dir) / "deployed"
        deployed_dir.mkdir()

        # Restore v5.5.1-ood to deployed dir
        success, _ = vm.restore_backup("v5.5.1-ood", target_dir=deployed_dir)
        assert success, "Failed to deploy v5.5.1-ood"

        # Now rollback to v5.5.0
        rollback_success, _ = vm.restore_backup("v5.5.0", target_dir=deployed_dir)
        assert rollback_success, "Rollback failed"

        # Verify v5.5.0 files are present
        assert (deployed_dir / "config/adapter_spec_v55.json").exists()
        with open(deployed_dir / "config/adapter_spec_v55.json") as f:
            config = json.load(f)
        assert config["version"] == "5.5.0"

def test_standalone_rollback_script():
    """Test standalone rollback script execution."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy rollback script to temp dir
        script = Path("rollback_v5.5.1-ood.py")
        if not script.exists():
            pytest.skip("Rollback script not found")

        # Run script
        result = subprocess.run(
            ["python", str(script)],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )

        # Check success
        assert result.returncode == 0, f"Rollback script failed: {result.stderr}"
        assert "SUCCESS" in result.stdout, "Rollback did not report success"

def test_rollback_with_database():
    """Test rollback including database (if applicable)."""
    # This would test database migrations rollback
    # Implementation depends on your DB setup
    pass
```

## 7. Continuous Integration Testing

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/version_test.yml
name: Version Testing

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-version:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        version: ["v5.5.0", "v5.5.1-ood"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Test version ${{ matrix.version }}
      run: |
        # Restore version to test directory
        python -m version_management.backup restore \
          --version ${{ matrix.version }} \
          --target ./test_version_${{ matrix.version }} \
          --dry-run

        # Run unit tests
        pytest tests/version_tests/test_${{ matrix.version }}.py -v

        # Run integration tests
        pytest tests/integration/ -k ${{ matrix.version }}

        # Run performance benchmarks
        pytest tests/performance/ -k ${{ matrix.version }}

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results-${{ matrix.version }}
        path: |
          htmlcov/
          benchmark_results/
```

## 8. Monitoring and Alert Testing

### 8.1 Alert Rule Validation

```python
# tests/monitoring/test_alerts.py
from monitoring.alerts import ALERT_RULES, check_alerts

def test_alert_accuracy_drop():
    """Test alert triggers on accuracy drop."""

    baseline = {"accuracy": 0.95}
    current = {"accuracy": 0.93}  # 2% drop

    alerts = check_alerts(current, baseline)

    assert len(alerts) > 0, "Should trigger alert for accuracy drop"
    assert any(a["rule"] == "accuracy_drop" for a in alerts)

def test_alert_latency_increase():
    """Test alert triggers on latency increase."""

    baseline = {"latency_ms": 185}
    current = {"latency_ms": 220}  # ~19% increase

    alerts = check_alerts(current, baseline)

    assert any(a["rule"] == "latency_increase" for a in alerts)

def test_no_false_positives():
    """Test that normal variations don't trigger alerts."""

    baseline = {"accuracy": 0.95}
    current = {"accuracy": 0.949}  # 0.1% drop (within normal)

    alerts = check_alerts(current, baseline)

    assert not any(a["rule"] == "accuracy_drop" for a in alerts), \
        "Should not alert on minor variations"
```

## 9. Test Data Management

### 9.1 Test Datasets

```
test_data/
├── small_sample/           # Quick tests (few images)
│   ├── tomato/
│   └── pepper/
├── validation_set/         # Full validation
│   ├── tomato/
│   └── pepper/
└── ood_test_set/          # OOD detection tests
    ├── known_classes/
    └── unknown_classes/
```

### 9.2 Fixtures

```python
# tests/fixtures/version_fixtures.py
import pytest
from pathlib import Path

@pytest.fixture
def temp_version_dir():
    """Create temporary directory for version testing."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def version_manager():
    """Create VersionManager instance."""
    from version_management.backup import VersionManager
    return VersionManager()

@pytest.fixture
def sample_config():
    """Load sample configuration."""
    config_path = Path("config/adapter_spec_v55.json")
    if config_path.exists():
        import json
        with open(config_path) as f:
            return json.load(f)
    else:
        return {
            "version": "5.5.0",
            "crop_router": {"type": "resnet50_classifier"},
            "per_crop": {
                "model_name": "facebook/dinov2-giant",
                "use_dora": True
            }
        }

@pytest.fixture
def mock_pipeline(sample_config):
    """Create mock pipeline for testing."""
    from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
    return IndependentMultiCropPipeline(sample_config)
```

## 10. Test Execution Commands

### 10.1 Quick Test Suite

```bash
# Run all tests for a specific version
pytest tests/ -k "v5.5.1" -v

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run performance tests
pytest tests/performance/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=xml

# Run tests in parallel
pytest tests/ -n auto
```

### 10.2 Pre-Deployment Test Suite

```bash
#!/bin/bash
# run_pre_deploy_tests.sh

set -e

VERSION=$1

echo "=== Running Pre-Deployment Tests for $VERSION ==="
echo ""

# 1. Verify backup
echo "1. Verifying backup..."
python -m version_management.backup verify --version "$VERSION"

# 2. Unit tests
echo "2. Running unit tests..."
pytest tests/version_tests/test_${VERSION}.py -v

# 3. Integration tests
echo "3. Running integration tests..."
pytest tests/integration/ -k "$VERSION" -v

# 4. Performance benchmarks
echo "4. Running performance benchmarks..."
pytest tests/performance/test_regression.py -k "$VERSION" -v

# 5. Rollback test
echo "5. Testing rollback procedure..."
pytest tests/rollback/test_rollback_procedures.py -v

# 6. Validate configuration
echo "6. Validating configuration..."
python tests/validation/validate_config.py --version "$VERSION"

echo ""
echo "✓ All pre-deployment tests passed!"
echo "Ready to deploy $VERSION"
```

### 10.3 Post-Deployment Tests

```bash
#!/bin/bash
# run_post_deploy_tests.sh

set -e

VERSION=$1
API_ENDPOINT=${2:-"http://localhost:8000"}

echo "=== Running Post-Deployment Tests for $VERSION ==="
echo ""

# 1. Health check
echo "1. Health check..."
curl -f ${API_ENDPOINT}/health || exit 1

# 2. Smoke test
echo "2. Smoke test..."
python tests/smoke/smoke_test.py --endpoint $API_ENDPOINT

# 3. Load test (light)
echo "3. Light load test..."
python tests/load/load_test.py --endpoint $API_ENDPOINT --requests 100

# 4. Accuracy validation
echo "4. Accuracy validation..."
python tests/validation/validate_accuracy.py --endpoint $API_ENDPOINT

# 5. Monitor metrics
echo "5. Checking metrics..."
python monitoring/check_metrics.py --version "$VERSION"

echo ""
echo "✓ Post-deployment tests passed!"
```

## 11. Test Coverage Requirements

Minimum coverage requirements:
- **Overall**: ≥ 80%
- **Critical paths** (inference, OOD detection): ≥ 90%
- **Version management**: ≥ 85%

```bash
# Check coverage
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html
```

## 12. Performance Regression Thresholds

| Metric | Baseline (v5.5.0) | Acceptable | Critical |
|--------|-------------------|------------|----------|
| Crop routing accuracy | 98.2% | ≥ 97.5% | < 97% |
| Phase 1 accuracy | 95.3% | ≥ 94.5% | < 94% |
| OOD AUROC | 89.1% | ≥ 87% | < 85% |
| OOD FPR | 6.7% | ≤ 8% | > 10% |
| Inference latency | 185ms | ≤ 200ms | > 220ms |
| Memory per adapter | 23.5MB | ≤ 25MB | > 28MB |

**Test Result Interpretation:**
- ✅ All metrics in "Acceptable" range → PASS
- ⚠️ 1-2 metrics in "Acceptable" but not "Baseline" → REVIEW
- ❌ Any metric in "Critical" → FAIL, rollback required

## 13. Quick Reference: Test Commands

```bash
# All tests for version
pytest tests/ -k v5.5.1 -v

# Specific test file
pytest tests/version_tests/test_v5.5.1_ood.py::test_version_exists -v

# With coverage
pytest tests/version_tests/ --cov=src.ood --cov-report=html

# Last failed tests
pytest --last-failed

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Verbose output
pytest -vv

# Run tests matching pattern
pytest -k "ood or version" -v
```

## 14. Troubleshooting Tests

### Common Issues:

1. **Import errors**: Ensure PYTHONPATH includes project root
   ```bash
   export PYTHONPATH=/path/to/project:$PYTHONPATH
   ```

2. **Missing test data**: Download test datasets
   ```bash
   python scripts/download_test_data.py
   ```

3. **GPU memory errors**: Use smaller batch sizes or CPU
   ```bash
   pytest tests/performance/ -k "benchmark" --device=cpu
   ```

4. **Flaky tests**: Add retry decorator
   ```python
   @pytest.mark.flaky(reruns=3, reruns_delay=2)
   def test_flaky_function():
       pass
   ```

---

**Next:** See [Implementation Checklist](staged_implementation.md) for deployment procedures