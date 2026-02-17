#!/usr/bin/env python3
"""
Tests for debugging and monitoring module.
"""

import pytest
import torch
from pathlib import Path
import sys
import tempfile
import shutil
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory
from src.debugging.monitoring import (
    DebugMonitor,
    TrainingMonitor,
    ModelDebugger,
    GradientTracker,
    ActivationTracker,
    DebugLogger
)


class TestDebugMonitor:
    """Test main debug monitor."""

    def test_singleton_instance(self):
        """Test that DebugMonitor works as a singleton."""
        monitor1 = DebugMonitor()
        monitor2 = DebugMonitor()
        assert monitor1 is monitor2

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = DebugMonitor()
        monitor.start()
        assert monitor.is_running is True
        monitor.stop()
        assert monitor.is_running is False

    def test_record_metric(self):
        """Test recording a metric."""
        monitor = DebugMonitor()
        monitor.start()
        monitor.record_metric("test_metric", 0.5)
        monitor.stop()

        # Check that metric was recorded
        assert "test_metric" in monitor.metrics

    def test_get_metrics(self):
        """Test retrieving recorded metrics."""
        monitor = DebugMonitor()
        monitor.start()
        monitor.record_metric("accuracy", 0.95)
        monitor.record_metric("loss", 0.1)
        monitor.stop()

        metrics = monitor.get_metrics()
        assert "accuracy" in metrics
        assert "loss" in metrics

    def test_reset_metrics(self):
        """Test resetting metrics."""
        monitor = DebugMonitor()
        monitor.start()
        monitor.record_metric("test", 1.0)
        monitor.reset()
        monitor.stop()

        metrics = monitor.get_metrics()
        assert "test" not in metrics or len(metrics["test"]) == 0


class TestTrainingMonitor:
    """Test training-specific monitoring."""

    @pytest.fixture
    def training_monitor(self):
        """Create a training monitor instance."""
        return TrainingMonitor()

    def test_record_epoch_metrics(self, training_monitor):
        """Test recording epoch-level metrics."""
        training_monitor.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_acc=0.85,
            val_acc=0.88
        )

        assert training_monitor.current_epoch == 1
        assert training_monitor.epoch_metrics[1]["train_loss"] == 0.5

    def test_record_batch_metrics(self, training_monitor):
        """Test recording batch-level metrics."""
        training_monitor.record_batch(
            batch=10,
            loss=0.3,
            learning_rate=0.001,
            gradient_norm=0.5
        )

        assert training_monitor.current_batch == 10
        assert len(training_monitor.batch_metrics) > 0

    def test_early_stopping_check(self, training_monitor):
        """Test early stopping logic."""
        # Simulate improving validation loss
        for i in range(5):
            training_monitor.record_epoch(
                epoch=i,
                train_loss=1.0 - i * 0.1,
                val_loss=1.0 - i * 0.15
            )

        # Should not trigger early stopping
        assert training_monitor.should_stop_early(patience=3) is False

        # Simulate no improvement
        for i in range(5, 10):
            training_monitor.record_epoch(
                epoch=i,
                train_loss=0.5,
                val_loss=0.5
            )

        # Should trigger early stopping after 3 epochs with no improvement
        assert training_monitor.should_stop_early(patience=3) is True

    def test_learning_rate_suggestions(self, training_monitor):
        """Test learning rate adjustment suggestions."""
        # Record loss that's not decreasing
        for i in range(5):
            training_monitor.record_epoch(
                epoch=i,
                train_loss=1.0,  # Constant loss
                val_loss=1.0
            )

        suggestion = training_monitor.suggest_learning_rate_adjustment()
        assert suggestion in ["reduce", "increase", "maintain"]

    def test_get_training_summary(self, training_monitor):
        """Test getting training summary."""
        for i in range(3):
            training_monitor.record_epoch(
                epoch=i,
                train_loss=1.0 - i * 0.1,
                val_loss=1.0 - i * 0.15,
                train_acc=0.7 + i * 0.1,
                val_acc=0.75 + i * 0.1
            )

        summary = training_monitor.get_summary()
        assert "total_epochs" in summary
        assert "best_val_loss" in summary
        assert "best_val_acc" in summary
        assert summary["total_epochs"] == 3


class TestModelDebugger:
    """Test model debugging functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.fc = nn.Linear(16 * 222 * 222, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return SimpleModel()

    def test_check_gradients(self, mock_model):
        """Test gradient checking."""
        debugger = ModelDebugger(mock_model)

        # Create dummy input and compute loss
        dummy_input = torch.randn(4, 3, 224, 224)
        output = mock_model(dummy_input)
        loss = output.sum()
        loss.backward()

        gradient_info = debugger.check_gradients()
        assert "gradient_norms" in gradient_info
        assert "vanishing_gradients" in gradient_info
        assert "exploding_gradients" in gradient_info

    def test_check_weights(self, mock_model):
        """Test weight checking."""
        debugger = ModelDebugger(mock_model)
        weight_info = debugger.check_weights()

        assert "weight_norms" in weight_info
        assert "dead_neurons" in weight_info
        assert "weight_stats" in weight_info

    def test_validate_forward_pass(self, mock_model):
        """Test forward pass validation."""
        debugger = ModelDebugger(mock_model)
        dummy_input = torch.randn(2, 3, 224, 224)

        # Should not raise errors
        result = debugger.validate_forward_pass(dummy_input)
        assert result is not None

    def test_detect_nan_inf(self, mock_model):
        """Test NaN/Inf detection."""
        debugger = ModelDebugger(mock_model)

        # Create input that might cause NaN/Inf
        dummy_input = torch.randn(2, 3, 224, 224)
        has_nan, has_inf = debugger.detect_nan_inf(dummy_input)

        assert isinstance(has_nan, bool)
        assert isinstance(has_inf, bool)

    def test_get_debug_report(self, mock_model):
        """Test generating debug report."""
        debugger = ModelDebugger(mock_model)
        report = debugger.generate_report()

        assert "model_summary" in report
        assert "parameter_stats" in report
        assert "gradient_stats" in report


class TestGradientTracker:
    """Test gradient tracking functionality."""

    def test_track_gradients(self):
        """Test gradient tracking."""
        tracker = GradientTracker()

        # Create simple model
        import torch.nn as nn
        model = nn.Linear(10, 5)

        # Register model
        tracker.register_model(model)

        # Compute gradients
        input_tensor = torch.randn(2, 10)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Track gradients
        tracker.track_gradients()

        # Check that gradients were tracked
        assert len(tracker.gradient_history) > 0

    def test_gradient_flow_analysis(self):
        """Test gradient flow analysis."""
        tracker = GradientTracker()

        # Should be able to analyze gradient flow
        analysis = tracker.analyze_gradient_flow()
        assert "layer_gradients" in analysis
        assert "avg_gradient_norm" in analysis

    def test_reset_tracker(self):
        """Test resetting tracker."""
        tracker = GradientTracker()
        tracker.reset()
        assert len(tracker.gradient_history) == 0


class TestActivationTracker:
    """Test activation tracking functionality."""

    def test_track_activations(self):
        """Test activation tracking."""
        tracker = ActivationTracker()

        # Create simple model
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        tracker.register_model(model)

        # Forward pass
        input_tensor = torch.randn(2, 10)
        with tracker.track_activations():
            _ = model(input_tensor)

        # Check that activations were recorded
        assert len(tracker.activation_history) > 0

    def test_activation_statistics(self):
        """Test activation statistics computation."""
        tracker = ActivationTracker()

        # Should compute statistics
        stats = tracker.get_activation_stats()
        assert "mean_activations" in stats
        assert "std_activations" in stats
        assert "sparsity" in stats

    def test_dead_activation_detection(self):
        """Test detection of dead neurons (always zero)."""
        tracker = ActivationTracker()

        # Should detect dead neurons
        dead_neurons = tracker.find_dead_neurons()
        assert isinstance(dead_neurons, dict)

    def test_reset_tracker(self):
        """Test resetting tracker."""
        tracker = ActivationTracker()
        tracker.reset()
        assert len(tracker.activation_history) == 0


class TestDebugLogger:
    """Test debug logging functionality."""

    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = DebugLogger()
        assert logger is not None

    def test_log_metric(self, tmp_path):
        """Test logging a metric."""
        logger = DebugLogger(log_dir=str(tmp_path))
        logger.log_metric("test_metric", 0.5, step=1)

        # Check that log file was created
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) > 0

    def test_log_message(self, tmp_path):
        """Test logging a message."""
        logger = DebugLogger(log_dir=str(tmp_path))
        logger.log("Test message", level="INFO")

        # Check that message was logged
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) > 0

    def test_log_histogram(self, tmp_path):
        """Test logging histogram data."""
        logger = DebugLogger(log_dir=str(tmp_path))
        data = torch.randn(100)
        logger.log_histogram("test_hist", data, step=1)

        # Check that histogram was logged
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) > 0

    def test_log_model_graph(self, tmp_path):
        """Test logging model graph."""
        import torch.nn as nn

        logger = DebugLogger(log_dir=str(tmp_path))
        model = nn.Linear(10, 5)
        logger.log_model_graph(model, input_size=(1, 10))

        # Check that graph was logged
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) > 0

    def test_export_metrics(self, tmp_path):
        """Test exporting metrics to file."""
        logger = DebugLogger(log_dir=str(tmp_path))

        # Log some metrics
        for i in range(5):
            logger.log_metric("accuracy", 0.8 + i * 0.02, step=i)

        # Export to JSON
        export_path = tmp_path / "metrics.json"
        logger.export_metrics(str(export_path))

        assert export_path.exists()
