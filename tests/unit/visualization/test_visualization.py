#!/usr/bin/env python3
"""
Tests for visualization module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory
from src.visualization.visualization import (
    PlotGenerator,
    ConfusionMatrixPlotter,
    ROCPlotter,
    PrecisionRecallPlotter,
    TrainingCurvePlotter,
    AttentionVisualizer,
    GradCAMVisualizer,
    save_plot,
    set_plot_style
)


class TestPlotGenerator:
    """Test base plot generator."""

    @pytest.fixture
    def plot_generator(self):
        """Create a plot generator instance."""
        return PlotGenerator()

    def test_initialization(self, plot_generator):
        """Test plot generator initialization."""
        assert plot_generator.figsize == (10, 6)
        assert plot_generator.dpi == 100
        assert plot_generator.style == "seaborn"

    def test_set_style(self, plot_generator):
        """Test setting plot style."""
        plot_generator.set_style("ggplot")
        assert plot_generator.style == "ggplot"

    def test_create_figure(self, plot_generator):
        """Test figure creation."""
        fig, ax = plot_generator.create_figure()
        assert fig is not None
        assert ax is not None

    def test_save_plot(self, plot_generator, tmp_path):
        """Test saving a plot."""
        fig, ax = plot_generator.create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        save_path = tmp_path / "test_plot.png"
        plot_generator.save_plot(fig, str(save_path))

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_save_plot_without_extension(self, plot_generator, tmp_path):
        """Test saving plot without explicit extension."""
        fig, ax = plot_generator.create_figure()
        ax.plot([1, 2, 3], [1, 4, 9])

        save_path = tmp_path / "test_plot"
        plot_generator.save_plot(fig, str(save_path), format="png")

        # Should add extension
        assert (tmp_path / "test_plot.png").exists()


class TestConfusionMatrixPlotter:
    """Test confusion matrix plotting."""

    @pytest.fixture
    def plotter(self):
        """Create confusion matrix plotter."""
        return ConfusionMatrixPlotter()

    @pytest.mark.parametrize("cm,class_names,title,normalize", [
        (np.array([[50, 5, 2], [3, 45, 7], [1, 4, 48]]), ["class_a", "class_b", "class_c"], "Test Confusion Matrix", False),
        (np.array([[50, 5, 2], [3, 45, 7], [1, 4, 48]]), ["a", "b", "c"], "Normalized CM", True),
        (np.eye(3), None, "Identity Matrix", False)
    ])
    def test_plot_confusion_matrix(self, plotter, tmp_path, cm, class_names, title, normalize):
        """Test plotting confusion matrix."""
        kwargs = {"cm": cm, "class_names": class_names, "title": title}
        if normalize:
            kwargs["normalize"] = True

        fig = plotter.plot_confusion_matrix(**kwargs)

        assert fig is not None

        # Save to verify rendering works (only for first test case with tmp_path)
        if tmp_path and not normalize and class_names == ["class_a", "class_b", "class_c"]:
            save_path = tmp_path / "confusion_matrix.png"
            plotter.save_plot(fig, str(save_path))
            assert save_path.exists()
        fig = plotter.plot_confusion_matrix(cm)
        assert fig is not None


class TestROCPlotter:
    """Test ROC curve plotting."""

    @pytest.fixture
    def plotter(self):
        """Create ROC plotter."""
        return ROCPlotter()

    def test_plot_roc_curve(self, plotter, tmp_path):
        """Test plotting ROC curve."""
        # Create sample predictions and labels
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4])

        fig = plotter.plot_roc_curve(y_true, y_score)
        assert fig is not None

        save_path = tmp_path / "roc_curve.png"
        plotter.save_plot(fig, str(save_path))
        assert save_path.exists()

    def test_plot_multiclass_roc(self, plotter):
        """Test plotting multiclass ROC curves."""
        # Multiclass case
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])

        fig = plotter.plot_roc_curve(y_true, y_score, multiclass=True)
        assert fig is not None

    def test_roc_auc_display(self, plotter):
        """Test that ROC curve displays AUC."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.8, 0.9, 0.7])

        fig = plotter.plot_roc_curve(y_true, y_score)
        # Should have AUC in the plot (can't easily test without inspecting figure)


class TestPrecisionRecallPlotter:
    """Test precision-recall curve plotting."""

    @pytest.fixture
    def plotter(self):
        """Create P-R plotter."""
        return PrecisionRecallPlotter()

    def test_plot_pr_curve(self, plotter, tmp_path):
        """Test plotting precision-recall curve."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4])

        fig = plotter.plot_pr_curve(y_true, y_score)
        assert fig is not None

        save_path = tmp_path / "pr_curve.png"
        plotter.save_plot(fig, str(save_path))
        assert save_path.exists()

    def plot_average_precision(self, plotter):
        """Test that AP is displayed."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.4, 0.8, 0.9, 0.7])

        fig = plotter.plot_pr_curve(y_true, y_score)
        # Should have AP in the plot


class TestTrainingCurvePlotter:
    """Test training curve plotting."""

    @pytest.fixture
    def plotter(self):
        """Create training curve plotter."""
        return TrainingCurvePlotter()

    def test_plot_loss_curves(self, plotter, tmp_path):
        """Test plotting loss curves."""
        train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
        val_losses = [1.1, 0.9, 0.7, 0.6, 0.5]

        fig = plotter.plot_loss_curves(
            train_losses=train_losses,
            val_losses=val_losses
        )

        assert fig is not None

        save_path = tmp_path / "loss_curves.png"
        plotter.save_plot(fig, str(save_path))
        assert save_path.exists()

    def test_plot_accuracy_curves(self, plotter):
        """Test plotting accuracy curves."""
        train_acc = [0.5, 0.7, 0.8, 0.85, 0.9]
        val_acc = [0.45, 0.65, 0.75, 0.8, 0.85]

        fig = plotter.plot_accuracy_curves(
            train_acc=train_acc,
            val_acc=val_acc
        )

        assert fig is not None

    def test_plot_metrics_comparison(self, plotter):
        """Test plotting multiple metrics comparison."""
        metrics = {
            'accuracy': [0.7, 0.8, 0.85, 0.9],
            'precision': [0.65, 0.78, 0.82, 0.88],
            'recall': [0.72, 0.81, 0.86, 0.91]
        }

        fig = plotter.plot_metrics_comparison(metrics)
        assert fig is not None


class TestAttentionVisualizer:
    """Test attention visualization."""

    @pytest.fixture
    def visualizer(self):
        """Create attention visualizer."""
        return AttentionVisualizer()

    def test_plot_attention_heatmap(self, visualizer, tmp_path):
        """Test plotting attention heatmap."""
        # Create sample attention weights
        attention = np.random.rand(10, 10)

        fig = visualizer.plot_attention_heatmap(
            attention,
            tokens=["token_" + str(i) for i in range(10)],
            title="Test Attention"
        )

        assert fig is not None

        save_path = tmp_path / "attention_heatmap.png"
        visualizer.save_plot(fig, str(save_path))
        assert save_path.exists()

    def test_plot_multihead_attention(self, visualizer):
        """Test plotting multi-head attention."""
        # Multi-head attention: [num_heads, seq_len, seq_len]
        attention = np.random.rand(4, 8, 8)

        fig = visualizer.plot_multihead_attention(
            attention,
            tokens=[f"t{i}" for i in range(8)]
        )

        assert fig is not None

    def test_plot_attention_rollout(self, visualizer):
        """Test plotting attention rollout."""
        # List of attention matrices from different layers
        attentions = [np.random.rand(5, 5) for _ in range(3)]

        fig = visualizer.plot_attention_rollout(attentions)
        assert fig is not None


class TestGradCAMVisualizer:
    """Test Grad-CAM visualization."""

    @pytest.fixture
    def visualizer(self):
        """Create Grad-CAM visualizer."""
        return GradCAMVisualizer()

    def test_generate_gradcam(self, visualizer):
        """Test generating Grad-CAM heatmap."""
        # This would need a real model and image
        # For now, test with mock data
        import torch.nn as nn

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.fc = nn.Linear(16 * 222 * 222, 10)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleModel()
        input_tensor = torch.randn(1, 3, 224, 224)

        # Generate Grad-CAM
        heatmap = visualizer.generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_layer=model.conv
        )

        assert heatmap is not None
        assert isinstance(heatmap, np.ndarray)

    def test_overlay_heatmap(self, visualizer, tmp_path):
        """Test overlaying heatmap on image."""
        # Create dummy image and heatmap
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224)

        result = visualizer.overlay_heatmap(image, heatmap, alpha=0.5)

        assert result is not None
        assert result.shape == image.shape

    def test_plot_gradcam_comparison(self, visualizer, tmp_path):
        """Test plotting Grad-CAM comparison."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224)

        fig = visualizer.plot_gradcam_comparison(image, heatmap)
        assert fig is not None

        save_path = tmp_path / "gradcam.png"
        visualizer.save_plot(fig, str(save_path))
        assert save_path.exists()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_save_plot_function(self, tmp_path):
        """Test save_plot convenience function."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        save_path = tmp_path / "test.png"
        save_plot(fig, str(save_path))

        assert save_path.exists()

    def test_set_plot_style_function(self):
        """Test set_plot_style convenience function."""
        set_plot_style("ggplot")
        # Should set global style
        assert True  # If no error, test passes

    def test_color_palette(self):
        """Test color palette generation."""
        from src.visualization.visualization import get_color_palette

        palette = get_color_palette("tab10", 10)
        assert len(palette) == 10

    def test_colormap_generation(self):
        """Test colormap generation."""
        from src.visualization.visualization import get_colormap

        cmap = get_colormap("viridis")
        assert cmap is not None
