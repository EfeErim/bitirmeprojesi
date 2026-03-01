#!/usr/bin/env python3
"""
Tests for evaluation metrics module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tests.fixtures.test_fixtures import mock_tensor_factory
from src.evaluation.metrics import (
    ClassificationMetrics,
    SegmentationMetrics,
    DetectionMetrics,
    compute_accuracy,
    compute_precision_recall_f1,
    compute_ap,
    compute_auc
)


class TestClassificationMetrics:
    """Test classification metrics."""

    @pytest.mark.parametrize("y_true,y_pred,expected", [
        (torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 1, 2, 0, 1, 2]), 1.0),
        (torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 0, 2, 0, 1, 1]), 4/6)
    ])
    def test_accuracy(self, y_true, y_pred, expected):
        """Test accuracy computation."""
        metrics = ClassificationMetrics()
        acc = metrics.accuracy(y_true, y_pred)
        assert abs(acc - expected) < 1e-6

    def test_precision_recall_f1(self):
        """Test precision, recall, F1 computation."""
        y_true = torch.tensor([0, 0, 1, 1, 2, 2])
        y_pred = torch.tensor([0, 1, 1, 1, 2, 0])
        metrics = ClassificationMetrics()

        results = metrics.precision_recall_f1(y_true, y_pred, average='macro')
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1'] <= 1

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        y_true = torch.tensor([0, 1, 2, 0, 1, 2])
        y_pred = torch.tensor([0, 1, 2, 0, 1, 2])
        metrics = ClassificationMetrics()
        cm = metrics.confusion_matrix(y_true, y_pred, num_classes=3)

        assert cm.shape == (3, 3)
        assert torch.diag(cm).sum() == 6  # All correct

    def test_per_class_metrics(self):
        """Test per-class metrics computation."""
        y_true = torch.tensor([0, 0, 1, 1, 2, 2])
        y_pred = torch.tensor([0, 1, 1, 1, 2, 0])
        metrics = ClassificationMetrics()

        per_class = metrics.per_class_metrics(y_true, y_pred, num_classes=3)
        assert len(per_class) == 3
        for class_metrics in per_class:
            assert 'precision' in class_metrics
            assert 'recall' in class_metrics
            assert 'f1' in class_metrics


class TestSegmentationMetrics:
    """Test segmentation metrics."""

    def test_iou(self):
        """Test Intersection over Union computation."""
        # Create simple binary masks
        pred = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        target = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)

        metrics = SegmentationMetrics()
        iou = metrics.iou(pred, target)
        # Intersection: 1, Union: 3 -> IoU = 1/3
        expected = 1.0 / 3.0
        assert abs(iou - expected) < 1e-6

    def test_dice_coefficient(self):
        """Test Dice coefficient computation."""
        pred = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        target = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)

        metrics = SegmentationMetrics()
        dice = metrics.dice_coefficient(pred, target)
        # Dice = 2*|A∩B| / (|A| + |B|) = 2*1 / (2+2) = 0.5
        expected = 0.5
        assert abs(dice - expected) < 1e-6

    def test_mean_iou(self):
        """Test mean IoU across multiple classes."""
        # Batch of 2 samples, 2 classes
        pred = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.float32)
        target = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.float32)

        metrics = SegmentationMetrics()
        miou = metrics.mean_iou(pred, target, num_classes=2)
        assert miou == 1.0  # Perfect match

    def test_pixel_accuracy(self):
        """Test pixel accuracy computation."""
        pred = torch.tensor([[0, 1], [1, 0]])
        target = torch.tensor([[0, 1], [1, 1]])

        metrics = SegmentationMetrics()
        acc = metrics.pixel_accuracy(pred, target)
        # 3 out of 4 correct
        assert acc == 0.75


class TestDetectionMetrics:
    """Test object detection metrics."""

    def test_iou_bbox(self):
        """Test IoU for bounding boxes."""
        # Format: [x1, y1, x2, y2]
        bbox1 = torch.tensor([0, 0, 10, 10])
        bbox2 = torch.tensor([5, 5, 15, 15])

        metrics = DetectionMetrics()
        iou = metrics.bbox_iou(bbox1, bbox2)

        # Intersection: (5,5)-(10,10) = 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.1429
        expected = 25.0 / 175.0
        assert abs(iou - expected) < 1e-4

    def test_precision_recall_at_iou(self):
        """Test precision/recall at specific IoU threshold."""
        # Simple case: 2 predictions, 1 ground truth
        pred_boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]])
        pred_scores = torch.tensor([0.9, 0.8])
        gt_boxes = torch.tensor([[0, 0, 10, 10]])

        metrics = DetectionMetrics()
        results = metrics.compute_ap(
            pred_boxes, pred_scores, gt_boxes,
            iou_threshold=0.5
        )

        assert 'ap' in results
        assert 0 <= results['ap'] <= 1

    def test_map(self):
        """Test mean Average Precision."""
        # Multiple classes
        all_pred_boxes = [
            torch.tensor([[0, 0, 10, 10]]),  # class 0
            torch.tensor([[5, 5, 15, 15]])   # class 1
        ]
        all_pred_scores = [
            torch.tensor([0.9]),
            torch.tensor([0.85])
        ]
        all_gt_boxes = [
            torch.tensor([[0, 0, 10, 10]]),  # class 0
            torch.tensor([[5, 5, 15, 15]])   # class 1
        ]

        metrics = DetectionMetrics()
        map_score = metrics.compute_map(
            all_pred_boxes, all_pred_scores, all_gt_boxes
        )

        assert 0 <= map_score <= 1


class TestUtilityFunctions:
    """Test utility metric functions."""

    def test_compute_accuracy(self):
        """Test compute_accuracy function."""
        y_true = torch.tensor([0, 1, 2, 0, 1])
        y_pred = torch.tensor([0, 1, 1, 0, 1])
        acc = compute_accuracy(y_true, y_pred)
        assert abs(acc - 0.8) < 1e-6

    def test_compute_precision_recall_f1(self):
        """Test compute_precision_recall_f1 function."""
        y_true = torch.tensor([0, 0, 1, 1, 1])
        y_pred = torch.tensor([0, 1, 1, 1, 0])
        results = compute_precision_recall_f1(y_true, y_pred, average='binary')

        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results

    def test_compute_ap(self):
        """Test compute_ap function."""
        pred_scores = torch.tensor([0.9, 0.8, 0.7])
        y_true = torch.tensor([1, 1, 0])
        ap = compute_ap(pred_scores, y_true)
        assert 0 <= ap <= 1

    def test_compute_auc(self):
        """Test compute_auc function."""
        y_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        y_true = torch.tensor([1, 1, 0, 1, 0])
        auc = compute_auc(y_scores, y_true)
        assert 0 <= auc <= 1
