"""
Comprehensive unit tests for DynamicOODThreshold.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import scipy.stats as stats

from src.ood.dynamic_thresholds import DynamicOODThreshold, AdaptiveThresholdManager, calibrate_thresholds_using_validation


class TestDynamicOODThresholdInitialization:
    """Test DynamicOODThreshold initialization."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        threshold = DynamicOODThreshold()
        
        assert threshold.threshold_factor == 2.0
        assert threshold.min_val_samples_per_class == 30
        assert threshold.fallback_threshold == 25.0
        assert threshold.confidence_level == 0.95
        assert threshold.max_fallback_threshold == 50.0
        assert threshold.min_fallback_threshold == 10.0
        assert threshold.use_confidence_intervals is True
        assert threshold.thresholds == {}
        assert threshold.class_stats == {}
        assert threshold.confidence_intervals == {}
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        threshold = DynamicOODThreshold(
            threshold_factor=2.5,
            min_val_samples_per_class=50,
            fallback_threshold=30.0,
            confidence_level=0.99,
            max_fallback_threshold=60.0,
            min_fallback_threshold=15.0,
            use_confidence_intervals=False
        )
        
        assert threshold.threshold_factor == 2.5
        assert threshold.min_val_samples_per_class == 50
        assert threshold.fallback_threshold == 30.0
        assert threshold.confidence_level == 0.99
        assert threshold.max_fallback_threshold == 60.0
        assert threshold.min_fallback_threshold == 15.0
        assert threshold.use_confidence_intervals is False
    
    def test_configuration_validation_warnings(self, caplog):
        """Test configuration validation warnings."""
        # Low min samples
        threshold = DynamicOODThreshold(min_val_samples_per_class=20)
        assert "below recommended 30" in caplog.text
        
        # Extreme confidence level
        threshold = DynamicOODThreshold(confidence_level=0.80)
        assert "outside typical range" in caplog.text
        
        # Threshold factor outside range
        threshold = DynamicOODThreshold(threshold_factor=0.5)
        assert "outside typical range" in caplog.text
        
        # Fallback threshold out of bounds
        threshold = DynamicOODThreshold(fallback_threshold=100.0)
        assert "outside bounds" in caplog.text


class TestDynamicOODThresholdConfidenceIntervals:
    """Test confidence interval computation."""
    
    def test_compute_confidence_interval_normal(self):
        """Test confidence interval with normal distribution (large sample)."""
        threshold = DynamicOODThreshold(use_confidence_intervals=True)
        
        # Generate sample data
        np.random.seed(42)
        data = np.random.normal(20, 5, 100)
        
        ci_lower, ci_upper = threshold._compute_confidence_interval(data, 0.95)
        
        # For large samples, should be approximately mean ± 1.96*SEM
        mean = np.mean(data)
        sem = stats.sem(data)
        expected_lower = mean - 1.96 * sem
        expected_upper = mean + 1.96 * sem
        
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower < ci_upper
        # Allow some tolerance for approximation
        assert abs(ci_lower - expected_lower) < 0.1
        assert abs(ci_upper - expected_upper) < 0.1
    
    def test_compute_confidence_interval_t_distribution(self):
        """Test confidence interval with t-distribution (small sample)."""
        threshold = DynamicOODThreshold(use_confidence_intervals=True)
        
        # Small sample
        data = np.array([20.0, 21.0, 19.5, 20.5, 21.5])
        
        ci_lower, ci_upper = threshold._compute_confidence_interval(data, 0.95)
        
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower < ci_upper
        # For small samples, t-distribution should be used (wider intervals)
        mean = np.mean(data)
        assert ci_lower < mean < ci_upper
    
    def test_compute_confidence_interval_single_sample(self):
        """Test confidence interval with single sample (edge case)."""
        threshold = DynamicOODThreshold()
        
        data = np.array([20.0])
        ci_lower, ci_upper = threshold._compute_confidence_interval(data, 0.95)
        
        # With single sample, should return mean as both bounds
        assert ci_lower == 20.0
        assert ci_upper == 20.0
    
    def test_compute_confidence_interval_two_samples(self):
        """Test confidence interval with two samples."""
        threshold = DynamicOODThreshold()
        
        data = np.array([20.0, 22.0])
        ci_lower, ci_upper = threshold._compute_confidence_interval(data, 0.95)
        
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower < ci_upper


class TestDynamicOODThresholdInsufficientSamples:
    """Test fallback strategies for insufficient samples."""
    
    def test_handle_insufficient_samples_zero(self):
        """Test handling with zero samples."""
        threshold = DynamicOODThreshold(fallback_threshold=25.0)
        
        result = threshold._handle_insufficient_samples(0, 0)
        
        assert result == 25.0
    
    def test_handle_insufficient_samples_few(self):
        """Test handling with 1-4 samples (very conservative)."""
        threshold = DynamicOODThreshold(
            fallback_threshold=25.0,
            max_fallback_threshold=50.0
        )
        
        for n in [1, 2, 3, 4]:
            result = threshold._handle_insufficient_samples(0, n)
            expected = min(25.0 * 1.5, 50.0)  # 1.5x base, capped at max
            assert result == expected
    
    def test_handle_insufficient_samples_moderate(self):
        """Test handling with 5-9 samples (moderately conservative)."""
        threshold = DynamicOODThreshold(
            fallback_threshold=25.0,
            max_fallback_threshold=50.0
        )
        
        for n in [5, 6, 7, 8, 9]:
            result = threshold._handle_insufficient_samples(0, n)
            expected = min(25.0 * 1.2, 50.0)  # 1.2x base, capped at max
            assert result == expected
    
    def test_handle_insufficient_samples_borderline(self):
        """Test handling with 10-29 samples (standard fallback)."""
        threshold = DynamicOODThreshold(fallback_threshold=25.0)
        
        for n in [10, 15, 20, 25, 29]:
            result = threshold._handle_insufficient_samples(0, n)
            assert result == 25.0
    
    def test_handle_insufficient_samples_bounds(self):
        """Test that fallback thresholds are within bounds."""
        threshold = DynamicOODThreshold(
            fallback_threshold=25.0,
            min_fallback_threshold=10.0,
            max_fallback_threshold=50.0
        )
        
        # Test with very conservative fallback that would exceed max
        result = threshold._handle_insufficient_samples(0, 2)
        assert 10.0 <= result <= 50.0


class TestDynamicOODThresholdComputeThresholds:
    """Test threshold computation from distances."""
    
    @pytest.fixture
    def mock_mahalanobis(self):
        """Create mock Mahalanobis distance calculator."""
        mock = MagicMock()
        mock.num_classes = 3
        return mock
    
    @pytest.fixture
    def mock_val_loader(self):
        """Create mock validation loader."""
        loader = MagicMock()
        # Simulate batches
        batches = []
        for i in range(5):
            images = torch.randn(16, 3, 224, 224)
            labels = torch.full((16,), i % 3, dtype=torch.long)
            batches.append((images, labels))
        loader.__iter__.return_value = iter(batches)
        loader.__len__.return_value = 5
        return loader
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.eval = MagicMock()
        
        def forward_side_effect(x):
            batch_size = x.shape[0]
            output = MagicMock()
            output.last_hidden_state = torch.randn(batch_size, 1, 768)
            return output
        
        model.side_effect = forward_side_effect
        return model
    
    def test_compute_thresholds_with_sufficient_samples(
        self, mock_mahalanobis, mock_val_loader, mock_model
    ):
        """Test threshold computation with sufficient samples."""
        threshold = DynamicOODThreshold(
            min_val_samples_per_class=10,
            threshold_factor=2.0,
            confidence_level=0.95
        )
        
        # Mock distances
        distances_per_class = {
            0: [15.0, 16.0, 14.5, 15.5, 16.5, 14.0, 15.2, 16.1, 14.8, 15.9, 16.2, 14.9],
            1: [20.0, 21.0, 19.5, 20.5, 21.5, 19.0, 20.2, 21.1, 19.8, 20.9, 21.2, 19.9],
            2: [25.0, 26.0, 24.5, 25.5, 26.5, 24.0, 25.2, 26.1, 24.8, 25.9, 26.2, 24.9]
        }
        
        # Mock mahalanobis compute_distance
        def mock_compute_distance(features, class_idx):
            # Return random distance from the class
            class_distances = distances_per_class[class_idx]
            return torch.tensor(np.random.choice(class_distances))
        
        mock_mahalanobis.compute_distance = mock_compute_distance
        
        # Compute thresholds
        result = threshold.compute_thresholds_from_distances(distances_per_class)
        
        assert len(result) == 3
        for class_idx, thresh in result.items():
            assert isinstance(thresh, float)
            assert thresh > 0
            # Threshold should be higher than mean due to conservative factor
            mean_dist = np.mean(distances_per_class[class_idx])
            assert thresh >= mean_dist
    
    def test_compute_thresholds_with_insufficient_samples(self):
        """Test threshold computation with insufficient samples."""
        threshold = DynamicOODThreshold(
            min_val_samples_per_class=30,
            fallback_threshold=25.0
        )
        
        distances_per_class = {
            0: [15.0, 16.0] * 5,  # 10 samples (insufficient)
            1: [20.0, 21.0] * 20,  # 40 samples (sufficient)
        }
        
        result = threshold.compute_thresholds_from_distances(distances_per_class)
        
        # Class 0 should use fallback
        assert result[0] == 25.0
        
        # Class 1 should compute real threshold
        assert isinstance(result[1], float)
        assert result[1] > 0
    
    def test_compute_thresholds_confidence_interval_usage(self):
        """Test that confidence intervals affect threshold computation."""
        # With confidence intervals
        threshold_with_ci = DynamicOODThreshold(
            use_confidence_intervals=True,
            threshold_factor=2.0
        )
        
        # Without confidence intervals
        threshold_without_ci = DynamicOODThreshold(
            use_confidence_intervals=False,
            threshold_factor=2.0
        )
        
        distances = [15.0, 16.0, 14.5, 15.5, 16.5, 14.0] * 10  # 60 samples
        
        result_with_ci = threshold_with_ci.compute_thresholds_from_distances({0: distances})
        result_without_ci = threshold_without_ci.compute_thresholds_from_distances({0: distances})
        
        # Both should produce thresholds
        assert isinstance(result_with_ci[0], float)
        assert isinstance(result_without_ci[0], float)
        
        # With CI should typically be more conservative (higher threshold)
        # due to using upper confidence bound
        assert result_with_ci[0] >= result_without_ci[0]


class TestDynamicOODThresholdValidation:
    """Test threshold validation."""
    
    def test_validate_thresholds_basic(self):
        """Test basic threshold validation."""
        threshold = DynamicOODThreshold()
        
        # Mock thresholds
        thresholds = {0: 25.0, 1: 30.0, 2: 35.0}
        
        # Mock val_loader
        val_loader = MagicMock()
        val_loader.__iter__.return_value = []
        val_loader.__len__.return_value = 0
        
        # Mock model
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock mahalanobis
        mahalanobis = MagicMock()
        mahalanobis.compute_distance = MagicMock(return_value=torch.tensor(20.0))
        
        metrics = threshold.validate_thresholds(thresholds, val_loader, model, mahalanobis)
        
        assert 'false_positive_rate' in metrics
        assert 'true_negative_rate' in metrics
        assert 'total_in_dist_samples' in metrics
        assert 'num_classes_tested' in metrics
    
    def test_validate_thresholds_with_ood_detection(self):
        """Test validation with actual OOD detection."""
        threshold = DynamicOODThreshold()
        
        thresholds = {0: 25.0}
        
        # Create validation data with some distances above and below threshold
        num_samples = 20
        distances = np.random.normal(20, 5, num_samples)  # Mean 20, std 5
        
        # Mock val_loader
        def mock_val_iter():
            batch_size = 5
            for i in range(0, num_samples, batch_size):
                images = torch.randn(batch_size, 3, 224, 224)
                labels = torch.zeros(batch_size, dtype=torch.long)
                yield images, labels
        
        val_loader = MagicMock()
        val_loader.__iter__.return_value = mock_val_iter()
        val_loader.__len__.return_value = 4
        
        # Mock model
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock mahalanobis to return controlled distances
        mahalanobis = MagicMock()
        distance_iter = iter(distances)
        def mock_compute_distance(features, class_idx):
            return torch.tensor(next(distance_iter))
        mahalanobis.compute_distance = mock_compute_distance
        
        metrics = threshold.validate_thresholds(thresholds, val_loader, model, mahalanobis)
        
        assert 0 <= metrics['false_positive_rate'] <= 1
        assert 0 <= metrics['true_negative_rate'] <= 1
        assert metrics['total_in_dist_samples'] == num_samples


class TestDynamicOODThresholdStatistics:
    """Test threshold statistics computation."""
    
    def test_get_threshold_statistics_empty(self):
        """Test statistics with empty thresholds."""
        threshold = DynamicOODThreshold()
        
        stats = threshold.get_threshold_statistics({})
        
        assert stats['mean_threshold'] == 0.0
        assert stats['std_threshold'] == 0.0
        assert stats['min_threshold'] == 0.0
        assert stats['max_threshold'] == 0.0
        assert stats['num_classes'] == 0
        assert stats['confidence_level'] == 0.95
    
    def test_get_threshold_statistics_with_data(self):
        """Test statistics with actual thresholds."""
        threshold = DynamicOODThreshold()
        
        thresholds = {0: 25.0, 1: 30.0, 2: 20.0, 3: 35.0}
        
        stats = threshold.get_threshold_statistics(thresholds)
        
        assert abs(stats['mean_threshold'] - 27.5) < 0.01
        assert abs(stats['std_threshold'] - 5.59) < 0.1  # Approximate std
        assert stats['min_threshold'] == 20.0
        assert stats['max_threshold'] == 35.0
        assert stats['num_classes'] == 4
        assert stats['confidence_level'] == 0.95
    
    def test_get_class_stats(self):
        """Test getting individual class statistics."""
        threshold = DynamicOODThreshold()
        
        # Manually populate class_stats
        threshold.class_stats = {
            0: {'mean': 20.0, 'std': 2.0, 'n': 50, 'ci_lower': 18.0, 'ci_upper': 22.0, 'threshold': 26.0},
            1: {'mean': 25.0, 'std': 3.0, 'n': 60, 'ci_lower': 23.0, 'ci_upper': 27.0, 'threshold': 31.0}
        }
        
        stats0 = threshold.get_class_stats(0)
        assert stats0['mean'] == 20.0
        assert stats0['threshold'] == 26.0
        
        stats1 = threshold.get_class_stats(1)
        assert stats1['mean'] == 25.0
        assert stats1['threshold'] == 31.0
        
        stats_nonexistent = threshold.get_class_stats(99)
        assert stats_nonexistent is None
    
    def test_get_all_class_stats(self):
        """Test getting all class statistics."""
        threshold = DynamicOODThreshold()
        
        threshold.class_stats = {
            0: {'mean': 20.0, 'threshold': 26.0},
            1: {'mean': 25.0, 'threshold': 31.0}
        }
        
        all_stats = threshold.get_all_class_stats()
        
        assert len(all_stats) == 2
        assert all_stats[0]['mean'] == 20.0
        assert all_stats[1]['mean'] == 25.0


class TestAdaptiveThresholdManager:
    """Test adaptive threshold management."""
    
    @pytest.fixture
    def manager(self):
        """Create adaptive threshold manager."""
        initial_thresholds = {0: 25.0, 1: 30.0, 2: 35.0}
        return AdaptiveThresholdManager(
            initial_thresholds=initial_thresholds,
            adaptation_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
            confidence_level=0.95
        )
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.thresholds == {0: 25.0, 1: 30.0, 2: 35.0}
        assert manager.adaptation_rate == 0.1
        assert manager.min_threshold == 1.0
        assert manager.max_threshold == 100.0
        assert manager.confidence_level == 0.95
        assert len(manager.threshold_history) == 3
        assert all(len(hist) == 1 for hist in manager.threshold_history.values())
    
    def test_update_thresholds_new_class(self, manager):
        """Test adding thresholds for new classes."""
        new_thresholds = {0: 26.0, 1: 31.0, 2: 34.0, 3: 40.0}
        
        updated = manager.update_thresholds(new_thresholds)
        
        assert 3 in updated
        assert updated[3] == 40.0
        assert manager.get_update_count(3) == 1
        assert manager.get_threshold_history(3) == [40.0]
    
    def test_update_thresholds_adaptation(self, manager):
        """Test exponential moving average adaptation."""
        new_thresholds = {0: 30.0}  # Higher than current 25.0
        
        updated = manager.update_thresholds(new_thresholds)
        
        # Should adapt: 25.0 + 0.1 * (30.0 - 25.0) = 25.5
        assert updated[0] == 25.5
        assert manager.get_threshold_for_class(0) == 25.5
        
        # Update again
        new_thresholds2 = {0: 35.0}
        updated2 = manager.update_thresholds(new_thresholds2)
        
        # Adapt from 25.5: 25.5 + 0.1 * (35.0 - 25.5) = 26.45
        assert abs(updated2[0] - 26.45) < 0.01
    
    def test_update_thresholds_bounds(self, manager):
        """Test that thresholds are clamped to bounds."""
        # Test lower bound
        new_thresholds = {0: 0.5}  # Below min 1.0
        updated = manager.update_thresholds(new_thresholds)
        assert updated[0] == 1.0
        
        # Test upper bound
        new_thresholds = {0: 150.0}  # Above max 100.0
        updated = manager.update_thresholds(new_thresholds)
        assert updated[0] == 100.0
    
    def test_threshold_history_tracking(self, manager):
        """Test threshold history tracking."""
        updates = [
            {0: 30.0},
            {0: 28.0},
            {0: 32.0}
        ]
        
        for new_thresh in updates:
            manager.update_thresholds(new_thresh)
        
        history = manager.get_threshold_history(0)
        assert len(history) == 4  # Initial + 3 updates
        assert history[0] == 25.0  # Initial
        assert history[-1] == pytest.approx(25.85, abs=0.01)  # Final after adaptations
    
    def test_get_threshold_for_class_missing(self, manager):
        """Test getting threshold for missing class returns default."""
        threshold = manager.get_threshold_for_class(99)
        assert threshold == 25.0  # Default fallback


class TestCalibrateThresholdsUsingValidation:
    """Test threshold calibration function."""
    
    @pytest.fixture
    def calibration_setup(self):
        """Setup for calibration tests."""
        # Mock model
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock val_loader with class-distributed distances
        class_distributions = {
            0: {'mean': 15.0, 'std': 2.0},
            1: {'mean': 25.0, 'std': 3.0},
            2: {'mean': 35.0, 'std': 4.0}
        }
        
        def create_mock_loader():
            loader = MagicMock()
            all_data = []
            for class_idx, dist in class_distributions.items():
                distances = np.random.normal(dist['mean'], dist['std'], 50)
                for distance in distances:
                    all_data.append((class_idx, distance))
            np.random.shuffle(all_data)
            
            def mock_iter():
                batch_size = 10
                for i in range(0, len(all_data), batch_size):
                    batch = all_data[i:i+batch_size]
                    images = torch.randn(len(batch), 3, 224, 224)
                    labels = torch.tensor([item[0] for item in batch])
                    # Store distances for later retrieval
                    for j, (class_idx, distance) in enumerate(batch):
                        setattr(images[j], '_distance', distance)
                    yield images, labels
            
            loader.__iter__.return_value = mock_iter()
            loader.__len__.return_value = 5
            return loader
        
        # Mock mahalanobis
        mahalanobis = MagicMock()
        def mock_compute(features, class_idx):
            # Get distance from the first feature in batch
            if hasattr(features[0], '_distance'):
                return torch.tensor(features[0]._distance)
            return torch.tensor(class_distributions[class_idx]['mean'])
        mahalanobis.compute_distance = mock_compute
        
        return {
            'model': model,
            'val_loader': create_mock_loader(),
            'mahalanobis': mahalanobis,
            'class_distributions': class_distributions
        }
    
    def test_calibrate_thresholds_basic(self, calibration_setup):
        """Test basic threshold calibration."""
        setup = calibration_setup
        
        thresholds = calibrate_thresholds_using_validation(
            model=setup['model'],
            val_loader=setup['val_loader'],
            mahalanobis=setup['mahalanobis'],
            target_fpr=0.05,
            device='cpu',
            confidence_level=0.95,
            min_samples=30
        )
        
        assert len(thresholds) > 0
        for class_idx, threshold in thresholds.items():
            assert isinstance(threshold, float)
            assert threshold > 0
    
    def test_calibrate_thresholds_target_fpr(self):
        """Test that calibration targets specific FPR."""
        # Create controlled data where we know the distribution
        class_distributions = {
            0: {'mean': 10.0, 'std': 1.0}  # Tight distribution
        }
        
        # Generate 100 samples from this distribution
        distances_0 = np.random.normal(10.0, 1.0, 100)
        
        # Mock setup
        model = MagicMock()
        model.eval = MagicMock()
        
        val_loader = MagicMock()
        def mock_iter():
            batch_size = 20
            for i in range(0, 100, batch_size):
                images = torch.randn(batch_size, 3, 224, 224)
                labels = torch.zeros(batch_size, dtype=torch.long)
                for j in range(batch_size):
                    setattr(images[j], '_distance', distances_0[i+j])
                yield images, labels
        val_loader.__iter__.return_value = mock_iter()
        val_loader.__len__.return_value = 5
        
        mahalanobis = MagicMock()
        def mock_compute(features, class_idx):
            return torch.tensor(features[0]._distance)
        mahalanobis.compute_distance = mock_compute
        
        thresholds = calibrate_thresholds_using_validation(
            model=model,
            val_loader=val_loader,
            mahalanobis=mahalanobis,
            target_fpr=0.10,  # 10% FPR
            device='cpu',
            min_samples=10
        )
        
        # For class 0, threshold should be around the 90th percentile
        # (since FPR=0.10 means 90% of in-dist samples should be below threshold)
        expected_90th = np.percentile(distances_0, 90)
        assert 0 in thresholds
        # Should be close to 90th percentile (with some margin for confidence interval)
        assert abs(thresholds[0] - expected_90th) < 2.0
    
    def test_calibrate_thresholds_insufficient_samples(self):
        """Test calibration with insufficient samples."""
        model = MagicMock()
        model.eval = MagicMock()
        
        # Only 5 samples (below min_samples=30)
        val_loader = MagicMock()
        distances = np.random.normal(20, 5, 5)
        def mock_iter():
            images = torch.randn(5, 3, 224, 224)
            labels = torch.zeros(5, dtype=torch.long)
            for i in range(5):
                setattr(images[i], '_distance', distances[i])
            yield images, labels
        val_loader.__iter__.return_value = iter([next(mock_iter())])
        val_loader.__len__.return_value = 1
        
        mahalanobis = MagicMock()
        mahalanobis.compute_distance = lambda f, c: torch.tensor(f[0]._distance)
        
        thresholds = calibrate_thresholds_using_validation(
            model=model,
            val_loader=val_loader,
            mahalanobis=mahalanobis,
            target_fpr=0.05,
            device='cpu',
            min_samples=30
        )
        
        # Should use fallback threshold
        assert thresholds[0] == 25.0


class TestDynamicOODThresholdEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_threshold_computation_empty_distances(self):
        """Test threshold computation with empty distance list."""
        threshold = DynamicOODThreshold()
        
        result = threshold.compute_thresholds_from_distances({0: []})
        
        # Should use fallback for empty
        assert result[0] == 25.0
    
    def test_threshold_factor_impact(self):
        """Test impact of threshold_factor on final threshold."""
        distances = np.random.normal(20, 3, 100).tolist()
        
        threshold_small = DynamicOODThreshold(threshold_factor=1.0)
        threshold_large = DynamicOODThreshold(threshold_factor=3.0)
        
        result_small = threshold_small.compute_thresholds_from_distances({0: distances})
        result_large = threshold_large.compute_thresholds_from_distances({0: distances})
        
        # Larger factor should produce more conservative (higher) thresholds
        assert result_large[0] > result_small[0]
    
    def test_confidence_level_impact(self):
        """Test impact of confidence level on threshold."""
        distances = np.random.normal(20, 3, 100).tolist()
        
        threshold_95 = DynamicOODThreshold(confidence_level=0.95)
        threshold_99 = DynamicOODThreshold(confidence_level=0.99)
        
        result_95 = threshold_95.compute_thresholds_from_distances({0: distances})
        result_99 = threshold_99.compute_thresholds_from_distances({0: distances})
        
        # Higher confidence level should produce more conservative (higher) thresholds
        # due to wider confidence intervals
        assert result_99[0] >= result_95[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])