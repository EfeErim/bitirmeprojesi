import torch

from src.adapter.multi_scale_fusion import MultiScaleFeatureFusion, select_multiscale_features


def test_select_multiscale_features_clamps_indices():
    states = [torch.randn(2, 4, 8) for _ in range(3)]
    selected = select_multiscale_features(states, [0, 2, 10])
    assert len(selected) == 3
    assert selected[-1] is states[-1]


def test_fusion_returns_stable_shape_for_token_inputs():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=16, num_scales=4)
    features = [torch.randn(3, 6, 8) for _ in range(4)]
    out = fusion(features)
    assert out.shape == (3, 16)


def test_fusion_returns_stable_shape_for_map_inputs():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=12, num_scales=4)
    features = [torch.randn(2, 8, 5, 5) for _ in range(4)]
    out = fusion(features)
    assert out.shape == (2, 12)
