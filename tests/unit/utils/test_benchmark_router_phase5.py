import torch

from scripts.benchmark_router_phase5 import build_pipeline, run_benchmark


def test_build_pipeline_runs_mocked_sam3_analysis_surface():
    pipeline = build_pipeline(['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'])
    result = pipeline.analyze_image(torch.rand(3, 224, 224))

    assert isinstance(result, dict)
    assert 'detections' in result
    assert 'stage_timings_ms' in result
    assert result.get('pipeline_type') == 'sam3_bioclip25'
    assert len(result['detections']) >= 1


def test_run_benchmark_returns_expected_contract():
    benchmark = run_benchmark(
        name='smoke',
        stage_order=['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess'],
        iterations=4,
    )

    assert benchmark.name == 'smoke'
    assert benchmark.iterations == 4
    assert benchmark.avg_wall_ms >= 0.0
    assert benchmark.p95_wall_ms >= benchmark.p50_wall_ms
    assert benchmark.avg_processing_ms >= 0.0
    assert benchmark.avg_detections >= 0.0
    assert {'preprocess', 'sam3_inference', 'roi_total', 'postprocess'} <= set(benchmark.avg_stage_timings_ms.keys())
