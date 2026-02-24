#!/usr/bin/env python3
"""Phase 5 router latency benchmark with deterministic mocked model hooks."""

import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router.vlm_pipeline import VLMPipeline


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    avg_wall_ms: float
    p50_wall_ms: float
    p95_wall_ms: float
    avg_processing_ms: float
    avg_stage_timings_ms: Dict[str, float]
    avg_detections: float


def build_pipeline(stage_order: List[str]) -> VLMPipeline:
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
                'crop_labels': ['tomato', 'potato'],
                'part_labels': ['leaf', 'whole plant'],
                'policy_graph': {
                    'execution': {
                        'sam3_stage_order': stage_order,
                    },
                    'dedupe': {
                        'enabled': True,
                        'detection_nms_iou_threshold': 0.75,
                        'detection_nms_same_crop_only': True,
                    },
                },
            }
        },
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = VLMPipeline(config=config, device='cpu')

    pipeline.models_loaded = True
    pipeline.actual_pipeline = 'sam3'

    pipeline._run_sam3 = lambda image, prompt, threshold=0.7: {
        'masks': torch.ones((2, 4, 4), dtype=torch.float32),
        'boxes': [[0.0, 0.0, 100.0, 100.0], [5.0, 5.0, 105.0, 105.0]],
        'scores': [0.95, 0.94],
    }

    def fake_ensemble(_image, _labels, label_type='generic', num_prompts=None):
        if label_type == 'part':
            return 'leaf', 0.80, {'leaf': 0.80, 'whole plant': 0.20}
        return 'tomato', 0.96, {'tomato': 0.96, 'potato': 0.04}

    pipeline._clip_score_labels_ensemble = fake_ensemble
    pipeline._select_best_crop_with_fallback = (
        lambda roi_image, crop_scores, part_label, part_scores, min_confidence=0.2: ('tomato', 0.96)
    )
    return pipeline


def run_benchmark(name: str, stage_order: List[str], iterations: int = 200) -> BenchmarkResult:
    pipeline = build_pipeline(stage_order)
    image = torch.rand(3, 224, 224)

    wall_times: List[float] = []
    processing_times: List[float] = []
    detections: List[int] = []
    stage_aggr: Dict[str, List[float]] = {
        'preprocess': [],
        'sam3_inference': [],
        'roi_total': [],
        'roi_classification': [],
        'postprocess': [],
        'avg_roi': [],
        'avg_classification_call': [],
    }

    for _ in range(iterations):
        t0 = time.perf_counter()
        result = pipeline.analyze_image(image)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        wall_times.append(wall_ms)

        processing_times.append(float(result.get('processing_time_ms', 0.0)))
        detections.append(len(result.get('detections', [])))

        stage_timings = result.get('stage_timings_ms', {}) or {}
        for key in stage_aggr:
            if key in stage_timings:
                stage_aggr[key].append(float(stage_timings[key]))

    wall_sorted = sorted(wall_times)
    p50 = wall_sorted[len(wall_sorted) // 2]
    p95 = wall_sorted[min(len(wall_sorted) - 1, int(len(wall_sorted) * 0.95))]

    avg_stage = {
        key: (statistics.mean(values) if values else 0.0)
        for key, values in stage_aggr.items()
    }

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        avg_wall_ms=statistics.mean(wall_times),
        p50_wall_ms=p50,
        p95_wall_ms=p95,
        avg_processing_ms=statistics.mean(processing_times),
        avg_stage_timings_ms=avg_stage,
        avg_detections=statistics.mean(detections),
    )


def main() -> None:
    scenarios = [
        ('full_pipeline', ['roi_filter', 'roi_classification', 'open_set_gate', 'postprocess']),
        ('no_postprocess', ['roi_filter', 'roi_classification', 'open_set_gate']),
        ('no_open_set_gate', ['roi_filter', 'roi_classification', 'postprocess']),
    ]

    results = [asdict(run_benchmark(name, order)) for name, order in scenarios]
    payload: Dict[str, Any] = {
        'generated_at_epoch': time.time(),
        'device': 'cpu',
        'results': results,
    }

    out_dir = Path('logs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'phase5_router_benchmark.json'
    out_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(json.dumps(payload, indent=2))
    print(f"\nSaved benchmark report to: {out_file}")


if __name__ == '__main__':
    main()
