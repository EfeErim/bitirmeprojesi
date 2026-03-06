from typing import Any, Dict, Tuple


def analysis_to_batch_item(analysis: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """Map analysis payload to route_batch item contract (crop dict + confidence)."""
    detections = analysis.get('detections') if isinstance(analysis, dict) else None
    if detections:
        detection = detections[0]
        return (
            {
                'crop': detection.get('crop', 'unknown'),
                'part': detection.get('part', 'unknown'),
                'bbox': detection.get('bbox', [0, 0, 0, 0]),
            },
            detection.get('crop_confidence', 0.0),
        )

    return ({'crop': 'unknown', 'part': 'unknown', 'bbox': [0, 0, 0, 0]}, 0.0)
