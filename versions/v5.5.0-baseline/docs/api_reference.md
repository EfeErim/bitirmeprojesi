# AADS-ULoRA v5.5 API Reference

## Base URL

```
Production: https://api.uyumsoft.com/v1
Staging: https://staging-api.uyumsoft.com/v1
```

## Authentication

Currently, the API uses IP whitelisting. Future versions will support JWT tokens.

## Endpoints

### Health Check

**GET** `/health`

Check API and model service health.

**Response:**
```json
{
  "status": "healthy",
  "router_loaded": true,
  "adapters_loaded": ["tomato", "pepper"],
  "device": "cuda"
}
```

---

### List Crops

**GET** `/crops`

List all supported crop types.

**Response:**
```json
{
  "crops": ["tomato", "pepper", "corn"],
  "router_accuracy_target": 0.98
}
```

---

### Get Adapter Status

**GET** `/adapters/{crop}/status`

Get status of a specific crop adapter.

**Path Parameters:**
- `crop` (string): Crop name (e.g., "tomato")

**Response:**
```json
{
  "crop": "tomato",
  "is_trained": true,
  "current_phase": 3,
  "num_classes": 5,
  "classes": ["healthy", "early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot"],
  "has_ood": true
}
```

---

### Diagnose Disease

**POST** `/diagnose`

Main endpoint for disease diagnosis with OOD detection.

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "base64_encoded_jpeg_string",
  "crop_hint": "tomato",
  "location": {
    "latitude": 41.0082,
    "longitude": 28.9784,
    "accuracy_meters": 10.0
  },
  "metadata": {
    "capture_timestamp": "2026-03-15T14:30:00Z",
    "device_model": "iPhone14,2",
    "os_version": "iOS 17.4"
  }
}
```

**Field Descriptions:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | string | Yes | Base64-encoded JPEG image |
| crop_hint | string | No | Optional crop type hint (speeds up routing) |
| location.latitude | float | No | GPS latitude |
| location.longitude | float | No | GPS longitude |
| location.accuracy_meters | float | No | GPS accuracy in meters |
| metadata.capture_timestamp | string | No | ISO 8601 timestamp |
| metadata.device_model | string | No | Mobile device model |
| metadata.os_version | string | No | OS version |

**Success Response (In-Distribution):**
```json
{
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-03-15T14:30:02.341Z",
  "crop": {
    "predicted": "tomato",
    "confidence": 0.987
  },
  "disease": {
    "class_index": 1,
    "name": "early_blight",
    "confidence": 0.943,
    "description": "Alternaria solani infection showing characteristic concentric rings"
  },
  "ood_analysis": {
    "is_ood": false,
    "mahalanobis_distance": 8.5,
    "threshold": 12.3,
    "ood_score": 0.69,
    "dynamic_threshold_applied": true
  },
  "recommendations": {
    "immediate_actions": ["Remove infected leaves", "Apply copper-based fungicide"],
    "prevention": ["Ensure proper spacing", "Avoid overhead irrigation"],
    "expert_consultation": false
  },
  "model_info": {
    "adapter_version": "tomato-phase3-v1",
    "ood_stats_version": "2026-03-10",
    "inference_time_ms": 187
  }
}
```

**OOD Response (New Disease Candidate):**
```json
{
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440001",
  "timestamp": "2026-03-15T14:35:10.123Z",
  "crop": {
    "predicted": "tomato",
    "confidence": 0.991
  },
  "disease": null,
  "ood_analysis": {
    "is_ood": true,
    "ood_type": "NEW_DISEASE_CANDIDATE",
    "mahalanobis_distance": 28.7,
    "threshold": 12.3,
    "ood_score": 2.33,
    "nearest_class": "late_blight",
    "nearest_distance": 24.1,
    "confidence": 0.95
  },
  "recommendations": {
    "immediate_actions": ["Isolate plant", "Document symptoms with photos"],
    "expert_consultation": true,
    "message": "Potential new disease pattern detected. Sample queued for expert review."
  },
  "follow_up": {
    "sample_stored": true,
    "sample_id": "sample-uuid-for-reference",
    "estimated_label_time": "24-48 hours",
    "notification_enabled": true
  }
}
```

**Error Responses:**

**400 Bad Request:**
```json
{
  "detail": "Image field is required"
}
```

**422 Unprocessable Entity:**
```json
{
  "detail": "Invalid base64 image data"
}
```

**503 Service Unavailable:**
```json
{
  "detail": "Service not initialized"
}
```

---

### Submit Expert Label

**POST** `/feedback/expert-label`

Submit expert label for OOD samples to improve the model.

**Request Body:**
```json
{
  "sample_id": "sample-uuid-from-ood-response",
  "true_label": "septoria_leaf_spot",
  "confidence": 0.95,
  "notes": "Confirmed by plant pathologist Dr. Smith"
}
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Expert label received and queued for processing",
  "sample_id": "sample-uuid-from-ood-response"
}
```

**Workflow:**
1. OOD samples are automatically stored when detected
2. Expert reviews and submits true label via this endpoint
3. System queues for Phase 2 training (if new disease)
4. Adapter is retrained with new class
5. New version deployed automatically

---

### Batch Submit Feedback

**POST** `/feedback/batch`

Submit multiple expert labels at once.

**Request Body:**
```json
[
  {
    "sample_id": "uuid-1",
    "true_label": "healthy",
    "confidence": 0.98
  },
  {
    "sample_id": "uuid-2",
    "true_label": "early_blight",
    "confidence": 0.92
  }
]
```

**Response:**
```json
{
  "status": "success",
  "processed": 2,
  "results": [
    {"sample_id": "uuid-1", "status": "accepted"},
    {"sample_id": "uuid-2", "status": "accepted"}
  ]
}
```

---

### Get System Info

**GET** `/system/info`

Get system configuration and version information.

**Response:**
```json
{
  "version": "5.5.0",
  "architecture": "independent_multicrop_dynamic_ood",
  "crops": ["tomato", "pepper", "corn"],
  "router_loaded": true,
  "adapters_loaded": ["tomato", "pepper"],
  "device": "cuda",
  "targets": {
    "crop_routing_accuracy": 0.98,
    "phase1_accuracy": 0.95,
    "phase2_retention": 0.90,
    "phase3_retention": 0.85,
    "ood_auroc": 0.92
  }
}
```

---

## Error Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 400 | Bad Request | Missing image field, invalid base64 |
| 404 | Not Found | Crop adapter not registered |
| 422 | Unprocessable Entity | Corrupt image data |
| 500 | Internal Server Error | Model inference failure |
| 503 | Service Unavailable | Service not initialized, GPU down |

## Rate Limiting

- Default: 100 requests per minute per IP
- Burst: 10 requests per second
- Headers:
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset time in seconds

## Best Practices

### Client-Side
1. **Compress images** to <1MB before sending
2. **Resize** to 224×224 on device to reduce bandwidth
3. **Implement retry** with exponential backoff (max 3 retries)
4. **Cache** adapter status to avoid unnecessary calls
5. **Use crop_hint** when user selects crop manually

### Error Handling
```python
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        # Crop not available, inform user
        pass
    elif e.response.status_code == 422:
        # Invalid image, ask user to retake
        pass
except requests.exceptions.ConnectionError:
    # Network error, queue for later
    pass
```

## Mobile Integration

### Android (Kotlin)
See `mobile/android/` for complete implementation:
- `AADSService.kt`: Retrofit interface
- `DiagnosisRequest.kt`: Request model
- `DiagnosisResponse.kt`: Response model
- `AADSApplication.kt`: DI container

### iOS (Swift)
Not yet implemented - planned for v5.6

## Testing

### Sandbox Environment
- URL: `https://staging-api.uyumsoft.com/v1`
- Test crops: `test_tomato`, `test_pepper`
- No rate limits

### Sample Test Image
Base64-encoded test images available in `tests/fixtures/sample_images/`

---

**API Version:** 1.0
**Specification:** OpenAPI 3.0
**Last Updated:** February 2026