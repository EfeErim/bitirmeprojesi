# AADS-ULoRA v5.5 Detailed Implementation Plan

**Project:** Agricultural AI Development System - ULoRA v5.5  
**Architecture:** Independent Multi-Crop Continual Learning with Dynamic OOD Detection  
**Timeline:** 12 Weeks  
**Target:** Production-ready deployment for Uyumsoft ZiraiTakip

---

## 1. Executive Summary

AADS-ULoRA v5.5 implements a practical, production-oriented multi-crop disease detection system using independent crop adapters with enhanced dynamic OOD detection. The system achieves:

- **98%+** crop routing accuracy
- **95%+** Phase 1 clean accuracy per crop
- **90%+** retention on old classes during CIL
- **85%+** retention during DIL fortification
- **Dynamic per-class OOD thresholds** (no manual tuning)
- **Rehearsal-free** continual learning
- **Asynchronous** crop updates

---

## 2. System Architecture Overview

### 2.1 Two-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    L1: Crop Router                          │
│  (Simple classifier: DINOv2-base or ResNet-50)             │
│  Function: Route image → correct crop adapter              │
│  Target Accuracy: ≥98%                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌──────────────────┐              ┌──────────────────┐
│  L2: Tomato      │              │  L2: Pepper      │
│  Adapter         │              │  Adapter         │
│  (Independent)   │              │  (Independent)   │
│                  │              │                  │
│  Phases:         │              │  Phases:         │
│  • DoRA (Base)   │              │  • DoRA (Base)   │
│  • SD-LoRA (CIL) │              │  • SD-LoRA (CIL) │
│  • CONEC (DIL)   │              │  • CONEC (DIL)   │
│                  │              │                  │
│  OOD: Dynamic    │              │  OOD: Dynamic    │
│  Mahalanobis     │              │  Mahalanobis     │
└──────────────────┘              └──────────────────┘
```

**Key Principle:** Zero cross-adapter communication. Each crop maintains independent lifecycle.

### 2.2 Per-Crop Lifecycle

| Phase | Method | Purpose | Epochs | Target |
|-------|--------|---------|--------|--------|
| Phase 1 | DoRA | Base initialization | 50 | ≥95% accuracy |
| Phase 2 | SD-LoRA | Add new disease (CIL) | 20 | ≥90% retention |
| Phase 3 | CONEC-LoRA | Fortify with domain shift (DIL) | 15 | ≥85% protected retention |

---

## 3. Technical Specifications

### 3.1 Adapter Configuration (v5.5)

```python
adapter_spec_v55 = {
    'adapter_id': 'aads_multicrop_independent_v55',
    'architecture': 'independent_multicrop_dynamic_ood',

    # Crop Router Configuration
    'crop_router': {
        'type': 'resnet50_classifier',  # or 'dinov2_linear_probe'
        'training_data': 'plantclef_crops',
        'target_accuracy': 0.98
    },

    # Per-Crop Adapter Configuration
    'per_crop': {
        'model_name': 'facebook/dinov2-giant',
        'use_dora': True,
        'lora_r': 32,
        'lora_alpha': 32,
        'loraplus_lr_ratio': 16,  # B matrix LR = 16x base
        'phase1_epochs': 50,
        'phase2_epochs': 20,
        'phase3_epochs': 15
    },

    # OOD Detection (Dynamic Mahalanobis)
    'ood_detection': {
        'method': 'dynamic_mahalanobis',
        'threshold_factor': 2.0,  # k-sigma (95% confidence)
        'min_val_samples_per_class': 10,
        'fallback_threshold': 25.0
    },

    # Performance Targets
    'targets': {
        'crop_routing_accuracy': 0.98,
        'phase1_accuracy': 0.95,
        'phase2_retention': 0.90,
        'phase3_retention': 0.85,
        'ood_auroc': 0.92,
        'ood_false_positive_rate': 0.05
    }
}
```

### 3.2 Core Components

#### 3.2.1 Simple Crop Router

**File:** `src/router/simple_crop_router.py`

```python
class SimpleCropRouter:
    """
    Lightweight crop classifier using DINOv2 linear probe.
    Target: 98%+ crop classification accuracy
    """
    def __init__(self, crops: List[str], device='cuda'):
        self.crops = crops
        self.device = device
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-base')
        self.classifier = nn.Linear(768, len(crops))

    def train(self, crop_dataset, epochs=10, lr=1e-3):
        """Train crop classifier on labeled crop images."""
        pass

    def route(self, image: torch.Tensor) -> str:
        """Predict crop type from image."""
        pass
```

#### 3.2.2 Independent Crop Adapter

**File:** `src/adapter/independent_crop_adapter.py`

```python
class IndependentCropAdapter:
    """
    Self-contained adapter for one crop with dynamic OOD detection.
    No communication with other crop adapters.
    """
    def __init__(self, crop_name: str, device='cuda'):
        self.crop_name = crop_name
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-giant')
        self.adapter = None  # DoRA adapter
        self.classifier = None
        self.prototypes = None
        self.ood_stats = {'class_means': {}, 'class_stds': {}}

    def phase1_initialize(self, train_data, val_data, config):
        """Phase 1: DoRA base initialization + dynamic OOD thresholds."""
        pass

    def phase2_add_disease(self, new_disease_data, config):
        """Phase 2: SD-LoRA class-incremental learning."""
        pass

    def phase3_fortify(self, fortification_data, config):
        """Phase 3: CONEC-LoRA domain-incremental learning."""
        pass

    def detect_ood_dynamic(self, image):
        """OOD detection using dynamic per-class thresholds."""
        pass
```

#### 3.2.3 Pipeline Orchestrator

**File:** `src/pipeline/independent_multi_crop_pipeline.py`

```python
class IndependentMultiCropPipeline:
    """
    Main pipeline orchestrating router and independent adapters.
    Key: No cross-adapter communication - fully independent.
    """
    def __init__(self, config):
        self.router = SimpleCropRouter(crops=config['crops'])
        self.adapters = {}  # crop_name -> IndependentCropAdapter
        self.ood_buffers = {}  # Phase 2/3 triggering

    def register_crop(self, crop_name: str, adapter_path: str):
        """Register pre-trained crop adapter with OOD stats."""
        pass

    def process_image(self, image: torch.Tensor, metadata: dict = None):
        """
        Main inference flow:
        1. Router determines crop
        2. Crop adapter predicts disease with dynamic OOD
        3. OOD detection triggers updates if needed
        """
        pass
```

---

## 4. Implementation Phases (12-Week Timeline)

### Week 1-2: Environment Setup & Crop Router

**Deliverables:**
- [ ] Google Colab Pro environment configured
- [ ] Dependencies installed (transformers, peft, torch, etc.)
- [ ] Directory structure created
- [ ] Crop router training pipeline implemented
- [ ] Crop router validated ≥98% accuracy

**Tasks:**

1. **Environment Setup** (`setup_environment.py`)
   - Verify GPU (A100 preferred, minimum 45GB VRAM)
   - Install dependencies with version constraints
   - Mount Google Drive
   - Create directory structure:
     ```
     /content/drive/MyDrive/AADS_v55_Independent/
     ├── data/
     ├── adapters/
     ├── router/
     ├── checkpoints/
     ├── prototypes/
     ├── logs/
     └── ood_stats/
     ```

2. **Crop Router Implementation** (`src/router/`)
   - Implement `SimpleCropRouter` class
   - Create `CropClassificationDataset`
   - Implement training loop with augmentation
   - Validate on held-out set

3. **Data Preparation**
   - Acquire PlantCLEF crop dataset or create custom crop images
   - Ensure minimum 1000 images per crop type
   - Split: 70% train, 15% val, 15% test

**Success Criteria:**
- Crop router accuracy ≥98% on test set
- Inference time <50ms per image

---

### Week 3-4: Phase 1 - First Crop (Tomato) with DoRA

**Deliverables:**
- [ ] Tomato adapter Phase 1 trained
- [ ] Clean accuracy ≥95% on test set
- [ ] Mahalanobis prototypes computed
- [ ] Dynamic OOD thresholds computed for all classes
- [ ] OOD statistics saved

**Tasks:**

1. **Phase 1 Training** (`src/training/phase1_training.py`)
   - Implement DoRA configuration:
     ```python
     lora_config = LoraConfig(
         task_type=TaskType.FEATURE_EXTRACTION,
         r=32, lora_alpha=32, use_dora=True,
         target_modules=['query', 'value'],
         lora_dropout=0.1
     )
     ```
   - Apply DoRA to DINOv2-giant backbone
   - Add classifier head (1536 → num_classes)
   - Implement LoRA+ optimizer (16x LR for B matrices)

2. **Prototype Computation** (`src/ood/compute_prototypes.py`)
   - Collect features from training set per class
   - Compute class means and covariances
   - Add regularization (1e-4) to covariance

3. **Dynamic OOD Thresholds** (`src/ood/dynamic_thresholds.py`)
   - Run validation data through model
   - Compute Mahalanobis distance per sample to predicted class prototype
   - Calculate mean and std per class
   - Threshold = mean + 2.0 × std (95% confidence)

4. **Validation**
   - Clean accuracy on test set
   - Verify OOD stats for all classes (no zeros)
   - Check per-class thresholds are reasonable

**Success Criteria:**
- Phase 1 clean accuracy ≥95%
- All classes have ≥10 validation samples for stats
- OOD AUROC ≥0.92 on held-out OOD test set

---

### Week 5: Phase 2 - Add New Disease (SD-LoRA)

**Deliverables:**
- [ ] New disease (septoria_leaf_spot) added to tomato
- [ ] Old class retention ≥90%
- [ ] OOD thresholds updated for new class
- [ ] Phase 2 checkpoint saved

**Tasks:**

1. **SD-LoRA Implementation** (`src/training/phase2_sd_lora.py`)
   - Freeze lora_A and lora_B (directions)
   - Train only lora_magnitude and classifier
   - Expand classifier to accommodate new class
   - Copy old weights, initialize new class with Xavier

2. **Directional Freezing**
   ```python
   for name, param in self.adapter.named_parameters():
       if 'lora_A' in name or 'lora_B' in name:
           param.requires_grad = False  # Freeze directions
       elif 'lora_magnitude' in name:
           param.requires_grad = True   # Train magnitudes
   ```

3. **OOD Update for New Class**
   - Compute prototypes for new class from its training data
   - Run validation samples of new class through model
   - Compute Mahalanobis distances to new class prototype
   - Calculate mean/std for new class
   - Update `ood_stats` dictionary

4. **Retention Validation**
   - Evaluate on old class test set
   - Ensure ≥90% retention
   - If below, reduce LR (try 5e-5), increase epochs

**Success Criteria:**
- Old class retention ≥90%
- New class accuracy ≥85%
- OOD detection still functional (AUROC ≥0.90)

---

### Week 6: Phase 3 - Fortify Existing Classes (CONEC-LoRA)

**Deliverables:**
- [ ] Tomato fortified with domain-shifted data
- [ ] Protected class retention ≥85%
- [ ] OOD thresholds updated for fortified classes
- [ ] Phase 3 checkpoint saved

**Tasks:**

1. **CONEC-LoRA Structure** (`src/training/phase3_conec_lora.py`)
   - Freeze early transformer blocks (0-5, i.e., 6 blocks)
   - Add new LoRA to late blocks (6-11)
   - Use standard LoRA (not DoRA) for late blocks
   - Smaller rank (r=16) for task-specific adaptation

2. **Layer-wise Freezing**
   ```python
   shared_blocks = 6
   for i in range(shared_blocks):
       block = self.adapter.base_model.model.blocks[i]
       for param in block.parameters():
           param.requires_grad = False
   ```

3. **Fortification Training**
   - Train on domain-shifted data (different lighting, camera angles)
   - Monitor protected (non-fortified) class retention
   - Save best checkpoint based on protected retention

4. **OOD Update**
   - Re-compute prototypes for all classes (including fortified)
   - Re-run validation through updated model
   - Update OOD statistics for fortified classes

**Success Criteria:**
- Protected class retention ≥85%
- Fortified classes show improved robustness to domain shift
- No degradation in OOD detection

---

### Week 7-8: Second Crop (Pepper) - Independent

**Deliverables:**
- [ ] Pepper adapter Phase 1 trained independently
- [ ] Clean accuracy ≥95%
- [ ] Dynamic OOD thresholds computed
- [ ] Zero interference with tomato adapter verified

**Tasks:**

1. **Independent Phase 1** (same as Week 3-4 but for pepper)
   - Use separate data directory: `./data/pepper/phase1/`
   - Classes: `healthy, bacterial_spot, powdery_mildew`
   - No transfer from tomato (no LEBA, no ELLA)

2. **Independence Validation**
   - Update pepper adapter
   - Test tomato adapter on tomato test set
   - Verify tomato accuracy unchanged (within ±0.5%)
   - Document zero cross-crop interference

3. **OOD Statistics**
   - Compute per-class thresholds for pepper
   - Save to separate `ood_stats/pepper_ood_stats.pt`

**Success Criteria:**
- Pepper Phase 1 accuracy ≥95%
- Tomato adapter unaffected by pepper training
- Both adapters can be loaded simultaneously

---

### Week 9-10: Third Crop (Corn) & Integration Testing

**Deliverables:**
- [ ] Corn adapter Phase 1 trained
- [ ] Full pipeline with all three crops integrated
- [ ] End-to-end testing completed
- [ ] Performance targets met across all crops

**Tasks:**

1. **Corn Phase 1** (similar to pepper)
   - Classes: `healthy, corn_rust, northern_leaf_blight`

2. **Pipeline Integration** (`src/pipeline/`)
   - Load all three adapters
   - Test crop router routing accuracy
   - Verify dynamic OOD working per crop
   - Test Phase 2/3 trigger logic

3. **Comprehensive Testing**
   - Multi-crop test set (1000+ images across all crops)
   - Measure:
     - Crop routing accuracy
     - Per-crop disease accuracy
     - OOD detection AUROC
     - Inference latency (target <200ms)
     - Memory per adapter (target <25MB)

**Success Criteria:**
- Average multi-crop accuracy ≥93%
- Crop routing ≥98%
- All individual metrics meet targets

---

### Week 11: Mobile Integration & API Development

**Deliverables:**
- [ ] Cloud API implemented (FastAPI)
- [ ] Mobile SDK/API client documented
- [ ] Offline queue implementation
- [ ] Push notification system configured
- [ ] Gradio demo interface

**Tasks:**

1. **Cloud API** (`api/`)
   - Implement FastAPI endpoints:
     - `POST /v1/diagnose` (main inference)
     - `GET /v1/crops` (list supported crops)
     - `GET /v1/adapters/{crop}/status` (check updates)
     - `POST /v1/feedback/expert-label` (OOD labeling)
   - Add authentication (JWT)
   - Implement rate limiting
   - Add request logging

2. **Mobile SDK** (`mobile/`)
   - Android (Kotlin):
     - Retrofit API client
     - Room database for offline queue
     - WorkManager for background sync
     - CameraX integration
   - iOS (Swift):
     - Alamofire client
     - CoreData for offline queue
     - Background tasks
     - AVFoundation integration

3. **Offline Queue**
   - SQLite schema for pending diagnoses
   - Retry logic with exponential backoff
   - Sync scheduling (every 15 minutes when online)
   - Conflict resolution (server wins)

4. **Gradio Demo** (`demo/`)
   - Simple web interface for testing
   - Upload image, select crop (or auto-detect)
   - Display diagnosis with OOD info
   - Show confidence, Mahalanobis distance, threshold

**Success Criteria:**
- API latency <2s end-to-end
- Offline queue handles network failures gracefully
- Demo runs locally with sample adapters

---

### Week 12: Deployment, Documentation & Final Validation

**Deliverables:**
- [ ] Production deployment checklist completed
- [ ] All documentation finalized
- [ ] Monitoring dashboards set up
- [ ] Final validation report
- [ ] Demo presentation ready

**Tasks:**

1. **Deployment Architecture**
   - Cloud infrastructure (AWS/GCP):
     - GPU instances (A100 or equivalent)
     - Load balancer with health checks
     - Auto-scaling policies (target p95 <500ms)
     - PostgreSQL + Redis for metadata
     - S3/MinIO for sample storage
   - CI/CD pipeline (GitHub Actions / GitLab CI)

2. **Monitoring & Alerting**
   - Prometheus + Grafana dashboards:
     - API latency (p50, p95, p99)
     - Error rates (4xx, 5xx)
     - GPU utilization
     - Adapter version distribution
   - Alerting (PagerDuty):
     - API error rate >1%
     - GPU memory >90%
     - Adapter loading failures

3. **Documentation**
   - `README.md` (project overview, quick start)
   - `docs/architecture.md` (detailed architecture)
   - `docs/api_reference.md` (API endpoints)
   - `docs/mobile_integration.md` (mobile SDK guide)
   - `docs/deployment.md` (infrastructure setup)
   - `docs/troubleshooting.md` (common issues)

4. **Final Validation**
   - Run full test suite (unit, integration, field tests)
   - Load testing (simulate 100 concurrent users)
   - Security audit (TLS, certificate pinning, data encryption)
   - Compliance check (Turkish data sovereignty)

5. **Demo Preparation**
   - Record demo video showing:
     - Crop router in action
     - Disease diagnosis on healthy samples
     - OOD detection triggering Phase 2
     - Mobile app integration
   - Prepare slides for presentation

**Success Criteria:**
- All performance targets met
- Zero critical bugs in test report
- Documentation complete and reviewed
- Demo runs smoothly

---

## 5. File Structure

```
AADS_ULoRA_v5.5/
├── src/
│   ├── router/
│   │   ├── simple_crop_router.py
│   │   ├── dataset.py
│   │   └── train_router.py
│   ├── adapter/
│   │   ├── independent_crop_adapter.py
│   │   ├── dora_adapter.py
│   │   └── ood_detection.py
│   ├── training/
│   │   ├── phase1_training.py
│   │   ├── phase2_sd_lora.py
│   │   ├── phase3_conec_lora.py
│   │   └── optimizer_config.py
│   ├── ood/
│   │   ├── prototypes.py
│   │   ├── mahalanobis.py
│   │   └── dynamic_thresholds.py
│   ├── pipeline/
│   │   └── independent_multi_crop_pipeline.py
│   └── utils/
│       ├── data_loader.py
│       ├── metrics.py
│       └── checkpoint.py
├── api/
│   ├── main.py (FastAPI app)
│   ├── endpoints/
│   │   ├── diagnose.py
│   │   ├── crops.py
│   │   ├── adapters.py
│   │   └── feedback.py
│   ├── models/
│   │   ├── request.py
│   │   └── response.py
│   ├── middleware/
│   │   ├── auth.py
│   │   └── rate_limit.py
│   └── config.py
├── mobile/
│   ├── android/
│   │   ├── app/src/main/java/com/uyumsoft/ziraitakip/aads/
│   │   └── build.gradle
│   ├── ios/
│   │   ├── ZiraiTakip/
│   │   └── Podfile
│   └── shared/
│       ├── api_client.py (example)
│       └── offline_queue.py
├── demo/
│   ├── app.py (Gradio)
│   ├── examples/
│   └── static/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── field/
│   └── fixtures/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_phase1_training.ipynb
│   └── 03_ood_analysis.ipynb
├── config/
│   ├── adapter_spec_v55.json
│   ├── training_config.yaml
│   └── api_config.yaml
├── data/
│   ├── plantclef_crops/ (for router)
│   ├── tomato/
│   │   ├── phase1/
│   │   ├── val/
│   │   ├── test/
│   │   ├── septoria_leaf_spot/ (Phase 2)
│   │   └── domain_shift/ (Phase 3)
│   ├── pepper/
│   └── corn/
├── adapters/
│   ├── tomato/
│   │   ├── phase1_best.pth
│   │   ├── phase2_best.pth
│   │   ├── phase3_best.pth
│   │   └── ood_stats.pt
│   ├── pepper/
│   └── corn/
├── router/
│   └── crop_router_best.pth
├── prototypes/
│   ├── tomato_prototypes.pt
│   ├── pepper_prototypes.pt
│   └── corn_prototypes.pt
├── ood_stats/
│   ├── tomato_ood_stats.pt
│   ├── pepper_ood_stats.pt
│   └── corn_ood_stats.pt
├── logs/
│   ├── phase1_tomato.log
│   ├── phase2_tomato.log
│   └── phase3_tomato.log
├── checkpoints/
│   └── training_checkpoints/
├── requirements.txt
├── setup.py
├── README.md
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   ├── mobile_integration.md
│   ├── deployment.md
│   └── troubleshooting.md
└── run.sh (convenience script)
```

---

## 6. Dependencies

**`requirements.txt`:**

```
torch>=2.0.0
transformers==4.56.0
peft>=0.8.0
accelerate
bitsandbytes
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pillow>=10.0.0
opencv-python>=4.7.0
albumentations>=1.3.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=4.5.0
boto3>=1.28.0
gradio>=3.45.0
numpy>=1.24.0
pandas>=2.0.0
jupyter>=1.0.0
```

---

## 7. API Specification

### 7.1 POST /v1/diagnose

**Request:**
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

**Success Response (In-Distribution):**
```json
{
  "status": "success",
  "request_id": "uuid-v4-string",
  "timestamp": "2026-03-15T14:30:02.341Z",
  "crop": {
    "predicted": "tomato",
    "confidence": 0.987,
    "from_hint": false
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
    "adapter_version": "tomato-phase2-v3",
    "ood_stats_version": "2026-03-10",
    "inference_time_ms": 187
  }
}
```

**OOD Response (New Disease Candidate):**
```json
{
  "status": "success",
  "request_id": "uuid-v4-string",
  "crop": {"predicted": "tomato", "confidence": 0.991},
  "disease": {"class_index": null, "name": null, "confidence": 0.0},
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

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Coverage Targets:**
- Core logic: ≥90%
- Data processing: ≥85%
- API endpoints: ≥80%

**Test Categories:**
1. **Router Tests** (`tests/unit/test_router.py`)
   - Crop classification accuracy
   - Routing decision correctness

2. **Adapter Tests** (`tests/unit/test_adapter.py`)
   - DoRA weight decomposition
   - SD-LoRA freezing correctness
   - CONEC-LoRA layer freezing

3. **OOD Tests** (`tests/unit/test_ood.py`)
   - Mahalanobis distance computation
   - Dynamic threshold calculation
   - OOD decision boundary

4. **Pipeline Tests** (`tests/unit/test_pipeline.py`)
   - End-to-end routing + inference
   - Phase 2/3 trigger logic

### 8.2 Integration Tests

**Test Scenarios:**
1. **Full Pipeline** (`tests/integration/test_full_pipeline.py`)
   - Image → crop router → adapter → OOD → response
   - Multi-crop concurrent requests

2. **API Contract** (`tests/integration/test_api.py`)
   - Request/response schema validation
   - Error handling (400, 422, 503)
   - Authentication

3. **Mobile Sync** (`tests/integration/test_offline_queue.py`)
   - Offline queue persistence
   - Retry logic
   - Conflict resolution

### 8.3 Field Tests

**Real-World Validation:**
- Deploy to 3-5 farms in Turkey
- Collect 1000+ real images across crops
- Measure:
  - Accuracy in field conditions (varying light, angles)
  - Latency on 4G/5G networks
  - OOD detection on novel diseases
  - User satisfaction (survey)

---

## 9. Deployment Architecture

### 9.1 Cloud Infrastructure

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                    │
│              SSL termination, rate limiting                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ↓                               ↓
    ┌──────────────┐              ┌──────────────┐
    │  Router      │              │  Adapter      │
    │  Service     │              │  Service      │
    │  (CPU)       │              │  (GPU A100)   │
    └──────────────┘              └──────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ↓
                ┌──────────────────────┐
                │  OOD Detector        │
                │  (Dynamic Mahalanobis)│
                └──────────┬───────────┘
                           │
            ┌──────────────┴──────────────┐
            ↓                              ↓
    ┌──────────────┐              ┌──────────────┐
    │  Adapter      │              │  Sample       │
    │  Registry     │              │  Storage      │
    │  (PostgreSQL) │              │  (S3/MinIO)   │
    └──────────────┘              └──────────────┘
```

**Components:**

| Component | Technology | Specs |
|-----------|------------|-------|
| API Gateway | NGINX / AWS ALB | SSL, rate limiting, health checks |
| Router Service | FastAPI + DINOv2-base | 86M params, 50ms inference, CPU/GPU flexible |
| Adapter Service | FastAPI + DINOv2-giant | 1.1B params, 200ms inference, GPU required |
| OOD Detector | Custom module | Dynamic Mahalanobis, 10ms overhead |
| Adapter Registry | PostgreSQL + Redis | Metadata, OOD stats, caching |
| Sample Storage | AWS S3 / MinIO | Raw images, OOD candidates, training batches |
| Training Pipeline | PyTorch + PEFT | Async Phase 2/3, model versioning |

### 9.2 Mobile Integration

**Architecture:**
- Cloud-first: All heavy ML runs on GPU servers
- Offline-first: Queue-based architecture for poor connectivity
- Incremental updates: Adapters download in background
- Push notifications: FCM/APNs for OOD labeling completion

**Key Features:**
1. **Image Preprocessing** (on device)
   - Resize to 224×224
   - Normalize (ImageNet stats)
   - Quality check (blur detection)
   - Compress to JPEG (85% quality)

2. **Offline Queue**
   - SQLite persistence
   - Periodic sync (every 15 min when online)
   - Exponential backoff retry
   - Priority queue (OOD samples high priority)

3. **Push Notifications**
   - Adapter updates available
   - Phase 2/3 completion
   - Expert label ready

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU memory OOM | Medium | High | Use gradient checkpointing, reduce batch size, use 4-bit quantization |
| Phase 2 forgetting >10% | Low | High | Reduce LR, increase epochs, verify freezing correctness |
| OOD false positives | Medium | Medium | Tune threshold_factor (try 2.5), ensure sufficient validation data |
| Crop router accuracy <98% | Low | Medium | Use pre-trained PlantCLEF weights, augment data, increase training epochs |
| Mobile API latency >3s | Medium | High | Optimize image compression, use CDN, implement caching |
| Cross-crop interference | Low | Medium | Validate independence after each crop training, unit tests |

### 10.2 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient validation data for OOD | Medium | High | Require minimum 10 samples per class, use fallback threshold |
| Class imbalance in crops | Medium | Medium | Oversample minority crops, weighted loss |
| Poor quality images from field | High | Medium | Implement blur detection, quality rejection in mobile app |
| Label noise in training data | Low | Medium | Manual review of samples, clean data before training |

### 10.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Cloud infrastructure cost overrun | Medium | Medium | Use spot instances, auto-scaling, monitor usage |
| Mobile app adoption low | Low | High | User testing, feedback collection, iterative improvements |
| Expert labeling bottleneck | Medium | High | Prioritize high OOD score samples, batch labeling, train multiple experts |
| Model drift over time | Medium | High | Monitor performance metrics, schedule periodic retraining |

---

## 11. Monitoring & Maintenance

### 11.1 Key Metrics

**System Health:**
- API latency (p50, p95, p99) - target <2s
- Error rate (4xx, 5xx) - target <1%
- GPU utilization - target 70-90%
- Memory per adapter - target <25MB

**Model Performance:**
- Crop routing accuracy - target ≥98%
- Per-crop disease accuracy - target ≥93% average
- OOD detection AUROC - target ≥0.92
- False positive rate - target ≤5%
- Retention metrics (Phase 2 ≥90%, Phase 3 ≥85%)

**Business Metrics:**
- Daily diagnoses count
- OOD detection rate (should be 1-5%)
- Phase 2 trigger frequency
- User satisfaction score (survey)

### 11.2 Alerting

**Critical (P0):**
- API error rate >5% for 5 minutes
- GPU memory >95% for 10 minutes
- Adapter service down
- Database connection failure

**Warning (P1):**
- API latency p95 >3s for 15 minutes
- Crop routing accuracy <95% (computed daily)
- OOD false positive rate >10%
- Storage capacity >80%

**Info (P2):**
- New adapter version deployed
- Phase 2 training completed
- High OOD score samples accumulated

### 11.3 Maintenance Tasks

**Daily:**
- Check error logs
- Monitor API latency and error rates
- Review OOD detection statistics

**Weekly:**
- Analyze model performance drift
- Review expert labeling queue
- Check storage growth

**Monthly:**
- Retrain adapters with newly labeled OOD samples (if Phase 2 triggered)
- Update crop router with new crop types (if needed)
- Security patches and dependency updates
- Backup verification

**Quarterly:**
- Full system audit
- Performance optimization review
- User feedback analysis
- Roadmap planning for new crops/features

---

## 12. Success Criteria & Validation

### 12.1 Quantitative Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Crop routing accuracy | ≥98% | Test set of 1000+ crop images |
| Phase 1 clean accuracy | ≥95% | Per-crop test set |
| Phase 2 old class retention | ≥90% | Old class test set after CIL |
| Phase 3 protected retention | ≥85% | Non-fortified class test set after DIL |
| Average multi-crop accuracy | ≥93% | Weighted average across all crops |
| OOD detection AUROC | ≥0.92 | Held-out OOD test set |
| OOD false positive rate | ≤5% | In-distribution validation set |
| Inference latency | <200ms | GPU A100, batch size 1 |
| Memory per adapter | <25MB | Model checkpoint size |
| API end-to-end latency | <2s | Including network, preprocessing |
| Mobile offline queue reliability | 99.9% | Successful sync rate |

### 12.2 Qualitative Goals

- [ ] System is production-ready and deployable to Uyumsoft ZiraiTakip
- [ ] Documentation is comprehensive and clear
- [ ] Code is maintainable, modular, and well-tested
- [ ] Mobile integration seamless and user-friendly
- [ ] OOD detection reliably identifies novel disease patterns
- [ ] System can be extended to new crops with minimal effort
- [ ] Continuous learning loop operational (OOD → expert label → Phase 2)

### 12.3 Validation Checklist

**Pre-Deployment:**
- [ ] All unit tests passing (≥90% coverage)
- [ ] All integration tests passing
- [ ] Field test accuracy meets targets
- [ ] Load testing passed (100 concurrent users)
- [ ] Security audit passed
- [ ] Documentation reviewed and approved
- [ ] Deployment scripts validated in staging

**Post-Deployment (Week 1):**
- [ ] API health checks passing
- [ ] No critical errors in logs
- [ ] Mobile app successfully connecting
- [ ] First diagnoses completed successfully
- [ ] OOD samples being captured and stored

**Post-Deployment (Month 1):**
- [ ] Performance metrics stable
- [ ] User feedback positive (≥4/5 rating)
- [ ] OOD detection rate within 1-5%
- [ ] Expert labeling workflow functional
- [ ] No major bugs reported

---

## 13. Conclusion

This implementation plan provides a clear, actionable roadmap for building AADS-ULoRA v5.5 as a production-ready agricultural disease detection system. The 12-week timeline is realistic for a graduate-level project with dedicated effort.

**Key Differentiators:**
1. **Simplicity:** Independent adapters eliminate complex cross-crop coordination
2. **Robustness:** Dynamic OOD detection adapts to each disease's variability
3. **Practicality:** Rehearsal-free, asynchronous updates, cloud-based mobile integration
4. **Proven Methods:** DoRA, SD-LoRA, CONEC-LoRA are published, validated techniques

**Expected Outcomes:**
- A fully functional multi-crop disease detection system
- Mobile app integration with Uyumsoft ZiraiTakip
- Continuous learning capability from field data
- Production deployment ready for real-world agricultural use in Turkey

**Next Steps:**
1. Review and approve this implementation plan
2. Set up development environment (Week 1 tasks)
3. Begin crop router implementation
4. Establish regular progress reviews (weekly)

---

**Document Version:** 1.0  
**Date:** March 2026  
**Authors:** Agricultural AI Development Team  
**Review Status:** Ready for Implementation