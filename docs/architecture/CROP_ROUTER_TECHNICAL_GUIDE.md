# Crop Router - Technical Implementation Guide

**Version**: AADS-ULoRA v5.5  
**Date**: February 2026  
**Target Audience**: Developers, ML Engineers, System Architects

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Simple Crop Router](#simple-crop-router)
4. [Enhanced Crop Router](#enhanced-crop-router)
5. [Parameter Reference](#parameter-reference)
6. [Code Flow & Execution](#code-flow--execution)
7. [Training Procedures](#training-procedures)
8. [Inference & Routing](#inference--routing)
9. [Caching Strategy](#caching-strategy)
10. [Metrics & Monitoring](#metrics--monitoring)
11. [Integration with Pipeline](#integration-with-pipeline)

---

## Architecture Overview

### Design Principles

The crop router follows these key principles:

```
┌─────────────────────────────────────────────────────────────┐
│          Input Image (Preprocessed Tensor)                  │
└─────────────────────────────────────────┬───────────────────┘
                                          │
                                          ↓
        ┌─────────────────────────────────────────────────────┐
        │   DINOv3 Backbone (Feature Extraction)              │
        │   - Frozen during inference                         │
        │   - Extracts global image features                  │
        │   - Output: Hidden state vectors                    │
        └─────────────────────────────────────┬───────────────┘
                                              │
                                              ↓
                        ┌─────────────────────────────────┐
                        │  Class-specific heads           │
                        ├─────────────────────────────────┤
                        │ Crop Classifier (Linear)        │
                        │ Part Classifier (Linear)        │
                        │ [Enhanced Router only]          │
                        └─────────────────────────────────┘
                                              │
                                              ↓
                        ┌─────────────────────────────────┐
                        │  Softmax → Confidence Scores    │
                        │  Argmax → Predicted Class       │
                        │  Top-K → Alternatives          │
                        └─────────────────────────────────┘
                                              │
                                              ↓
        ┌─────────────────────────────────────────────────────┐
        │   Confidence Filtering                              │
        │   - Compare against threshold                       │
        │   - Accept or Reject prediction                     │
        │   - Cache result                                    │
        └─────────────────────────────────────┬───────────────┘
                                              │
                                              ↓
                        ┌─────────────────────────────────┐
                        │  Route to Specialist Adapter    │
                        └─────────────────────────────────┘
```

### Key Differences: Simple vs Enhanced

| Aspect | Simple Router | Enhanced Router |
|--------|---------------|-----------------|
| **Outputs** | Crop only | Crop + Part |
| **Classification Heads** | 1 (crops) | 2 (crops + parts) |
| **Use Case** | Fast crop identification | Detailed plant analysis |
| **Accuracy** | 98%+ | 96%+ |
| **Latency** | ~100ms | ~120ms |
| **Adapter Routing** | By crop | By (crop, part) tuple |

---

## Core Components

### 1. Feature Extraction Backbone: DINOv3

**Purpose**: Extract semantic features from raw images

**Implementation**:
```python
# From both routers
self.backbone = AutoModel.from_pretrained('facebook/dinov3-base')
self.config = AutoConfig.from_pretrained('facebook/dinov3-base')
```

**Key Details**:
- **Model**: DINOv3 (Self-Supervised Vision Transformer)
- **Input Shape**: `(batch_size, 3, 224, 224)` - Preprocessed RGB images
- **Output**: Hidden state with shape `(batch_size, num_tokens, hidden_size)`
- **Hidden Size**: Extracted from config
  - DINOv3-Base: 768 dimensions
  - DINOv3-Large: 1024 dimensions
  - DINOv3-Giant: 1536 dimensions

**Frozen During Inference**: ✅ **YES**
- The backbone weights are NOT updated during router training
- Only lightweight classification heads are trained (Linear layers)
- This is called "Linear Probe" fine-tuning

**Feature Pooling Strategy**:
```python
outputs = self.backbone(images)
pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token
```
- Takes the first token (CLS token) from the transformer output
- Shape: `(batch_size, hidden_size)`
- Standard approach in vision transformers

### 2. Classification Heads

#### Simple Router - Single Classification Head

```python
class SimpleCropRouter:
    def __init__(self, crops: List[str], ...):
        # Single classification head
        self.classifier = nn.Linear(self.hidden_size, len(crops))
        self.classifier = self.classifier.to(self.device)
```

**Architecture**:
```
Input: (batch_size, hidden_size)
  → Linear layer: hidden_size → num_crops
Output: (batch_size, num_crops)
  → Softmax → Probabilities
  → Argmax → Crop index
```

**Parameters**:
- Weight matrix: `(num_crops, hidden_size)` - **TRAINABLE**
- Bias: `(num_crops,)` - **TRAINABLE**
- Total trainable parameters: `num_crops * hidden_size + num_crops`

**Example** (3 crops, 768 hidden size):
```
Parameters = 3 × 768 + 3 = 2,307 (very lightweight!)
```

#### Enhanced Router - Dual Classification Heads

```python
class EnhancedCropRouter:
    def _init_routing_models(self, config):
        # Two separate classification heads
        self.crop_classifier = nn.Linear(self.hidden_size, len(self.crops))
        self.part_classifier = nn.Linear(self.hidden_size, len(self.parts))
```

**Architecture**:
```
Input: (batch_size, hidden_size)
  └─→ Crop Classifier: Linear(768 → num_crops)
  │   Output: (batch_size, num_crops)
  │   → Softmax → Crop probabilities
  │
  └─→ Part Classifier: Linear(768 → num_parts)
      Output: (batch_size, num_parts)
      → Softmax → Part probabilities
```

**Dual Routing**:
```
Combined Output: (crop_name, part_name, crop_conf, part_conf)
Adapter Key: (crop, part)  # e.g., ('tomato', 'leaf')
```

---

## Simple Crop Router

### Class Definition & Initialization

```python
class SimpleCropRouter:
    def __init__(
        self,
        crops: List[str],
        model_name: str = 'facebook/dinov3-base',
        device: str = 'cuda',
        confidence_threshold: float = 0.92,
        top_k_alternatives: int = 3
    ):
```

### Parameter Definitions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crops` | `List[str]` | - | List of supported crop names (e.g., `['tomato', 'pepper', 'corn']`) |
| `model_name` | `str` | `'facebook/dinov3-base'` | HuggingFace model identifier for backbone |
| `device` | `str` | `'cuda'` | Compute device (`'cuda'` for GPU, `'cpu'` for CPU) |
| `confidence_threshold` | `float` | `0.92` | Minimum confidence (0.0-1.0) to accept prediction |
| `top_k_alternatives` | `int` | `3` | Number of alternative predictions to show on rejection |

### Internal State Variables

```python
self.image_cache = {}                           # Dict[str, Tuple]
self.cache_hits = 0                             # int
self.cache_misses = 0                           # int
self.cache_ttl_seconds = None                   # Optional[int]

self.confidence_stats = {
    'total_predictions': 0,                     # Total predictions made
    'high_confidence': 0,                       # Accepted predictions
    'low_confidence': 0,                        # Below threshold
    'rejected_predictions': 0,                  # Same as low_confidence
    'fallback_used': 0,                         # Fallback triggered
    'confidence_sum': 0.0                       # Sum of all confidences
}
```

### Training Method

```python
def train(
    self,
    train_dataset: 'CropDataset',
    val_dataset: 'CropDataset',
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: Optional[str] = None
) -> Dict[str, float]:
```

**Parameter Details**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_dataset` | `CropDataset` | - | Training dataset (must have `__len__`, `__getitem__`) |
| `val_dataset` | `CropDataset` | - | Validation dataset for early stopping |
| `epochs` | `int` | 10 | Number of complete passes through training data |
| `batch_size` | `int` | 32 | Samples per gradient update |
| `learning_rate` | `float` | 1e-3 | Optimizer step size (AdamW) |
| `save_path` | `Optional[str]` | None | Path to save best model weights |

**Training Loop Logic**:

```python
for epoch in range(epochs):
    # Training phase
    self.backbone.eval()              # Freeze backbone
    self.classifier.train()           # Enable dropout, batch norm
    
    for images, labels in train_loader:
        # Forward pass
        with torch.no_grad():
            features = self.backbone(images)
            pooled = features.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled)
        loss = self.criterion(logits, labels)  # CrossEntropyLoss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).sum() / len(labels)
    
    # Validation phase  
    val_metrics = self.validate(val_loader)
    
    # Save best model
    if val_metrics['accuracy'] > best_val_accuracy:
        self.save_model(save_path)
```

**Cost Function**: `CrossEntropyLoss`
```
L = -Σ target[i] * log(softmax(logits[i]))
```

**Optimizer**: `AdamW` (Adam + Weight Decay)
```python
optimizer = torch.optim.AdamW(
    self.classifier.parameters(),
    lr=learning_rate,
    weight_decay=0.01  # Default L2 regularization
)
```

### Validation Method

```python
def validate(self, val_loader: DataLoader) -> Dict[str, float]:
    """
    Returns:
        {
            'loss': float,      # Average cross-entropy loss
            'accuracy': float   # Fraction correct predictions
        }
    """
```

**Execution**:
- Backbone: **eval mode** (frozen)
- Classifier: **eval mode** (no dropout/batch norm updates)
- Forward pass: features → logits → probabilities
- Metric: Accuracy = (predictions == labels).sum() / total

---

## Enhanced Crop Router

### Dual Classification Architecture

```python
class EnhancedCropRouter:
    def __init__(
        self,
        crops: List[str],
        parts: List[str],
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
```

### Parameter Definitions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crops` | `List[str]` | - | Supported crop names |
| `parts` | `List[str]` | - | Supported plant parts (leaf, stem, fruit, etc.) |
| `config` | `Dict` | - | Configuration dict with `crop_router` sub-dict |
| `device` | `str` | `'cuda'` | Compute device |

### Adapter Registry

```python
self.adapters: Dict[Tuple[str, str], 'IndependentCropAdapter'] = {}
# Key: (crop_name, part_name)
# Example: ('tomato', 'leaf') → tomato_leaf_adapter_instance
```

### Dual Classification Heads

**Crop Classifier**:
```python
self.crop_classifier = nn.Linear(hidden_size, len(crops))
```

**Part Classifier**:
```python
self.part_classifier = nn.Linear(hidden_size, len(parts))
```

Both independently classify from the same feature vector (CLS token).

### Router Registration

```python
def register_adapter(
    self,
    crop: str,
    part: str,
    adapter_path: str,
    config: Optional[Dict] = None
) -> bool:
```

**Process**:
1. Validate crop exists in `self.crops`
2. Validate part exists in `self.parts`
3. Load adapter from disk
4. Register in `self.adapters` dict
5. Initialize metrics for this adapter

**Returns**: `True` if successful, `False` if validation fails

### Routing with Adapter Fallback

```python
def route(
    self,
    image: torch.Tensor,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[str, str, float, float, Dict[str, Any]]:
```

**Returns**:
```python
(
    'tomato',                    # Predicted crop
    'leaf',                      # Predicted part
    0.95,                        # Crop confidence
    0.87,                        # Part confidence
    {
        'status': 'success',
        'crop': 'tomato',
        'part': 'leaf',
        'adapter_key': ('tomato', 'leaf'),
        'scenario': 'diagnostic_scouting',
        'fallback_used': False
    }
)
```

**Fallback Logic**:
```
If adapter not available for (crop, part):
    → Try to find ANY adapter for crop
    → Use that part instead
    → Set fallback_used = True

If NO adapters exist for crop:
    → Return error status
    → status: 'error'
```

---

## Parameter Reference

### Model Backbone Options

Available DINOv3 models in HuggingFace:

| Model | Hidden Size | Parameters | Speed | Use Case |
|-------|------------|-----------|-------|----------|
| `facebook/dinov3-small` | 384 | 29M | ✅ Fast | Mobile/Edge |
| `facebook/dinov3-base` | 768 | 87M | ✅ Standard | Default choice |
| `facebook/dinov3-large` | 1024 | 300M | ⚠️ Slower | Higher accuracy |
| `facebook/dinov3-giant` | 1536 | 1.3B | ❌ Very slow | Not for routing |

### Confidence Threshold Tuning

```
Threshold = 0.85 → More permissive, more false positives
Threshold = 0.92 → Balanced (default)
Threshold = 0.98 → Very strict, many rejections
```

**Impact on Metrics**:
```
Higher Threshold:
  ✓ Fewer false positives
  ✓ Higher quality predictions
  ✗ More images rejected
  ✗ User frustration

Lower Threshold:
  ✓ Fewer rejections
  ✓ Better user experience
  ✗ More errors
  ✗ Garbage in → Garbage out for adapter
```

**Recommended Settings**:
- Production: 0.92 (balance accuracy & UX)
- Research: 0.85 (gather data, accept noise)
- Mission-critical: 0.95+ (strict accuracy required)

### Data Loader Configuration

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,          # Samples per gradient step
    shuffle=True,           # Random order
    num_workers=4           # Parallel data loading
)
```

**Batch Size Recommendations**:
```
GPU Memory: 8GB   → batch_size=16-32
GPU Memory: 16GB  → batch_size=32-64
GPU Memory: 24GB+ → batch_size=64-128
CPU only  → batch_size=4-8
```

---

## Code Flow & Execution

### Initialization Flow

```python
from src.router.simple_crop_router import SimpleCropRouter

# 1. Create router instance
router = SimpleCropRouter(
    crops=['tomato', 'pepper', 'corn'],
    model_name='facebook/dinov3-base',
    device='cuda',
    confidence_threshold=0.92,
    top_k_alternatives=3
)

# 2. What happens internally:
#    ✓ Load DINOv3 from HuggingFace (12.5 GB cache)
#    ✓ Extract hidden_size from config (768)
#    ✓ Create Linear layer: (768 → 3)
#    ✓ Move models to GPU
#    ✓ Initialize statistics tracking
#    ✓ Log initialization details
```

### Feature Extraction Flow

```python
def _extract_features(image: torch.Tensor) -> torch.Tensor:
    """
    Input:  image shape (1, 3, 224, 224)
    Output: features shape (1, 768)
    """
    # Run through frozen DINOv3
    with torch.no_grad():
        outputs = self.backbone(image)  # Vision Transformer processing
        # outputs.last_hidden_state shape: (1, 197, 768)  # 196 patches + 1 CLS
        
        # Extract CLS token (global representation)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 768)
    
    return features
```

**Why CLS Token?**
- In Vision Transformers, the CLS token aggregates information from all patches
- Similar to BERT's strategy in NLP
- Provides a holistic image understanding

### Prediction Flow

```python
import torch

# Prepare image
image = torch.randn(1, 3, 224, 224)  # Simulated preprocessed image

# Route image
try:
    crop, confidence = router.route(image)
    print(f"✓ Routed to: {crop} (confidence: {confidence:.2%})")
    
except ValueError as e:
    # Low confidence - extract alternatives
    print(f"✗ Confidence too low: {e}")
    # Error message includes top-K alternatives
```

**Internal Logic**:

```python
def route(self, image: torch.Tensor) -> Tuple[str, float]:
    cache_key = self._generate_cache_key(image)
    
    # Check cache
    if cache_key in self.image_cache:
        self.cache_hits += 1
        return self.image_cache[cache_key]
    
    self.cache_misses += 1
    
    # Feature extraction
    with torch.no_grad():
        outputs = self.backbone(image)
        features = outputs.last_hidden_state[:, 0, :]  # CLS pooling
        logits = self.classifier(features)              # Classification
        
        # Softmax for probabilities
        probs = torch.softmax(logits, dim=1)            # (1, 3)
        pred_idx = torch.argmax(probs, dim=1).item()   # scalar
        confidence = probs[0, pred_idx].item()          # 0.95
    
    # Confidence check
    if confidence < self.confidence_threshold:
        self.confidence_stats['rejected_predictions'] += 1
        raise ValueError(f"Confidence {confidence:.4f} below {self.confidence_threshold}")
    
    # Cache and return
    result = (self.crops[pred_idx], confidence)
    self.image_cache[cache_key] = result
    self.confidence_stats['high_confidence'] += 1
    
    return result
```

---

## Training Procedures

### Phase 1: Data Preparation

```python
from src.utils.data_loader import CropDataset

# Assuming CropDataset is implemented
train_dataset = CropDataset(
    root_dir='data/crops/train',
    crop_names=['tomato', 'pepper', 'corn'],
    transform=preprocessing_transforms
)

val_dataset = CropDataset(
    root_dir='data/crops/val',
    crop_names=['tomato', 'pepper', 'corn'],
    transform=preprocessing_transforms
)
```

**Required Dataset Structure**:
```
data/crops/train/
├── tomato/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── pepper/
│   └── ...
└── corn/
    └── ...
```

**CropDataset Expected Interface**:
```python
class CropDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        """Total number of samples"""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: torch.Tensor shape (3, 224, 224), values in [0, 1]
            label: int, 0 to num_crops-1
        """
```

### Phase 2: Training Execution

```python
router = SimpleCropRouter(crops=['tomato', 'pepper', 'corn'])

metrics = router.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    save_path='models/router_best.pth'
)

print(metrics)
# Output:
# {
#     'train_loss': 0.0234,
#     'train_accuracy': 0.9876,
#     'val_accuracy': 0.9814
# }
```

### Phase 3: Training Process Details

**For each epoch:**

```
1. Set backbone.eval() - Freeze features
2. Set classifier.train() - Enable training mode
3. For each batch:
   a. Forward: features = backbone(images)
   b. Classification: logits = classifier(features)
   c. Loss: loss = CrossEntropyLoss(logits, labels)
   d. Backward: loss.backward()
   e. Update: optimizer.step()
   f. Track: record loss and accuracy
4. Validation:
   a. Set backbone.eval(), classifier.eval()
   b. Run through val_loader (no gradients)
   c. Compute accuracy
5. Save best model if val_accuracy improves
```

**Convergence Monitoring**:

```python
# Typical training curve for crop classification:
Epoch 1:   Train Loss=2.8, Train Acc=0.45, Val Acc=0.52
Epoch 2:   Train Loss=1.2, Train Acc=0.75, Val Acc=0.78
Epoch 3:   Train Loss=0.6, Train Acc=0.88, Val Acc=0.90
Epoch 4:   Train Loss=0.3, Train Acc=0.95, Val Acc=0.94
Epoch 5:   Train Loss=0.2, Train Acc=0.97, Val Acc=0.96
...
Epoch 10:  Train Loss=0.02, Train Acc=0.99, Val Acc=0.98
```

**Typical Training Time**:
- Small dataset (1K images): ~5-10 minutes
- Medium dataset (10K images): ~30-60 minutes
- Large dataset (100K images): ~3-6 hours
- GPU: NVIDIA V100 or better recommended

---

## Inference & Routing

### Single Image Inference

```python
from src.utils.data_loader import preprocess_image

# Load and preprocess image
image_path = 'test_image.jpg'
image_tensor = preprocess_image(image_path)  # Returns (1, 3, 224, 224)

# Route
try:
    crop, confidence = router.route(image_tensor)
    print(f"Crop: {crop}, Confidence: {confidence:.2%}")
    # Output: Crop: tomato, Confidence: 94.23%
    
except ValueError as e:
    print(f"Could not route image: {e}")
```

### Batch Inference

```python
# Process multiple images
batch_images = torch.stack([
    preprocess_image(f'image_{i}.jpg')
    for i in range(10)
])

crops, confidences = router.route_batch(batch_images)

for crop, conf in zip(crops, confidences):
    print(f"{crop}: {conf:.2%}")
```

**Batch Processing Benefits**:
1. **Parallel computation** on GPU
2. **50-70% speedup** vs sequential processing
3. **Better GPU utilization**

### Enhanced Router Inference

```python
# Dual classification output
crop, part, crop_conf, part_conf, routing_info = \
    enhanced_router.route(image_tensor)

print(f"Crop: {crop} ({crop_conf:.2%})")
print(f"Part: {part} ({part_conf:.2%})")
print(f"Routing info: {routing_info}")

# Adapter retrieval
adapter = enhanced_router.get_adapter(crop, part)
if adapter:
    # Use adapter for disease detection
    prediction = adapter.predict(image_tensor)
```

---

## Caching Strategy

### Cache Key Generation

```python
def _generate_cache_key(self, image_tensor: torch.Tensor) -> str:
    """
    Generate MD5 hash of tensor bytes for caching.
    
    Purpose: Avoid reprocessing identical images
    
    Method:
    1. Convert tensor to byte representation
    2. Compute MD5 hash (32 hex characters)
    3. Use as dictionary key
    """
    tensor_bytes = image_tensor.cpu().numpy().tobytes()
    import hashlib
    return hashlib.md5(tensor_bytes).hexdigest()
```

**Example Hash**:
```
Image 1: tensor([[[...]]]) → "a4f8b2c1d9e4f7a3b6c2e1f0d9a4b6c7"
Image 1 (duplicate): same tensor → same hash → CACHE HIT ✓
Image 2: different tensor → "x8y2z1w4v7u3t6s9r2q5p8o1n4m7l0k"
```

### Cache Statistics

```python
# Access cache metrics
metrics = {
    'total_predictions': router.confidence_stats['total_predictions'],
    'cache_hits': router.cache_hits,
    'cache_misses': router.cache_misses,
    'hit_rate': router.cache_hits / (router.cache_hits + router.cache_misses)
}

print(f"Cache Hit Rate: {metrics['hit_rate']:.2%}")
# Output: Cache Hit Rate: 45.32%
```

### Cache Size Management

```python
# Implicit management - dictionary grows unbounded
self.image_cache = {}

# To limit cache size:
MAX_CACHE_SIZE = 1000

if len(self.image_cache) > MAX_CACHE_SIZE:
    # Clear oldest entries (FIFO)
    # Or clear entire cache
    self.image_cache.clear()
```

**Recommended Cache Configuration**:
```
Device              Max Cache Size      TTL
────────────────────────────────────────
GPU with 8GB RAM    500 images          None
GPU with 16GB RAM   2000 images         None
CPU                 100 images          300 seconds
```

---

## Metrics & Monitoring

### Confidence Statistics

```python
router.confidence_stats = {
    'total_predictions': 1000,
    'high_confidence': 950,        # Accepted (≥threshold)
    'low_confidence': 50,          # Rejected (<threshold)
    'rejected_predictions': 50,    # Same as low_confidence
    'fallback_used': 0,            # Only if fallback implemented
    'confidence_sum': 894.3        # Sum of all confidences
}

# Compute metrics
avg_confidence = router.confidence_stats['confidence_sum'] / \
                 router.confidence_stats['total_predictions']
acceptance_rate = router.confidence_stats['high_confidence'] / \
                  router.confidence_stats['total_predictions']

print(f"Average Confidence: {avg_confidence:.2%}")
print(f"Acceptance Rate: {acceptance_rate:.2%}")
# Output:
# Average Confidence: 89.43%
# Acceptance Rate: 95.00%
```

### Routing Metrics (Enhanced Router)

```python
metrics = enhanced_router.get_routing_metrics()

# Returns:
{
    'total_routes': 1000,
    'cache_hits': 450,
    'cache_misses': 550,
    'cache_hit_rate': 0.45,
    'adapter_hits': {
        ('tomato', 'leaf'): 300,
        ('tomato', 'fruit'): 150,
        ('pepper', 'leaf'): 250,
        ('corn', 'leaf'): 300
    },
    'adapter_hit_rates': {
        "('tomato', 'leaf')": 0.30,
        "('tomato', 'fruit')": 0.15,
        # ...
    },
    'registered_adapters': [
        ('tomato', 'leaf'),
        ('tomato', 'fruit'),
        ('pepper', 'leaf'),
        ('corn', 'leaf')
    ]
}
```

### Logging Configuration

```python
import logging

logger = logging.getLogger(__name__)

# Log levels in output
logger.info("CropRouter initialized on device: cuda")
logger.debug("Cache hit for image (key: a4f8b2c1...)")
logger.warning("Low confidence prediction: 0.78 < 0.92")
logger.error("Failed to register adapter: torch.cuda.OutOfMemoryError")
```

---

## Integration with Pipeline

### Pipeline Integration

```python
from src.pipeline.independent_multi_crop_pipeline import IndependentMultiCropPipeline
from src.router.simple_crop_router import SimpleCropRouter
from src.adapter.independent_crop_adapter import IndependentCropAdapter

# 1. Create pipeline
config = {
    'crops': ['tomato', 'pepper', 'corn'],
    'cache_enabled': True,
    'cache_size': 1000
}

pipeline = IndependentMultiCropPipeline(config=config, device='cuda')

# 2. Initialize router (train or load)
train_datasets = {...}  # Dict[crop_name, CropDataset]
val_datasets = {...}

pipeline.initialize_router(
    train_datasets=train_datasets,
    val_datasets=val_datasets
)

# 3. Register adapters for each crop+part combination
pipeline.register_adapter('tomato', None)  # adapter_path
pipeline.register_adapter('pepper', None)
pipeline.register_adapter('corn', None)

# 4. Full pipeline inference
results = pipeline.predict(image_tensor)
# {
#     'crop': 'tomato',
#     'disease': 'Early Blight',
#     'confidence': 0.94,
#     'router_confidence': 0.96
# }
```

### Router within Pipeline Context

```python
class IndependentMultiCropPipeline:
    def predict(self, image_tensor, ...) -> Dict:
        """
        1. Route image to crop adapter
        2. Run adapter to identify disease
        3. Return combined results
        """
        
        # Step 1: Route
        try:
            crop, confidence = self.router.route(image_tensor)
        except ValueError:
            return {'status': 'rejected', 'reason': 'low_confidence'}
        
        # Step 2: Get adapter
        adapter = self.router.get_adapter(crop)  # Simplified for SimpleCropRouter
        if not adapter:
            return {'status': 'error', 'reason': 'adapter_not_found'}
        
        # Step 3: Disease classification
        disease_pred = adapter.predict(image_tensor)
        
        # Step 4: Return results
        return {
            'crop': crop,
            'crop_confidence': confidence,
            'disease': disease_pred['disease'],
            'disease_confidence': disease_pred['confidence'],
            'status': 'success'
        }
```

---

## Common Issues & Solutions

### Issue 1: Low Cache Hit Rate

**Symptom**: Cache hit rate < 10%

**Causes**:
1. Limited repeated images in dataset
2. Data augmentation applied (same image → different tensor)
3. Preprocessing randomness

**Solutions**:
```python
# Keep original (non-augmented) images for caching
# Or disable augmentation during inference:
dataset_val = CropDataset(
    root_dir='data',
    transform=None  # No augmentation
)
```

### Issue 2: Low Confidence Predictions

**Symptom**: Many predictions rejected due to low confidence

**Debugging**:
```python
# Lower threshold temporarily
router.confidence_threshold = 0.85

# Analyze confidence distribution
import numpy as np
confidences = []

for image in test_images:
    try:
        _, conf = router.route(image)
        confidences.append(conf)
    except ValueError:
        pass

print(f"Mean confidence: {np.mean(confidences):.2%}")
print(f"Min confidence: {np.min(confidences):.2%}")
print(f"Max confidence: {np.max(confidences):.2%}")
```

### Issue 3: Memory Issues

**Symptom**: `CUDA out of memory` error

**Solutions**:
```python
# Reduce batch size
batch_size = 8  # Instead of 32

# Use CPU inference
router = SimpleCropRouter(..., device='cpu')

# Clear cache periodically
router.image_cache.clear()
```

---

## Summary Table

| Component | Purpose | Train? | Size |
|-----------|---------|--------|------|
| DINOv3 Backbone | Feature extraction | ❌ No | 87-1280 MB |
| Crop Classifier | Crop classification | ✅ Yes | 12-38 KB |
| Part Classifier | Part classification | ✅ Yes | 12-38 KB |
| Cache | Speed up inference | N/A | 50 MB-1 GB |

---

## References

- **DINOv3 Paper**: "Exploring Plain Vision Transformer Backbones for Object Detection"
- **HuggingFace**: https://huggingface.co/docs/transformers/
- **PyTorch**: https://pytorch.org/docs/

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Maintainance**: Agricultural AI Team
