# AADS-ULoRA Colab Cheatsheet

## Quick Commands

### Setup
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set workspace
import os
os.environ['AADS_WORKSPACE'] = '/content/drive/MyDrive/aads_ulora'
```

### Installation
```bash
# Run installer
!python scripts/install_colab.py

# Or install manually
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers peft accelerate datasets evaluate
!pip install numpy pandas pillow scikit-learn tqdm psutil
!pip install python-multipart
```

### GPU Info
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Monitor memory
!nvidia-smi
```

### Data Preparation
```python
# Download dataset
import gdown
gdown.download('https://drive.google.com/uc?id=FILE_ID', 'data.zip')

# Extract
import zipfile
with zipfile.ZipFile('data.zip', 'r') as f:
    f.extractall('data/')
```

### Training Commands

#### Phase 1
```python
from src.training.colab_phase1_training import ColabPhase1Trainer

trainer = ColabPhase1Trainer(
    model_name='facebook/dinov3-giant',  # Falls back to local stub if unavailable
    num_classes=5,
    lora_r=32,
    learning_rate=1e-4,
    batch_size=8,
    device='cuda'
)
trainer.setup_optimizer()
history = trainer.train(train_loader, val_loader, num_epochs=10)
trainer.save_adapter('./models/phase1')
```

#### Phase 2
```python
from src.training.colab_phase2_sd_lora import ColabPhase2Trainer

trainer = ColabPhase2Trainer(
    adapter_path='./models/phase1',
    lora_r=16,
    learning_rate=1e-4,
    batch_size=4,
    device='cuda'
)
trainer.setup_optimizer()
history = trainer.train(train_loader, val_loader, num_epochs=5)
trainer.save_adapter('./models/phase2')
```

#### Phase 3
```python
from src.training.colab_phase3_conec_lora import ColabPhase3Trainer, CoNeCConfig

config = CoNeCConfig(
    lora_r=8,
    learning_rate=5e-5,
    batch_size=16,
    temperature=0.07,
    prototype_dim=128,
    num_prototypes=10,
    contrastive_weight=0.1,
    orthogonal_weight=0.01,
    device='cuda'
)
trainer = ColabPhase3Trainer(config)
trainer.setup_optimizer()
history = trainer.train(train_loader, val_loader, num_epochs=10)
trainer.save_checkpoint('./models/phase3')
```

### DataLoaders
```python
from src.dataset.colab_dataloader import ColabDataLoader

# Simple kwargs interface (recommended)
loader = ColabDataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2  # Auto-retries with 0 if multiprocessing fails
)

# With config object
from src.dataset.colab_dataloader import DataLoaderConfig
config = DataLoaderConfig(
    batch_size=16,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=2
)
loader = ColabDataLoader(dataset, config=config)
```

### Datasets
```python
from src.dataset.colab_datasets import ColabCropDataset, ColabDomainShiftDataset
from torchvision import transforms

# Standard crop dataset (returns tuples)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ColabCropDataset(
    data_dir='./data/train',
    transform=transform
)

# Domain shift dataset for Phase 3 (returns dicts)
domain_dataset = ColabDomainShiftDataset(
    data_dir='./data/domain_a',
    transform=transform,
    domain_label=0
)
```

### Memory Management
```python
import torch
import gc

# Clear cache
torch.cuda.empty_cache()
gc.collect()

# Check memory
allocated = torch.cuda.memory_allocated() / 1024**3
cached = torch.cuda.memory_reserved() / 1024**3
print(f"Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
```

### Mixed Precision
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation
```python
accumulation_steps = 4
for i, (batch, labels) in enumerate(train_loader):
    loss = model(batch, labels)
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Checkpointing
```python
# Save
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1)
        # Calculate metrics
```

### OOD Detection
```python
# Get prototypes
prototypes = trainer.prototype_manager.get_prototypes()

# Compute distances
features = extract_pooled_output(model, images)
distances = torch.cdist(features, prototypes)
min_distances, nearest = distances.min(dim=1)

# Threshold
threshold = 2.0  # Adjust based on validation
ood_mask = min_distances > threshold
```

### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./logs/tensorboard')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/val', acc, epoch)
writer.close()
```

### Common Issues

#### OOM Error
```python
# Reduce batch size
batch_size = 4  # Instead of 16

# Increase gradient accumulation
gradient_accumulation_steps = 4

# Disable mixed precision
use_amp = False
```

#### Slow Training
```python
# Reduce image size
image_size = 128  # Instead of 224

# Reduce workers
num_workers = 1

# Use smaller model
model_name = 'facebook/dinov2-small'  # Instead of giant
```

#### Corrupted Data
```python
# Skip errors
from torchvision.datasets import ImageFolder
dataset = ImageFolder(root='data/', transform=transform, loader=loader_with_error_handling)
```

### File Paths
```python
workspace = Path('/content/drive/MyDrive/aads_ulora')
config_path = workspace / 'config' / 'colab.json'
data_dir = workspace / 'data'
models_dir = workspace / 'models'
checkpoints_dir = workspace / 'checkpoints'
logs_dir = workspace / 'logs'
outputs_dir = workspace / 'outputs'
```

### Environment Variables
```python
import os
os.environ['AADS_WORKSPACE'] = str(workspace)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'
```

### Colab-Specific
```python
# Get notebook name
import json
with open('/content/colab/Notebook.ipynb', 'r') as f:
    nb = json.load(f)
print(nb['metadata']['colab']['name'])

# Install ipywidgets for tqdm
!pip install ipywidgets
!jupyter nbextension enable --py widgetsnbextension

# Use tqdm notebook
from tqdm.notebook import tqdm
for i in tqdm(range(100)):
    pass
```

### Debugging
```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Print shapes
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {outputs.shape}")

# Check for NaN
if torch.isnan(loss):
    print("Loss is NaN!")
    breakpoint()
```

### Performance Tips
```python
# Pin memory for faster data transfer
pin_memory = True

# Use prefetching
prefetch_factor = 2

# Enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# Use channels last memory format
inputs = inputs.to(memory_format=torch.channels_last)
```

### Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

### Save Outputs
```python
# Save predictions
import pandas as pd
df = pd.DataFrame({'pred': preds, 'label': labels})
df.to_csv('predictions.csv', index=False)

# Save metrics
import json
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Load Config
```python
import json
with open('config/colab.json', 'r') as f:
    config = json.load(f)

batch_size = config['training']['phase1']['batch_size']
```

### Restart Runtime
```python
# Programmatically restart (use with caution)
import os
os.kill(os.getpid(), 9)
```

### Free Memory
```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

### Check Disk Space
```python
!df -h

# Or in Python
import shutil
total, used, free = shutil.disk_usage('/content')
print(f"Free: {free // (2**30)} GB")
```

### Download Files
```python
from google.colab import files

# Download single file
files.download('model.pth')

# Download directory
!zip -r models.zip models/
files.download('models.zip')
```

### Upload Files
```python
from google.colab import files

uploaded = files.upload()
for fn in uploaded.keys():
    print(f"Uploaded: {fn}")
```

## Common Imports
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path
import json
import logging
```

## Key Functions

### Extract Features
```python
def extract_pooled_output(model, images):
    """Extract pooled features from model."""
    outputs = model(images)
    if hasattr(outputs, 'pooler_output'):
        return outputs.pooler_output
    elif hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state.mean(dim=1)
    else:
        return outputs.mean(dim=[2, 3])  # For CNN features
```

### Compute Metrics
```python
def compute_metrics(preds, labels):
    """Compute classification metrics."""
    accuracy = (preds == labels).mean()
    return {'accuracy': accuracy}
```

### Set Seed
```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [Colab Documentation](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

## Emergency Stops

```python
# Interrupt training
from IPython.display import Javascript
Javascript('IPython.notebook.kernel.interrupt()')

# Clear output
from IPython.display import clear_output
clear_output()
```

## Quick Reference Card

| Task | Command |
|------|---------|
| Mount Drive | `drive.mount('/content/drive')` |
| Check GPU | `!nvidia-smi` |
| Clear Cache | `torch.cuda.empty_cache()` |
| Save Model | `torch.save(model.state_dict(), 'model.pth')` |
| Load Model | `model.load_state_dict(torch.load('model.pth'))` |
| Enable AMP | `from torch.cuda.amp import autocast, GradScaler` |
| Progress Bar | `from tqdm.notebook import tqdm` |
| Plot | `import matplotlib.pyplot as plt` |
| Config | `import json; config = json.load(open('config.json'))` |
| Logging | `logging.basicConfig(level=logging.INFO)` |