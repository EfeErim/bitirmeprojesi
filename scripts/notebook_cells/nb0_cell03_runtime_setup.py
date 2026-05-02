# Auto-extracted from colab_notebooks/0_prepare_grouped_dataset_for_training.ipynb cell 3.
# Keep notebook execute-only cells thin; edit behavior here.

import json
import sys
import torch
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

from scripts.colab_repo_bootstrap import (
    export_current_colab_notebook,
    install_colab_requirements,
    mirror_checkpoint_state_to_repo,
    mirror_path_to_repo,
    push_repo_paths_to_github,
    resolve_github_token,
    running_in_colab,
)
from scripts.colab_live_telemetry import ColabLiveTelemetry
from src.core.config_manager import ConfigurationManager

# Install Colab requirements if running in Colab
IN_COLAB = running_in_colab()
if IN_COLAB:
    install_colab_requirements(ROOT / 'colab_notebooks' / 'requirements_colab.txt', IN_COLAB)

# Initialize telemetry and run directories
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
NOTEBOOK_FILENAME = '0_prepare_grouped_dataset_for_training.executed.ipynb'
REPO_RUN_DIR = ROOT / 'runs' / RUN_ID
REPO_NOTEBOOK_OUTPUT_PATH = REPO_RUN_DIR / 'notebooks' / NOTEBOOK_FILENAME
LOCAL_OUTPUT_DIR = ROOT / 'outputs' / 'colab_notebook_data_prep'
REPO_OUTPUT_DIR = REPO_RUN_DIR / 'outputs' / 'colab_notebook_data_prep'
REPO_TELEMETRY_DIR = REPO_RUN_DIR / 'telemetry'
REPO_CHECKPOINT_STATE_DIR = REPO_RUN_DIR / 'checkpoint_state'

TELEMETRY = ColabLiveTelemetry(
    notebook_name='0_prepare_grouped_dataset_for_training.ipynb',
    run_id=RUN_ID,
)
CHECKPOINT_ROOT = TELEMETRY.drive_run_dir if TELEMETRY.drive_run_dir.exists() else TELEMETRY.local_run_dir

# Configure device and settings
REQUESTED_DEVICE = str(ConfigurationManager(config_dir=str(ROOT / 'config'), environment='colab').load_all_configs().get('training', {}).get('continual', {}).get('device', 'cuda'))
DEVICE = REQUESTED_DEVICE if not REQUESTED_DEVICE.startswith('cuda') or torch.cuda.is_available() else 'cpu'
if DEVICE != REQUESTED_DEVICE:
    print(f"[PREP] Requested device {REQUESTED_DEVICE!r} is unavailable; using {DEVICE!r}.")

# Dataset and model configuration
EMBEDDING_BATCH_SIZE = 32 if DEVICE.startswith('cuda') else 8
NEIGHBORS = 4
PREP_DINOV3_MODEL_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
PREP_BIOCLIP_MODEL_ID = 'imageomics/bioclip-2.5-vith14'
PREP_ARTIFACT_ROOT = 'outputs/colab_notebook_data_prep/artifacts'
PREPARED_RUNTIME_ROOT = 'data/prepared_runtime_datasets'
OOD_DATASET_ROOT = 'data/ood_dataset'
OOD_DATASET_NAME = ''
OOD_ROOT = ''
ASK_FOR_OOD_ROOT = False
PREPARED_CLASS_ROOT = 'data/prepared_class_root_datasets'
PREPARE_DATASET_FROM_REPORTS = True
MATERIALIZE_AFTER_REVIEW = True
INTERACTIVE_AUDIT_REVIEW = True
MAX_INTERACTIVE_REVIEW_ITEMS = 25
UNDER_MIN_EVAL_POLICY = 'skip'
SAVE_RUNTIME_DATASET_TO_GITHUB = True
RUNTIME_DATASET_PUSH_REMOTE_NAME = 'origin'
RUNTIME_DATASET_PUSH_BRANCH = None
CLEANUP_SEED = 42

LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_NOTEBOOK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_run_outputs_to_repo() -> dict[str, str]:
    exports: dict[str, str] = {}
    mirrored_outputs = mirror_path_to_repo(LOCAL_OUTPUT_DIR, REPO_OUTPUT_DIR)
    if mirrored_outputs is not None:
        exports['outputs'] = str(mirrored_outputs)
    telemetry_source = TELEMETRY.drive_run_dir if TELEMETRY.drive_run_dir.exists() else TELEMETRY.local_run_dir
    mirrored_telemetry = mirror_path_to_repo(telemetry_source, REPO_TELEMETRY_DIR)
    if mirrored_telemetry is not None:
        exports['telemetry'] = str(mirrored_telemetry)
    mirrored_checkpoint_state = mirror_checkpoint_state_to_repo(CHECKPOINT_ROOT, REPO_CHECKPOINT_STATE_DIR)
    if mirrored_checkpoint_state is not None:
        exports['checkpoint_state'] = str(mirrored_checkpoint_state)
    return exports

TELEMETRY.configure_repo_output_export(
    output_dir=REPO_OUTPUT_DIR,
    notebook_filename=NOTEBOOK_FILENAME,
    export_notebook_fn=export_current_colab_notebook,
)
TELEMETRY.update_latest({'phase': 'bootstrap_ready', 'run_id': RUN_ID})
print(f'[BOOTSTRAP] run_id={RUN_ID}')
