# Checkpoint System Guide

## Overview

The AADS-ULoRA auto-training pipeline includes an **automatic checkpoint system** that creates recovery points at every major training step. This enables seamless recovery from failures, saves computational resources by skipping completed phases, and provides clear visibility into training progress.

## Key Features

### ✅ Automatic Recovery
If training is interrupted due to connection loss, Colab timeout, or runtime errors:
- Completed phases are saved as checkpoints
- Re-running the notebook automatically detects and skips completed phases
- Training resumes from where it left off without redundant work

### ⏭️ Skip Completed Phases
Once a phase completes:
- Its checkpoint is automatically saved with timestamp
- Next run: that phase is detected as done and execution is skipped
- Saves hours of compute time on Colab GPUs

### 🔍 Clear Progress Visibility
After each run, see exactly:
- Which phases completed ✅
- Which phases are pending ⊘
- When each checkpoint was created
- Total runtime for each phase

### 📝 Configuration Logging
Each checkpoint stores:
- Training configuration (crops, batch size, learning rate, etc.)
- Phase-specific parameters used
- Completion timestamp
- Status details

## Checkpoint Stages

The pipeline creates checkpoints at these major steps:

1. **Setup** - Environment configuration, GPU detection, Drive mounting
2. **Data Prep** - Dataset verification and preparation
3. **Phase 1** - DoRA (Difference of Rectified Activations) training
4. **Phase 2** - SD-LoRA (Stable Diffusion LoRA) training
5. **Phase 3** - CoNeC-LoRA (Congruent Enhancement LoRA) training
6. **Validation** - Model validation and metric evaluation
7. **Monitoring** - Performance reports generation

## Checkpoint Location

Checkpoints are stored in a hidden `.checkpoints` directory relative to your output directory:

```
your_output_directory/
├── model_checkpoints/
├── logs/
└── .checkpoints/           ← Checkpoint tracking
    └── checkpoint_log.json  ← Master checkpoint file
```

The `checkpoint_log.json` file records completion status and timestamps for all stages.

## Using the Checkpoint System

### First Run - All Phases Execute
```
📊 CHECKPOINT STATUS:
============================================================
setup        ✅ COMPLETED     (2025-02-20T10:15:23.456789)
data_prep    ✅ COMPLETED     (2025-02-20T10:16:45.123456)
phase1       ✅ COMPLETED     (2025-02-20T10:45:23.789012)
phase2       ✅ COMPLETED     (2025-02-20T11:30:15.345678)
phase3       ⊘ PENDING
validation   ⊘ PENDING
monitoring   ⊘ PENDING
```

### Second Run - Completed Phases Skip
```
✅ Phase 1 already completed at 2025-02-20T10:45:23.456789
   Skipping Phase 1 (checkpoint found)

✅ Phase 2 already completed at 2025-02-20T11:30:15.123456
   Skipping Phase 2 (checkpoint found)

🔴 PHASE 3: CoNeC-LoRA TRAINING
============================================================
Training Phase 3 (Congruent Enhancement LoRA)...
```

### Partial Re-training
If you modify configuration (e.g., change learning rate) and re-run:
- Setup, Data Prep, and completed phases skip automatically
- Only affected phases (e.g., Phase 1) re-run with new parameters
- Other phases remain skipped

## Checkpoint Management Commands

After the **Setup cell** runs, use these commands to manage checkpoints:

### Check Checkpoint Status
```python
checkpoint_manager.display_checkpoint_status()
```
Shows a table of all checkpoint states and timestamps.

### Get Details of a Checkpoint
```python
cp = checkpoint_manager.get_checkpoint('phase1')
print(cp)
# Output: {'timestamp': '...', 'completed': True, 'details': {...}}
```

### Check if Phase is Completed
```python
if checkpoint_manager.has_checkpoint('phase1'):
    print("Phase 1 is done!")
else:
    print("Phase 1 needs to run")
```

### Clear a Single Checkpoint
```python
checkpoint_manager.clear_checkpoints(['phase1'])
```
This forces Phase 1 to re-run on the next notebook execution.

### Clear Multiple Checkpoints
```python
checkpoint_manager.clear_checkpoints(['phase1', 'phase2', 'validation'])
```
Forces specific phases to re-run.

### Clear All Checkpoints
```python
checkpoint_manager.clear_checkpoints()
```
Resets entire training pipeline - all phases will re-run from scratch.

## Troubleshooting Scenarios

### Scenario 1: Training Interrupted at Phase 3
**What happens when you re-run:**
- Phases 1-2 detected as complete → skipped
- Phase 3 checkpoint not found → automatically restarts
- Saves hours compared to re-running everything

### Scenario 2: Changed Learning Rate Configuration
**To retrain with new learning rate:**
```python
# Option 1: Clear affected phases only
checkpoint_manager.clear_checkpoints(['phase1', 'phase2', 'phase3'])

# Option 2: Modify config and re-run
# (Completed phases still skip, only new phases run)
```

### Scenario 3: Validation Failed
**To retry validation:**
```python
checkpoint_manager.clear_checkpoints(['validation'])
# Then re-run the validation cell
```

### Scenario 4: Start Completely Fresh
**To reset everything:**
```python
checkpoint_manager.clear_checkpoints()
# Then re-run from the beginning
```

## Understanding Checkpoint Output

In the final summary, you'll see checkpoint information:

```
💾 CHECKPOINT STATUS:
============================================================
setup        ✅ COMPLETED     (2025-02-20T10:15:23)
data_prep    ✅ COMPLETED     (2025-02-20T10:16:45)
phase1       ✅ COMPLETED     (2025-02-20T10:45:23)
phase2       ✅ COMPLETED     (2025-02-20T11:30:15)
phase3       ✅ COMPLETED     (2025-02-20T12:20:10)
validation   ✅ COMPLETED     (2025-02-20T12:35:45)
monitoring   ✅ COMPLETED     (2025-02-20T12:50:30)

🚀 PHASES EXECUTED:
  ✓ Phase 1: DoRA Training
     ✅ Loaded from checkpoint (2025-02-20T10:45:23)
  ✓ Phase 2: SD-LoRA Training
     ✅ Loaded from checkpoint (2025-02-20T11:30:15)
  ✓ Phase 3: CoNeC-LoRA Training
  ✓ Validation & Testing

🔄 RESUMING FROM CHECKPOINTS:
  To resume training from checkpoints in future runs:
  1. Check the checkpoint log at: your_output_directory/.checkpoints/checkpoint_log.json
  2. Previously completed phases will be automatically skipped
  3. Modify configuration and re-run cells to update incomplete phases
  4. To force re-run a phase, clear its checkpoint
```

## Best Practices

### ✅ DO:
- **Leave checkpoints intact** after successful runs (enables quick resume)
- **Check checkpoint status** at the start of training to see what has completed
- **Modify configuration between runs** - only affected phases will re-execute
- **Use `clear_checkpoints()` carefully** - it forces re-computation

### ❌ DON'T:
- **Delete checkpoint files manually** - use the checkpoint manager instead
- **Assume phases reuse models** - each run generates fresh models (checkpoints track progress, not models)
- **Skip the Setup cell** - it initializes the checkpoint system
- **Ignore checkpoint errors** - they indicate phases that need attention

## FAQ

### Q: Does checkpoint manager store trained models?
**A:** No. Checkpoints track *progress* and *completion status*, not models. Models are saved separately in the models directory. Checkpoints make it quick to skip completed phases.

### Q: Can I resume with different crops?
**A:** Not recommended for Phases 1-3, as they train crop adapters. If you change crops, clear the phase checkpoints and re-train from scratch.

### Q: How much disk space do checkpoints use?
**A:** Minimal - checkpoints are just JSON files (~1KB each). The bulk of storage is models and logs.

### Q: What if checkpoint_log.json gets corrupted?
**A:** The system will recreate it automatically on next run. You may lose progress information but can reconstruct it by checking model timestamps.

### Q: Can I move the checkpoint directory?
**A:** Not recommended. Checkpoints are relative to your output directory. If you move the output directory, clear checkpoints and start fresh.

## Advanced: Inspecting Checkpoint Log

View the raw checkpoint log:

```python
import json
checkpoint_file = Path(TRAINING_CONFIG['checkpoint_directory']) / 'checkpoint_log.json'
with open(checkpoint_file) as f:
    log = json.load(f)
    print(json.dumps(log, indent=2))
```

This shows the exact timestamp and status of each phase for complete visibility into training history.

---

**Remember:** Checkpoints are your safety net for reliable, efficient training. They enable you to train large models without fear of losing progress to timeouts or interruptions.
