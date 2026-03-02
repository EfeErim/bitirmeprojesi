## Plan: Two-Notebook Colab Overhaul

**TL;DR** — Replace the current 6-notebook suite with two self-contained Colab notebooks: (1) a **Crop Router Pipeline** notebook for image upload → VLMPipeline inference with visual results, and (2) an **Interactive Adapter Training** notebook with ipywidgets-driven parameter input, dataset validation, live progress bars during training, and OOD calibration. This requires adding a progress callback to the training loop in `src/`, archiving old notebooks, and creating two new `.ipynb` files with embedded setup sections. Existing `src/` APIs are nearly ready — the main gaps are the lack of a training callback and a dict-wrapping collation step for `CropDataset`.

**Steps**

### Phase 0 — Archive & Housekeeping

1. Move all existing notebooks (colab_notebooks/0_AUTO_TRAIN_COMPLETE_PIPELINE.ipynb, 1_data_preparation.ipynb, 2_continual_sd_lora_training.ipynb, 5_testing_validation.ipynb, 6_performance_monitoring.ipynb) into colab_notebooks/archive/.
2. Archive colab_bootstrap.ipynb from root into the same archive folder.
3. Update colab_notebooks/README.md to reference the two new notebooks.

### Phase 1 — `src/` Changes (Minimal, Targeted)

4. **Add progress callback to `ContinualSDLoRATrainer.train_increment()`** in src/training/continual_sd_lora.py (line ~341). Add an optional `progress_callback: Optional[Callable[[dict], None]] = None` parameter. At the end of each batch, call `progress_callback({"epoch": e, "batch": b, "total_batches": N, "batch_loss": loss, "epoch_progress": b/N})`. At the end of each epoch, call with `{"epoch_done": e, "epoch_loss": avg_loss}`. This is backward-compatible (default `None` = no callback).

5. **Propagate callback through `IndependentCropAdapter.train_increment()`** in src/adapter/independent_crop_adapter.py (line ~133) — pass `progress_callback` kwarg down to the trainer.

6. **Add a dict-collating DataLoader helper** in src/utils/data_loader.py. The trainer's `training_step()` expects `{"images": Tensor, "labels": Tensor}` but `CropDataset.__getitem__` returns `(tensor, int)` tuples. Add a `dict_collate_fn(batch)` utility that converts the default `(images, labels)` tuple-batch into the expected dict format. Also add a convenience `create_training_loaders()` function that wraps `create_data_loaders()` and applies this collation.

### Phase 2 — Notebook 1: Crop Router Pipeline

7. Create **`colab_notebooks/1_crop_router_pipeline.ipynb`** with these cells:

   - **Cell 1 — Setup & Dependencies**: GPU detection, Google Drive mount (optional), pip install from requirements_colab.txt, sys.path configuration, CUDA/cuDNN settings. Consolidates logic from `colab_bootstrap.ipynb` cells §1–§3.

   - **Cell 2 — Load Config & Initialize Router**: Load config/base.json, optionally generate GPU-optimized overrides (from bootstrap §4 logic). Instantiate `VLMPipeline(config, device)` and call `load_models()`. Show a profile selector widget (dropdown: `fast`, `balanced`, `calibrated`, `leaf_fruit_production`) from the profiles defined in config/base.json, defaulting to `balanced`. Apply via `router.set_runtime_profile()`.

   - **Cell 3 — Upload Image**: Use `google.colab.files.upload()` (pattern from scripts/colab_test_upload.py) or alternatively an ipywidgets `FileUpload` widget. Load as PIL Image. Display the uploaded image inline via `matplotlib`.

   - **Cell 4 — Run Router & Display Results**: Call `router.analyze_image(image, confidence_threshold=threshold)`. Parse the returned detections list. For each detection, draw bounding boxes on the image using `matplotlib.patches.Rectangle`, annotated with crop label, part label, and confidence. Display a summary table (ipywidgets `HTML` or `pandas` DataFrame) with columns: Detection #, Crop, Part, Confidence, BBox.

   - **Cell 5 — Full Pipeline (Optional)**: If the user has trained adapters available, instantiate `IndependentMultiCropPipeline`, call `process_image()` to also run disease classification and OOD scoring. Display the complete result including diagnosis and OOD status.

   - **Cell 6 — Re-run Section**: A "Run Again" cell that loops back to upload + analyze, so the user can test multiple images without restarting the kernel.

### Phase 3 — Notebook 2: Interactive Adapter Training

8. Create **`colab_notebooks/2_interactive_adapter_training.ipynb`** with these cells:

   - **Cell 1 — Setup & Dependencies**: Same embedded setup as Notebook 1 (GPU detect, Drive mount, deps, sys.path).

   - **Cell 2 — Interactive Parameter Input**: An ipywidgets form panel with:
     - `Text` widget: Dataset root path (default: `data/class_root_dataset`)
     - `Text` widget: Crop name to train (e.g., `tomato`)
     - `IntSlider`: Number of epochs (range 1–50, default from config)
     - `IntSlider`: Batch size (range 1–64, default GPU-adaptive)
     - `FloatLogSlider`: Learning rate (range 1e-6 to 1e-2, default `1e-4`)
     - `IntSlider`: LoRA rank `r` (range 4–64, default `16`)
     - `IntSlider`: LoRA alpha (range 8–128, default `32`)
     - `FloatSlider`: LoRA dropout (range 0.0–0.5, default `0.1`)
     - `FloatSlider`: OOD threshold factor (range 1.0–5.0, default `2.0`)
     - `Button`: "Validate & Start Training"
     
     All defaults are pre-populated from config/base.json → `training.continual` section. Widgets update a shared config dict.

   - **Cell 3 — Dataset Validation**: On button click (or manual cell run), validate the dataset path using `evaluate_layout()` from scripts/evaluate_dataset_layout.py. Display: number of classes found, images per class, any structural warnings. Show class names discovered and let user confirm before proceeding.

   - **Cell 4 — Initialize Training Engine**: Build `ContinualSDLoRAConfig` from widget values. Create `IndependentCropAdapter`, call `initialize_engine(class_names=..., config=...)`. Print model summary: total params, trainable params, LoRA modules applied, fusion layer dimensions. Create data loaders via `create_training_loaders()` (the new dict-collating helper from Step 6).

   - **Cell 5 — Train with Live Progress**: Call `adapter.train_increment(train_loader, num_epochs=N, progress_callback=cb)`. The callback drives:
     - `ipywidgets.IntProgress` bar (0→total_batches per epoch)
     - `ipywidgets.HTML` panel showing: current epoch, batch, running loss, elapsed time, ETA
     - A matplotlib live-updating loss curve (epoch-level) using `IPython.display.clear_output` + re-plot
     
     After training completes, display final training history: per-epoch loss values, total training time, final loss.

   - **Cell 6 — OOD Calibration**: Run `adapter.calibrate_ood(val_loader)`. Display calibration results: number of classes calibrated, calibration version, per-class thresholds if available. Show a status indicator (ipywidgets `HTML` with green/red).

   - **Cell 7 — Save Adapter**: `adapter.save_adapter(checkpoint_dir)`. Print saved file listing (adapter weights, classifier, fusion, metadata JSON). Show the full path so user knows where to find it.

   - **Cell 8 — Validation (Optional)**: Run the saved adapter on the validation set, compute accuracy per class, display a confusion matrix plot and classification report. This reuses model components already in memory.

### Phase 4 — Documentation & Sync

9. Update colab_notebooks/README.md with a usage guide for the two new notebooks: prerequisites, expected Colab runtime, Google Drive structure.
10. Update docs/REPO_FILE_RELATIONS.md to reflect the new notebook structure.
11. Update scripts/README.md if any script references change.

### Verification

- **Static**: Run `scripts/validate_notebook_imports.py` against both new notebooks to confirm all imports resolve.
- **Unit**: Verify `ContinualSDLoRATrainer.train_increment(progress_callback=mock_fn)` calls the mock with correct keys, and that `progress_callback=None` still works identically to before.
- **Collation**: Test that `dict_collate_fn` correctly transforms `[(tensor, 0), (tensor, 1)]` → `{"images": stacked_tensor, "labels": tensor([0, 1])}`.
- **Manual Colab smoke test**: Open each notebook in Colab, run through the setup cells, confirm widgets render, upload an image in NB1, and verify the parameter form displays correctly in NB2.
- **Markdown links**: Run `scripts/check_markdown_links.py` to verify README updates.

### Decisions

- **Embedded setup over shared script**: Each notebook is self-contained so a user can open either one independently without running a prerequisite.
- **VLMPipeline (full) as the router**: Provides SAM3 bounding-box detection + BioCLIP classification with open-set rejection — matches the project's core capability.
- **ipywidgets over @param or input()**: Richer UI with sliders, dropdowns, and progress bars; already declared in `requirements_colab.txt`.
- **Progress callback added to src/**: Minimal, backward-compatible change (optional param defaulting to `None`) that enables the notebook's live progress without coupling `src/` to ipywidgets.
- **Dict collation helper**: Bridges the `CropDataset` tuple output to the trainer's expected dict input without modifying either class's interface.
