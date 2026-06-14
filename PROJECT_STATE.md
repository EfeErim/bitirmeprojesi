# Project State

## Current Objective

Keep the narrow plant-disease repo stable while supporting grouped dataset preparation, continual SD-LoRA adapter training, router-guided inference, OOD/readiness checks, and a clearer client handoff surface.

## Current Implementation Status

- Canonical workflows live in `src/workflows/training.py` and `src/workflows/inference.py`.
- Maintained notebook surfaces are 0, 1, 2, 3, 5, and 8; Notebook 4 is a convenience wrapper, and Notebooks 6/7 are validation surfaces.
- Notebook 16 is the maintained ROI/bbox evidence-gate ablation surface. It keeps full-image adapter prediction as the final decision, uses router/Grounding DINO bbox evidence only for review flags, and can auto-discover matching prepared datasets plus Colab adapter exports for multi-adapter evaluation. Historical ROI ablation reports remain under `docs/ablation_results/<condition>/`.
- The repo is pip-based and uses `./scripts/python.cmd` on Windows so the local `.venv` is preferred.
- CI already covers notebook/import validation, config schema checks, OOD evidence consistency, router calibration stability, adapter smoke tests, metadata completeness, dataset integrity, notebook outputs, and benchmark capture.

## Recent Meaningful Changes

- Latest commit added Codex instructions, a config example, and output-capping helpers, and refreshed `AGENTS.md`.
- Recent notebook calibration work tuned Notebook 5 router calibration defaults to `12/12`, exposed adaptive hyperparameters, and published failure-analysis artifacts.
- The SOTA literature updater filters query-specific candidates more narrowly, restricts BioCLIP candidates to plant-domain context, deduplicates titles, and preserves the previous managed scan when every configured query fails.
- The SOTA literature updater now checks visual-context terms with token boundaries so words like "supervision" do not admit non-visual OOD candidates.
- The SOTA literature updater's default run also performs a lightweight repo-local bug, weak-point, and suboptimal-code scan so every guide application surfaces triage candidates beyond papers.
- `docs/SOTA_AUTOMATION_GUIDE.md` is now an operating guide for the SOTA refresh loop: every pass should refresh machine evidence, run narrow guardrails, classify skips/failures, and select a concrete next repo action instead of maintaining a static wishlist.
- Removed the obsolete one-condition ROI wrapper notebooks and kept Notebook 16 as the single active ROI/bbox evidence-gate notebook. Production inference remains unchanged; historical ROI ablation reports remain under `docs/ablation_results/<condition>/`.
- Removed Notebook 9 because the presentation-only recording wrapper is no longer part of the maintained handoff surface; Notebook 8 remains the canonical single-image router-to-adapter trial surface.
- Notebook 16 target-aware ROI fallback uses Grounding DINO prompts in the Hugging Face lowercase/dot-terminated text-query format and records `grounding_dino_status`/`grounding_dino_error` in ablation rows so detector failures are visible instead of silently becoming zero-candidate fallbacks.
- Notebook 16 now treats full-image adapter prediction as the final decision and uses router/Grounding DINO ROI only as evidence for review flags (`roi_evidence_status`, `requires_review`, `review_reasons`) after ROI scoring underperformed full-image scoring.
- Notebook 16 is no longer tied to `tomato__fruit`; it can evaluate multiple crop/part adapters with target-specific Grounding DINO prompts generated from each target crop and part.
- Notebook 16 reports now include `failure_analysis` from `src/pipeline/roi_evidence_analysis.py`, classifying router, bbox, adapter, confidence/OOD, and review-gate failure signals before retraining decisions.
- Notebook 16 auto-disconnects the Colab runtime after the multi-target report exists and the ablation output push succeeds.
- Latest Notebook 16 multi-target report covers 2,946 samples with 0.8836 accuracy and 0.8563 macro-F1. Its failure analysis shows router bucket count 0, so the next immediate problem is not router handoff; adapter/data quality is the main later investigation, especially `strawberry__fruit` at 0.4510 accuracy, while review-gate capture remains low at 0.3557.
- `docs/architecture/evidence_gate_calibration_plan.md` now records the implementation plan for automated target-aware evidence-gate calibration so review thresholds are not tuned manually per adapter.
- Added the advisory evidence-gate calibration workflow in `src/pipeline/evidence_gate_calibration.py` plus `scripts/calibrate_evidence_gate.py`; it reads Notebook 16 `multi_target_report.json` and writes `docs/ablation_results/dual_view_inference/evidence_gate_calibration.json` without changing runtime inference behavior.
- Evidence-gate calibration now supports `--schema-version v2`, extending v1 into automated report-only hierarchical calibration with wider confidence thresholds, target/group/global fallback, risk-coverage curves, holdout stability checks, and an audit queue.
- The current default v2 calibration run still finds no eligible global policy. It finds target-specific advisory policies for `grape__leaf` and `tomato__leaf`, group fallback policies for `apricot__fruit`, `apricot__leaf`, `strawberry__leaf`, and `tomato__fruit`, and explicit `no_eligible_policy` outcomes for `grape__fruit` and `strawberry__fruit`.
- Updated `docs/ablation_results/dual_view_inference/evidence_gate_calibration_summary.md` as the short repo-facing handoff for the current v2 Notebook 16 calibration result.
- Added `docs/ablation_results/dual_view_inference/multi_target_failure_prioritization.md` to rank Notebook 16 failures across all targets; it keeps `strawberry__fruit` as a real outlier but identifies `tomato__leaf` review-gate miss volume and target-specific calibration as broader next priorities.
- Added `docs/architecture/review_gate_failure_analysis_and_literature.md`; the review gate misses errors because many wrong full-image predictions are high-confidence, so the next solution direction is target-conditional selective prediction/risk-control rather than one global confidence or ROI rule.
- Added `docs/architecture/evidence_gate_calibration_v2_literature_plan.md`, defining v2 as automated hierarchical report-only calibration with target/group/global fallback, risk-coverage curves, holdout stability checks, and an audit queue rather than manual per-adapter tuning.
- `docs/roi_ablation_memory.md` is the durable handoff note for the ROI/bbox/router/adapter retraining discussion, including completed experiments, decisions, and the next-step plan.
- `AGENTS.md` now codifies default Codex context discipline: targeted reads, capped noisy command output, and avoidance of high-token data/generated/notebook/report surfaces unless explicitly needed.
- Added `scripts/summarize_large_report.py` as the default bounded JSON/CSV report summarizer so Codex can inspect metrics, statuses, and representative rows without loading large artifacts into context.
- Tightened the root README and docs index so canonical surfaces and generated/local-only paths are easier to scan during handoff.
- Added a repo-wide code organization map plus `scripts/audit_code_organization.py` so notebook, script, workflow, runtime, service, and shared-code boundaries are explicit and machine-checkable.

## Important Decisions

- Use `requirements.txt` and `requirements-dev.txt` as the dependency source of truth.
- Treat `runs/`, `models/adapters/`, `outputs/`, and `.runtime_tmp/` as generated or local-only surfaces.
- Treat large datasets, notebooks, generated reports, and broad command output as high-token surfaces; use targeted extraction or summaries before loading them into Codex context.
- Use `./scripts/python.cmd scripts/summarize_large_report.py <json-or-csv>` before inspecting large report artifacts directly.
- Prefer narrow validation commands before broad test runs.
- Keep durable logic in `src/`; scripts and notebook cells should orchestrate canonical helpers/workflows instead of becoming independent implementations.

## Known Issues / Risks

- No dedicated build step is defined; validation scripts are the main maintenance surface.
- Router calibration and OOD evidence can drift if evaluation data changes.
- Notebook-exported artifacts should not be confused with canonical source code.
- The current evidence gate catches only about 35.6% of wrong Notebook 16 full-image decisions; global aggressive ROI-quality rules would raise capture but create too many false positives, so any threshold work should be target-specific.
- Evidence-gate calibration is advisory only; v2 default constraints still reject the global policy surface, and selected target/group policies must not be promoted into runtime behavior without a separate validation decision.

## Next Recommended Steps

- Before editing, read this file first.
- Make the smallest change that addresses the request.
- If durable state changes, update this file at the end of the task.
- When calibration, runtime, or benchmark surfaces change, run the narrowest relevant validation first and capture benchmarks if the workflow contract changed.

## Verified Commands

- `./scripts/python.cmd -m pip install -r requirements.txt`
- `./scripts/python.cmd -m pip install -r requirements-dev.txt`
- `./scripts/python.cmd scripts/validate_notebook_imports.py`
- `./scripts/python.cmd scripts/validate_config_schema.py`
- `pytest tests/unit tests/colab/test_smoke_training.py -q`
- `pytest tests/integration -q --runintegration`
- `./scripts/python.cmd scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json`
- `./scripts/python.cmd scripts/audit_code_organization.py`
- `./scripts/python.cmd scripts/summarize_large_report.py docs/ablation_results/dual_view_inference/multi_target_report.json`
- `./scripts/python.cmd scripts/calibrate_evidence_gate.py`
- `./scripts/python.cmd scripts/calibrate_evidence_gate.py --schema-version v2`
- `ruff check src scripts tests`
- `mypy --follow-imports skip src/shared src/data src/training/continual_sd_lora.py src/router src/workflows src/pipeline/router_adapter_runtime.py src/pipeline/inference_payloads.py src/adapter src/core/config_manager.py scripts/benchmark_surfaces.py`
