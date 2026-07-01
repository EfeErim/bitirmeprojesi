# Open-World Router Hardening Plan

## Summary

Strengthen the Notebook 8 router/prototype reconciler for gated production use: route only when the crop/part target is supported and confident; otherwise abstain or send the image to review. The plan uses a fresh curated open-world negative set, keeps adapter disease retraining out of scope, and promotes the router only if the evidence supports a strict demo-production safety claim.

Target acceptance claim: **0 observed false accepts on at least 300 fresh open-world negatives**, giving a rough 95% upper bound near 1% for false-accept risk, plus no wrong supported-target handoffs on the supported balanced manifest.

## Key Changes

- Add a fresh open-world router validation manifest under the M2 demo assets:
  - at least `300` negative/OOD rows, disjoint by SHA-256 from training, prototype, calibration, and prior demo images
  - slices: unsupported crops, same-crop unsupported/wrong-part cases, plant-like non-targets, non-plant distractors, and low-quality/ambiguous user-photo cases
  - each row must include `image_id`, `source`, `expected_target`, `expected_crop`, `expected_part`, `expected_behavior`, `ood_slice`, `origin_url`, and provenance notes
- Extend the router evaluation/reporting path so the production gate is measured directly:
  - use the current Notebook 8 router + BioCLIP prototype reconciler, not raw router-only output
  - report supported target handoff accuracy, abstain/review rate, negative false accepts, wrong-part false accepts, opposite-part labels, per-slice OOD results, and latency
  - compute confidence intervals using the simple zero-failure upper-bound rule and include them in the Markdown/JSON report
- Add a production-router gate report:
  - pass only if `wrong_supported_target_handoffs == 0`
  - pass only if `negative_false_accepts == 0` across the fresh 300-negative set
  - pass only if `wrong_part_false_accepts == 0`
  - pass only if supported balanced-manifest route coverage remains at least `80%`
  - pass only if p95 latency is not more than `25%` above the latest accepted M2 router baseline
- Keep the runtime behavior conservative:
  - no adapter is loaded for unsupported crop, unknown crop, unknown part, weak prototype evidence, part conflict, or non-plant rejection
  - do not loosen prototype thresholds to recover adapter coverage unless all safety gates stay green
  - keep disease-class adapter accuracy out of this router-production signoff

## Implementation Changes

- `scripts/build_open_world_router_manifest.py` builds `docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv` from fresh external iNaturalist images, with SHA-256 de-duplication against existing M2/demo/prototype image roots and slice summaries.
- `scripts/run_router_open_world_validation.py` is the thin validation wrapper that can run:
  - the `664` balanced supported manifest for supported-route coverage
  - the fresh open-world negative manifest for rejection confidence
  - one combined report under `docs/demo_results/router_open_world/<timestamp>/`
- `scripts/evaluate_router_open_world_readiness.py` reads the two result JSON files and writes:
  - `router_open_world_readiness.json`
  - `router_open_world_readiness.md`
  - per-slice failure CSVs for any false accepts or wrong handoffs
- Update Notebook 8/run-state docs only after a passing run, documenting the router contract as "gated production candidate," not "forced classifier."

## Current Evidence Set

- Current manifest: `docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv`
- Current image root: `docs/demo_assets/open_world_router/images/`
- Current summary: `docs/demo_assets/open_world_router/manifests/m2_open_world_router_summary.json`
- Current row count: `306`
- Current slice counts:
  - `unsupported_crop`: `115`
  - `same_crop_wrong_part`: `51`
  - `plant_like_non_target`: `60`
  - `non_plant_distractor`: `50`
  - `low_quality_ambiguous`: `30`
- Local manifest audit: `pass`, `306` local hashed images, `0` issues, `0` SHA-256 overlaps against `docs/demo_assets/m2_full_image_set/images`, `data/prepared_runtime_datasets`, and `docs/demo_assets/prototype_curation`.
- Local asset audit: `306/306` asset-ready.

## Commands

Build or refresh the open-world manifest:

```powershell
./scripts/python.cmd scripts/build_open_world_router_manifest.py
```

Local asset audit:

```powershell
./scripts/python.cmd scripts/run_demo_checklist.py --no-checklist --mode asset-audit --extra-manifest docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv --output .runtime_tmp/open_world_router_asset_audit.json --markdown-output .runtime_tmp/open_world_router_asset_audit.md
```

Notebook 8/open-world launch preflight:

```powershell
./scripts/python.cmd scripts/validate_router_open_world_preflight.py --json-output .runtime_tmp/router_open_world_preflight.json --markdown-output .runtime_tmp/router_open_world_preflight.md --fail-on-invalid
```

This verifies the durable run-state JSON, visible Notebook 8 parameter cell, baseline summary availability, supported balanced manifest, open-world manifest/summary, and SHA-256 disjointness before starting the long Colab/GPU run.

Colab/GPU production-gate run, after prototype bank/calibration paths are available:

```powershell
./scripts/python.cmd scripts/run_router_open_world_validation.py --supported-manifest docs/demo_assets/m2_full_image_set/manifests/m2_balanced_80_run_manifest.csv --open-world-manifest docs/demo_assets/open_world_router/manifests/m2_open_world_router_manifest.csv --enable-prototype-reconciler --prototype-artifact-dir docs/demo_results/m2/<accepted_or_current_run>/ --baseline-summary docs/demo_results/m2/<accepted_or_current_run>/summary.json --require-latency-baseline --fail-on-not-ready
```

Maintained Notebook 8 entry point:

- Set `M2_RUN_OPEN_WORLD_ROUTER_VALIDATION = True` in the parameter cell before the Colab/GPU M2 run.
- Keep `M2_OPEN_WORLD_BASELINE_SUMMARY = M2_COMPARISON_BASELINE` unless using a different accepted baseline summary.
- The M2 cell runs this gate after a successful M2 report, reusing the selected prototype bank, taxonomy registry, calibration report, batch sizes, and handoff-cache policy. When enabled, the gate is included in completion checks and auto-push paths.

The wrapper auto-resolves `prototype_bank.json`, `taxonomy_registry.json`, and `router_prototype_calibration.json` from `--prototype-artifact-dir`. If the baseline or candidate reports have no explicit p95 latency field, the gate derives conservative per-image latency from `runner_elapsed_seconds / summary.total` or `elapsed_seconds / row_count`; pass `--baseline-p95-latency-ms` to override the baseline with a measured router p95 value.

Each validation run writes copied provenance under `docs/demo_results/router_open_world/<timestamp>/provenance/`, including the supported manifest, open-world manifest, baseline summary when provided, and prototype artifacts when resolved. If an underlying runner exits before producing JSON, the wrapper still writes a failed `router_open_world_readiness.json`/Markdown report with `supported_report_written` / `open_world_report_written` checks.

Validate a completed result folder before documenting production-candidate status:

```powershell
./scripts/python.cmd scripts/validate_router_open_world_result.py docs/demo_results/router_open_world/<timestamp> --fail-on-invalid
```

By default, the result validator is a production-candidate gate: it requires a passing readiness status, zero wrong supported handoffs, zero negative false accepts, zero wrong-part false accepts, non-regressed latency with both baseline and candidate latency values present, zero runner exit codes, empty failure CSVs, copied manifest provenance, baseline summary provenance, and prototype bank/taxonomy/calibration provenance.

## Test Plan

- Unit tests:
  - manifest builder rejects duplicate hashes and missing provenance
  - readiness gate fails on one negative false accept
  - readiness gate fails on one wrong supported-target handoff
  - readiness gate fails when supported route coverage drops below `80%`
  - readiness gate computes the zero-failure 95% upper bound correctly
- Local validation:
  - `./scripts/python.cmd scripts/validate_notebook_imports.py`
  - `./scripts/python.cmd scripts/validate_router_open_world_preflight.py --fail-on-invalid`
  - targeted pytest for the new manifest/gate tests
  - asset audit over the fresh open-world manifest
- Colab/GPU validation:
  - run the `664` balanced supported manifest with the current router/prototype settings
  - run the fresh `300+` open-world negative manifest
  - require the readiness report to pass before changing docs/run-state to production-candidate

## Assumptions

- Adapter disease-class retraining is explicitly out of scope for this router hardening pass.
- The production contract is gated: abstain/review is acceptable and preferred over forced routing.
- Fresh curated data is preferred over relying only on existing repo OOD images.
- The current BioCLIP prototype reconciler remains the router surface under evaluation.
- The minimum statistical target is strict demo-production confidence, not a formal open-world certification for arbitrary global uploads.
