# SOTA Automation Guide for AADS v6

This is the operating guide for the repo's self-refreshing SOTA automation loop. Its job is not to be a static wishlist. Every use of this file should either refresh evidence, run guardrails, classify blockers, or select the next repo change that moves AADS v6 forward.

**Last maintained:** June 2026  
**Canonical refresh target:** `docs/SOTA_AUTOMATION_GUIDE.md`  
**Managed updater:** `scripts/update_sota_references.py`  
**Scheduled workflow:** `.github/workflows/sota_auto_update.yml`

---

## Purpose

Use this guide to answer one practical question:

> What is the next evidence-backed change that improves training, OOD readiness, router reliability, adapter safety, or notebook/release hygiene?

The guide has three living surfaces:

1. **Accepted repo guidance**: the guard inventory, operating loop, decision rules, and literature anchors below.
2. **Machine-generated candidates**: the managed candidate block between `BEGIN SOTA AUTOMATION CANDIDATES` and `END SOTA AUTOMATION CANDIDATES`.
3. **Execution evidence**: guard reports in `.runtime_tmp/`, CI logs, notebook artifacts, and current repo state.

Only accepted repo guidance is binding. Candidate-scan output is input for review, not a decision by itself.

---

## Operating Loop

Run this loop whenever the user asks to apply the SOTA automation guide or when the scheduled workflow changes the guide.

1. **Read current state**
   - Read `PROJECT_STATE.md`.
   - Read this guide.
   - Check `git status --short` and avoid unrelated user-owned changes.

2. **Refresh machine evidence**
   - Run the updater:

```powershell
.\scripts\python.cmd scripts\update_sota_references.py --output docs/SOTA_AUTOMATION_GUIDE.md
```

   - If every configured literature query fails and the script prints `Preserved existing SOTA candidate scan because all configured queries failed`, treat the pass as an evidence-preserving no-op, not as a successful literature refresh.
   - If at least one query succeeds, review the managed candidate block for repo relevance before promoting anything into accepted guidance.
   - Every default updater run also performs a lightweight repo-local bug, weak-point, and suboptimal-code scan over the configured source/documentation roots. Treat those results as triage candidates, then confirm before changing code.

3. **Run the narrow guard set**
   - Prefer the narrowest checks that match the touched or questioned surface.
   - Read JSON/report fields literally; separate missing-data skips from real failures.
   - Do not mark a guide pass green just because a guard skipped.

4. **Classify the next action**
   - **Promote literature** when a candidate directly changes ML method, OOD policy, data-curation policy, calibration logic, or evaluation protocol.
   - **Fix repo automation** when a guard exposes a real bug, stale contract, false positive, weak report, or Windows-specific failure.
   - **Collect artifacts/data** when the blocker is missing adapters, router eval data, OOD samples, or calibration output.
   - **Close as no-op** only when the updater preserved prior evidence and focused verification shows no repo delta.

5. **Persist durable changes**
   - Update this guide when accepted guidance changes.
   - Update `PROJECT_STATE.md` only for durable architecture, workflow, bug/workaround, supported-command, or TODO changes.
   - Keep generated outputs under `runs/`, `models/adapters/`, `outputs/`, and `.runtime_tmp/` out of commits unless explicitly requested.

---

## Guard Inventory

These are the current repo-native automation surfaces. Use report fields and exit codes as evidence; do not infer pass/fail from file existence alone.

| Surface | Command | Primary question | Expected interpretation |
|---|---|---|---|
| OOD evidence consistency | `.\scripts\python.cmd scripts\validate_ood_evidence_consistency.py --runs-root runs --output .runtime_tmp/ood_consistency_report.json` | Do readiness verdicts have real OOD/fallback evidence? | Missing evidence is a real readiness problem unless the report marks it as an absent-artifact skip. |
| Router calibration stability | `.\scripts\python.cmd scripts\validate_router_calibration_stability.py --router-eval-root data/router_eval --output .runtime_tmp/router_calibration_stability_report.json` | Did router confidence/margin gates drift? | `calibration_results_missing` means run calibration before claiming stability. |
| Adapter smoke test | `.\scripts\python.cmd scripts\colab_adapter_smoke_test.py --adapter-root models/adapters --output .runtime_tmp/adapter_smoke_test.json --strict` | Can exported adapters load and emit valid payloads? | `adapter_count=0` is an absent-adapter skip, not a model-quality pass. |
| Adapter metadata completeness | `.\scripts\python.cmd scripts\validate_adapter_metadata_completeness.py --adapter-root models/adapters --output .runtime_tmp/adapter_metadata_completeness.json` | Do adapters carry readiness and calibration metadata? | Missing metadata is actionable for exported adapters; no adapters is a skip. |
| Dataset integrity | `.\scripts\python.cmd scripts\monitor_dataset_integrity.py --root data/prepared_runtime_datasets --output .runtime_tmp/dataset_integrity.json` | Are splits leaked, imbalanced, or contaminated? | Exact image hash overlap across splits is a real failure. Low counts may be warnings depending on dataset mode. |
| Notebook outputs | `.\scripts\python.cmd scripts\validate_notebook_outputs.py --output .runtime_tmp/notebook_output_validation.json` | Do maintained notebooks emit required artifacts and schemas? | Missing notebook output artifacts should be routed to the notebook that owns them. |
| Training-run index | `.\scripts\python.cmd scripts\index_training_runs.py --runs-root runs --output-root .runtime_tmp/run_index` | Can runs, adapters, configs, and readiness evidence be traced? | Backfilled indexes are useful evidence; missing source artifacts remain blockers. |
| Router drift | `.\scripts\python.cmd scripts\detect_router_threshold_drift.py --root runs --output .runtime_tmp/router_drift_report.json` | Have router score distributions shifted? | Drift alerts should usually route to calibration, not immediate threshold edits. |
| Notebook/import/config checks | `.\scripts\python.cmd scripts\validate_notebook_imports.py`, `.\scripts\python.cmd scripts\validate_config_schema.py` | Are maintained surfaces importable and schema-compatible? | These are low-cost baseline checks after guide or automation changes. |
| Updater unit tests | `.\scripts\python.cmd -m pytest tests/unit/scripts/test_update_sota_references.py -q` | Did updater behavior stay stable? | Required after editing `scripts/update_sota_references.py`. |

---

## Decision Rules

### Literature Candidate Promotion

Promote a machine-collected paper into the accepted literature anchors only when all are true:

- It is directly relevant to plant disease detection, OOD detection, selective prediction, router calibration, segmentation-assisted routing, adapter safety, or reproducible ML operations.
- It changes a repo decision or creates a concrete next action.
- The summary is reviewed by a human; the candidate block alone is not authority.
- The accepted note is placed outside the managed candidate block.

Reject or leave candidates unpromoted when they are only keyword matches, unrelated cross-domain work, general ML background with no repo action, or duplicates of already accepted guidance.

### Source-Outage Handling

The updater queries external sources and can hit arXiv rate limits or timeouts. Handle outcomes this way:

- **All queries failed**: preserve the existing managed candidate block and report a no-op refresh.
- **Some queries succeeded**: accept the partial scan as fresh machine evidence, keep source-failure notes visible, and review surviving candidates normally.
- **No new relevant papers**: keep the guide unchanged unless the repo bug/weak-point/suboptimal-code scan finds a real local issue.

### Guard Skip Handling

Skips are useful evidence only when they identify the missing prerequisite.

- `adapter_count=0`: collect/export adapters before validating adapter quality.
- `sample_count=0` or no eval images: collect evaluation samples before claiming router or notebook behavior.
- `calibration_results_missing`: run `scripts/calibrate_router_surface.py` before treating router thresholds as stable.
- Missing OOD/OE pools: classify readiness as blocked or provisional according to the report and current readiness policy.

---

## Current SOTA Alignment

| Accepted area | Repo surface | Status | Next useful pressure |
|---|---|---|---|
| Effective-number class weighting | `src/training/services/class_balance.py` | Active | Verify class-count reports when new datasets are prepared. |
| LogitNorm regularization | training loss path | Active | Keep calibration and overconfidence metrics visible in readiness reports. |
| Energy and Mahalanobis OOD scoring | `src/training/services/ood_benchmark.py` | Supported | Prefer real OOD/OE evidence over fallback-only claims. |
| Fine-grained OOD stress | real `ood/` slices, `ood_type_breakdown`, and readiness reports | Active as evaluation pressure | Add or review near-OOD, background/style, and subtle-symptom slices before adopting heavier OOD architectures. |
| Outlier Exposure | prepared dataset `oe/` split | Optional | Grow OE pools through reviewed, licensed candidates only. |
| Selective prediction and abstention | router `unknown_crop`, `router_uncertain` | Active | Track risk-coverage and false accepts, not just accuracy. |
| Router calibration | router calibration and stability scripts | Active but artifact-dependent | Rebuild `.runtime_tmp/router_calibration.json` when missing or stale. |
| Segmentation-assisted routing | `src/router/router_pipeline.py` | Active | Validate crop/part routing with current router eval data. |
| Adapter contract safety | adapter smoke and metadata validators | Active | Treat missing exported adapters as a prerequisite gap, not a pass. |
| Dataset leakage control | `scripts/monitor_dataset_integrity.py` | Active | Exact split overlap is a hard blocker. |
| Artifact lineage | `scripts/index_training_runs.py` | Active | Keep run/adapters/readiness traceability current before release claims. |

---

## Accepted Literature Anchors

| Topic | Anchor | Repo use |
|---|---|---|
| Class imbalance | Cui et al., 2019, CVPR | Effective-number weighting in class-balance services. |
| Overconfidence | Wei et al., 2022, ICML | LogitNorm-style training regularization and calibration pressure. |
| Energy OOD | Liu et al., 2020, NeurIPS | Energy score support in OOD benchmark/readiness evidence. |
| Mahalanobis OOD | Lee et al., 2018 | Feature-distance OOD detector baseline. |
| Outlier Exposure | Hendrycks et al., 2018 | Separate auxiliary OE pools from readiness OOD evidence. |
| OOD evaluation distortion | Bitterwolf et al., 2023, ICML | Prefer clean, leak-free OOD evidence over large weak pools. |
| Fine-grained OOD feature decoupling | Li, Huang, and Guan, 2026 | Use as a review trigger for near-OOD, background/style, and subtle visual confounder stress before changing detector architecture. |
| Plant-domain self-supervised representations | Moummad et al., 2026 | Review augmentation policy and representation choices for plant-specific cues; avoid cue-erasing transforms unless local ablation supports them. |
| Calibration | Guo et al., 2017, ICML | Post-hoc calibration and threshold stability checks. |
| Selective prediction | Geifman and El-Yaniv, 2019 | Explicit abstention and risk-coverage reporting. |
| Conformal prediction | Angelopoulos and Bates, 2021 | Future multi-crop set-prediction guidance under exchangeability. |
| Reproducible ML operations | NIST SSDF SP 800-218 | Traceable artifacts, configs, readiness evidence, and release hygiene. |

---

## Managed Candidate Scan

The updater owns only the block below. Do not put accepted decisions inside it; they may be overwritten by the next successful refresh.

<!-- BEGIN SOTA AUTOMATION CANDIDATES -->
#### Latest Automated Candidate Scan

Generated: `2026-06-14T10:28:36Z`

These are machine-collected literature candidates for human review. They are not accepted repo guidance until a maintainer promotes them into the relevant Literature Anchors table above.

##### Self-Supervised Learning of Plant Image Representations

- Query: `bioclip`
- Published: `2026-04-30T07:41:49Z`
- Authors: Ilyass Moummad, Kawtar Zaher, Hervé Goëau, Jean-Christophe Lombardo, Pierre Bonnet, Alexis Joly
- Link: http://arxiv.org/abs/2604.27538v1
- Repo action hint: Review plant-domain representation and augmentation policy; avoid transformations that erase subtle leaf/symptom cues unless a local ablation proves they help.
- Review note: Automated plant recognition plays a crucial role in biodiversity monitoring and conservation, yet current approaches rely heavily on supervised learning, which is limited by the availability of expert-labeled data. Self-supervised learning (SSL) offers a scalable alternative, but existing methods and training protocols are largely designed for coarse-grained visual tasks and may not transfer well to fine-grained domains such as plant species recognition. In this work, we investigate SSL for plant image representation learning. We show that commonly used augmentations in SSL pipelines - such as Gaussian blur, grayscale conversion, and solarization - are detrimental in the context of plant images, as they remove subtle discriminative cues essential for fine-grained recognition. We instead identify alternative transformations, including affine and posterization, that are better suited to this domain. We further demonstrate that training SimDINOv2 on the iNaturalist 2021 Plantae subset yields significantly stronger representations than training on ImageNet-1K, highlighting the importance of domain-specific data for SSL. Our findings are consistent across both ViT-Base and ViT-Large architectures. Moreover, our models achieve competitive performance and sometimes outperform strong supervised baselines Pl@ntCLEF and BioCLIP on downstream plant recognition tasks in few-shot settings. Overall, our results highlight the critical importance of domain-adapted augmentation strategies and dataset selection in self-supervised learning, and provide practical guidelines for building scalable models for biodiversity monitoring.


#### Repo Bug / Weak Point / Suboptimal Code Scan

No lightweight repo-local bug, weak-point, or suboptimal-code signals found in the configured roots.

<!-- END SOTA AUTOMATION CANDIDATES -->

---

## Local Verification Sets

Use the smallest verification set that matches the change.

### Guide-only edit

```powershell
.\scripts\python.cmd scripts\update_sota_references.py --output docs/SOTA_AUTOMATION_GUIDE.md --skip-repo-scan
.\scripts\python.cmd -m pytest tests/unit/scripts/test_update_sota_references.py -q
```

### Updater script edit

```powershell
.\scripts\python.cmd -m py_compile scripts/update_sota_references.py
.\scripts\python.cmd -m pytest tests/unit/scripts/test_update_sota_references.py -q
```

### Automation behavior or CI-contract edit

```powershell
.\scripts\python.cmd scripts\validate_notebook_imports.py
.\scripts\python.cmd scripts\validate_config_schema.py
.\scripts\python.cmd scripts\validate_notebook_outputs.py
```

### Broad confidence check

```powershell
.\scripts\python.cmd -m ruff check src scripts tests
.\scripts\python.cmd -m pytest tests/unit tests/colab/test_smoke_training.py -q
```

---

## Closing Criteria

A SOTA guide pass is complete only when the final report states:

- whether the literature refresh was fresh, partial, or preserved because of source outage;
- which guard commands ran;
- which skips were legitimate absent-artifact states;
- which failures are real blockers;
- what concrete next action follows from the evidence;
- whether `PROJECT_STATE.md` needed an update.

Do not close with generic "guide updated" language. The closeout should make the next engineering move obvious.
