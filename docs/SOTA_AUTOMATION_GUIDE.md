# SOTA-Aligned Automation Guide for AADS v6

This guide tailors continuous automation strategies to AADS v6's core surfaces: continual SD-LoRA training, OOD readiness, router-driven inference, and Colab notebook wrappers. Each automation is grounded in literature and repo architecture.

**Last updated:** May 2026  
**Scope:** Training pipeline, OOD calibration, router performance, artifact integrity, and deployment readiness

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [SOTA Context: What This Repo Implements](#sota-context-what-this-repo-implements)
3. [Recommended Automation Cascade](#recommended-automation-cascade)
4. [Implementation Priority](#implementation-priority)
5. [Literature Anchors](#literature-anchors)

---

## Executive Summary

AADS v6 is a **production-oriented plant disease detection pipeline** that chains training, OOD readiness assessment, and router-driven inference. Current CI already validates imports, schemas, datasets, and runs unit + integration tests with coverage gates.

### What To Automate Next (Prioritized)

**Tier 1 (Immediate):**

1. **OOD Evidence Consistency** – Enforce that `production_readiness.json` verdicts align with actual OOD/fallback data
2. **Router Calibration Regression** – Auto-sweep router thresholds and detect when confidence/margin gates degrade
3. **Adapter Smoke Test CI** – Run Notebook 3 directly in CI to catch export/load contract breaks
4. **Code Quality & Bug-Fix Automation** – Automated linting, type checks, test gating, and crash/error monitoring

**Tier 2 (Foundation, 1–2 weeks each):**

4. **Dataset Integrity Monitoring** – Detect data leakage, class imbalance, and OOD pool contamination automatically
5. **Adapter Metadata Validation** – Ensure every exported adapter includes complete calibration state and readiness evidence
6. **Risk-Coverage Reporting** – Generate router risk-coverage curves alongside standard accuracy metrics

**Tier 3 (Hardening, 2–4 weeks each):**

7. **Artifact Manifest + Lineage** – Auto-index training runs and link each to its dataset, config, and readiness verdict
8. **Threshold Drift Detection** – Monitor when router/adapter confidence distributions shift across runs
9. **Notebook Output Validation** – Verify that notebook cells produce expected artifact types and schemas

---

## SOTA Context: What This Repo Implements

### Current SOTA Techniques Already Applied

| Technique | Source | Repo Surface | Status |
|-----------|--------|--------------|--------|
| **Effective-Number Class Weighting** | Cui et al. 2019 | `src/training/services/class_balance.py` | Active |
| **LogitNorm Regularization** | Wei et al. 2022, ICML | Integrated into training loss | Active |
| **Energy-Based OOD Scoring** | Liu et al. 2020, NeurIPS | Supported in `src/training/services/ood_benchmark.py` | Active |
| **Mahalanobis Distance** | Lee et al. 2018 | OOD detector backbone | Active |
| **Outlier Exposure (OE)** | Hendrycks et al. 2018 | `data/<crop>/oe/` support | Ready but optional |
| **Selective Prediction (Abstention)** | Geifman & El-Yaniv 2017 | Router `unknown_crop`, `router_uncertain` | Active |
| **Risk-Coverage Evaluation** | Selective prediction lit. | `scripts/evaluate_router_part_surface.py` | Partial (parts only) |
| **Temperature Scaling Calibration** | Guo et al. 2017, ICML | Supported in inference | Ready but not auto-swept |
| **SAM3 + BioCLIP-2.5 Routing** | Kirillov et al. 2024, Radford et al. 2021 | `src/router/router_pipeline.py` | Active |
| **Held-Out OOD Benchmark** | Bitterwolf et al. 2023, ICML | `src/training/services/ood_benchmark.py` | Fallback when real OOD missing |

### Key Architectural Boundaries

- **Canonical training entrypoint**: `src/workflows/training.py` → `TrainingWorkflow.run(...)`
- **Canonical inference entrypoint**: `src/workflows/inference.py` → `InferenceWorkflow.predict(...)`
- **Adapter contract**: `models/adapters/<crop>/<part>/continual_sd_lora_adapter/`
- **Readiness verdict**: `production_readiness.json` (three verdicts: `ready`, `provisional`, `not_ready`)
- **Router evaluation**: `scripts/evaluate_router_part_surface.py` (crop + part routing)
- **Dataset contract (training)**: `data/prepared_runtime_datasets/<dataset_key>/{continual,val,test,ood,oe}/<class>/*`

---

## Recommended Automation Cascade

### Tier 1: Immediate Wins (High ROI, Low Cost)

#### 1A. OOD Evidence Consistency Check

**Problem:** Training can produce `production_readiness.json` with a `ready` verdict even if the actual `data/<crop>/ood/` or `oe/` pools are missing or incomplete.

**Automation:**

```powershell
# scripts/validate_ood_evidence_consistency.py

# On every CI run after training:
1. Load production_readiness.json
2. Check verdict logic:
   - If verdict == "ready" or "provisional":
     - Verify data/<crop>/ood/ exists and contains ≥10 OOD images
     - If held-out benchmark was used (no real ood/):
       - Verify fallback_benchmark_evidence.json exists
       - Check that fallback FPR @ 95% TPR < threshold
3. Check readiness policy alignment:
   - Verify readiness.json uses current ood_recommendation.md policy
   - Flag if BER gates or OOD gates diverge from policy
4. Emit report to .runtime_tmp/ood_consistency_report.json
```

**Trigger:** On every training run + daily scheduled check of existing runs  
**Pass/Fail Criteria:** All verdicts must have evidence; all gates must match policy  
**Literature Anchor:** Bitterwolf et al. 2023 (OOD evaluation distortion); ood_recommendation.md § "Real OOD Evidence Matters"

---

#### 1B. Router Calibration Regression Test

**Problem:** Router thresholds can be left with stale `router_min_confidence` / `router_min_margin` values when a new router version or image distribution arrives.

**Automation:**

```powershell
# scripts/validate_router_calibration_stability.py

# On every PR that touches src/router/ or inference config:
1. If data/router_eval/ exists:
   - Run src/router/calibrate_router_surface.py sweep:
     - Sweep router_min_confidence: [0.5, 0.6, 0.7, 0.75, 0.8]
     - Sweep router_min_margin: [0.05, 0.1, 0.15]
     - Fix SAM3 and BioCLIP versions
   - Compare curves to baseline (e.g., main branch results)
   - Fail if FPR @ target TPR degrades >2% or abstention rate >30%
   - Report: risk-coverage curves + threshold recommendations
2. Else (no eval set):
   - Warn and suggest data/router_eval/ collection
```

**Trigger:** On PR with router code changes + monthly scheduled sweep  
**Pass/Fail Criteria:** Risk-coverage must not degrade; top-1 crop accuracy maintained  
**Literature Anchor:** 
- SAM 3 paper (Kirillov et al., 2024) on prompt-guided object detection
- Guo et al. 2017 on post-hoc calibration
- router_performance_literature_review.md § "Calibrate Existing Gates Before Adding Models"

---

#### 1C. Adapter Smoke Test in CI

**Problem:** Exported adapters can have latent load/inference bugs that are only caught when Notebook 3 runs manually.

**Automation:**

```powershell
# Integrate into .github/workflows/ci.yml:

- name: Run adapter smoke test
  if: contains(github.event.head_commit.modified, 'src/adapter/') || contains(github.event.head_commit.modified, 'src/training/')
  run: |
    python scripts/colab_adapter_smoke_test.py \
      --adapter-root models/adapters \
      --output .runtime_tmp/adapter_smoke_test.json \
      --strict

# scripts/colab_adapter_smoke_test.py enhancements:
1. Iterate over every exported adapter in models/adapters/*/
2. For each adapter:
   - Load config and adapter state
   - Run 5 synthetic test images through inference
   - Verify output payload contract (crop, disease, confidence, ood_info)
   - Check confidence ranges and OOD calibration
3. Report pass/fail per adapter
4. Fail CI if any adapter fails or confidence violates bounds
```

**Trigger:** On every commit to `src/adapter/` or `src/training/`; post-training runs  
**Pass/Fail Criteria:** All adapters must load, infer, and produce valid payloads  
**Literature Anchor:** Selective prediction (Geifman & El-Yaniv 2019) – output contracts are part of safe inference

---

#### 1D. Code Quality, Bug-Fix, and Optimization Automation

**Problem:** Style, type, regression, and performance issues slip into training/inference code; slow detection causes flaky runs, silent failures, and suboptimal implementation choices.

**Automation:**

```powershell
# Add to CI: lint, types, and unit test gate
- name: Lint, Types, Tests
   run: |
      python -m pip install -r requirements-dev.txt
      ruff check src scripts tests
      mypy --follow-imports skip src
      pytest tests/unit -q --maxfail=1

# Add a lightweight optimization signal:
- name: Performance and complexity checks
  run: |
   python scripts/benchmark_surfaces.py --output .runtime_tmp/benchmarks.json
   ruff check src scripts tests --select C90

# Recommended extras:
# - Pre-commit hooks: ruff/black/isort/mypy
# - Dependabot or scheduled GitHub Action for dependency updates
# - Flaky-test detection (pytest-rerunfailures / flaky) to surface nondeterminism
# - Runtime exception monitoring (Sentry) for deployed adapters
# - Duplicate-code or dead-code review for repeated or unreachable logic
# - Benchmark regression alerts for slow paths, excessive memory use, or unstable latency
# - Cyclomatic-complexity / maintainability guardrails for high-risk modules
```

**Trigger:** On every PR and push to main/master; scheduled dependency checks
**Pass/Fail Criteria:** Lint/type checks pass; no new unit-test regressions; no new flaky-test alerts; benchmark and complexity thresholds stay within policy
**Literature Anchor:** NIST SSDF (traceability) and CI/CD best practices

---

### Tier 2: Foundation (Deeper Integration, Higher Impact)

#### 2A. Dataset Integrity Monitoring

**Problem:** Training can use datasets that have subtle leakage (same image in train+test), class imbalance that violates guardrails, or OOD contamination.

**Automation:**

```powershell
# scripts/monitor_dataset_integrity.py

# Before training begins:
1. Check for exact-image duplicates within {continual, val, test, ood, oe}:
   - Compute image hashes (perceptual + MD5)
   - Fail if same hash appears in train + test
   - Warn if same hash appears in train + ood (should not train on OOD)
   
2. Class balance analysis:
   - For each supported class in continual split:
     - Count images
     - Warn if <100 images (unless few-shot mode)
     - Compute largest/smallest ratio
     - If >1.5x, record for sampler/loss rebalance decision
   
3. OOD pool composition:
   - Verify ood/ and oe/ are disjoint (different images)
   - Verify ood/ contains no supported-class images (manual audit flag)
   - Warn if od/ and oe/ pools <20 images each
   
4. Emit structured report:
   - .runtime_tmp/dataset_integrity.json
   - Include leakage matrix, class counts, duplication rate
```

**Trigger:** On dataset materialization (Notebook 0) + before every training run  
**Pass/Fail Criteria:** 
- No exact duplicates between splits
- Class balance within policy
- OOD/OE disjointness verified
**Literature Anchor:** 
- data_augmentation_leakage_prevention.md
- augmentation_implementation_guide.md
- Hendrycks et al. 2018 on Outlier Exposure pool separation

---

#### 2A.1 Automated OOD/OE Augmentation from Web (Human-in-the-loop)

**Problem:** Small or weak `ood/` and `oe/` pools reduce OOD evaluation quality. Manual curation is slow.

**Automation (high level):**

1. Detect weak splits during dataset integrity checks:
   - Thresholds: `ood_count < 50` or `oe_count < 50` (configurable per-crop)
2. Run candidate image discovery against licensed public sources:
   - Prefer sources with explicit licenses: Wikimedia Commons, Flickr (license-filtered), specialized plant repositories, and vetted public datasets.
   - Avoid scraping copyrighted paywalled sources; require license metadata (CC0/CC-BY) before download.
   - Use curated keyword templates derived from crop + "disease", "symptom", "leaf spot", "unknown", and domain synonyms.
3. Download candidates into the repo-native quarantine inbox: `data/internet_image_candidates/<run>/` with provenance metadata (source URL, license, query, intended adapter, intended split).
4. Automatic de-duplication and leakage checks:
   - Compute perceptual hashes (pHash/average hash) and MD5 for each candidate.
   - Compare against `data/prepared_runtime_datasets/<dataset_key>/{continual,val,test}` hashes — reject exact or near-duplicate matches (threshold e.g., hamming distance <= 6).
   - Run a nearest-neighbor feature check in repo's feature space (if available) to flag overly similar images to supported classes.
5. Human review UI and acceptance workflow:
   - Generate review grids and a manifest under `data/internet_image_candidates/<run>/` with provenance and similarity scores.
   - A human reviewer accepts/rejects each candidate; accepted images are promoted into the matching prepared-runtime `ood/` or `oe/` split by manifest.
6. Final validation:
   - Re-run leakage checks, compute new class counts, and emit `.runtime_tmp/dataset_integrity_postaugment.json`.

**Implementation notes & safeguards:**

- Avoid contamination: ensure `oe/` (auxiliary outlier exposure) and `ood/` (readiness evidence) remain disjoint sets and do not contain supported-class images.

**Promotion script:** `scripts/apply_internet_candidate_manifests.py`

Basic usage (local test):

```powershell
.\scripts\python.cmd scripts\apply_internet_candidate_manifests.py \
  --manifest-root data/internet_image_candidates \
  --repo-root . \
  --dry-run
```

**Trigger:** Run automatically when dataset integrity detects weak OOD/OE pools or run on-demand from notebook 0. Always require human review before promoting into prepared-runtime `ood/` or `oe/`.

**Pass/Fail Criteria:**
- Candidates collected and quarantined with provenance metadata.
- No perceptual or MD5 duplicates promoted into train/val/test.
- Human reviewer accepted set has license and provenance recorded.

**Literature & policy anchors:** Bitterwolf et al. 2023 (evaluation quality), Hendrycks et al. 2018 (OE pool separation), data_augmentation_leakage_prevention.md

---

#### 2B. Adapter Metadata Validation

**Problem:** Exported adapters can be missing OOD state, calibration metadata, or readiness evidence. Inference then fails silently or uses stale state.

**Automation:**

```powershell
# scripts/validate_adapter_metadata_completeness.py

# On every adapter export + CI sweep:
1. For each adapter in models/adapters/<crop>/<part>/continual_sd_lora_adapter/:
   - Check metadata.json:
     - ✓ model_version, crop, part, training_date
     - ✓ calibration_state (temperature, energy bounds, kNN distances)
     - ✓ readiness_verdict + readiness_evidence_path
     - ✓ supported_classes list
     - ✓ effective_number_weights (if applied)
     - ✓ ood_detector_config (energy threshold, mahalanobis params)
     
   - Check adapter state:
     - ✓ LoRA weights present + loadable
     - ✓ Classifier head present
     - ✓ OOD feature statistics (mean, cov for mahalanobis)
     
   - Check readiness artifact at readiness_evidence_path:
     - ✓ production_readiness.json
     - ✓ metric_gate.json
     - ✓ BER plots / confusion matrices
     
2. Emit report:
   - Per-adapter checklist (pass/fail/warn)
   - Missing fields list
   - Calibration state audit
```

**Trigger:** Post-training export + weekly sweep of deployed adapters  
**Pass/Fail Criteria:** All required metadata fields present; readiness evidence reachable  
**Literature Anchor:** LogitNorm (Wei et al. 2022) and calibration state carry forward inference safety

---

#### 2C. Risk-Coverage Reporting (Full Router)

**Problem:** Router evaluation currently reports part precision/recall, but not full crop+part risk-coverage trade-off or false-accept rates.

**Automation:**

```powershell
# Extend scripts/evaluate_router_part_surface.py:

# If data/router_eval/ exists with full crop/part/part-wrongness labels:

1. Compute crop risk-coverage curves:
   - For each confidence threshold: (TPR, FPR on off-crop / non-plant negatives)
   - For each margin threshold: (Abstention rate, accuracy on accepted)
   - Report: tables + plots of ROI operating points
   
2. Compute part precision @ crop-abstention thresholds:
   - Part non-unknown precision (avoid spurious `part=unknown`)
   - Part recall conditional on crop=accepted
   - Report: part quality vs. crop abstention coupling
   
3. Full metrics table:
   - Crop accuracy | Off-crop FPR | Non-plant FPR | Crop abstention rate
   - Part prec. | Part recall | Part unknown rate | Unsupported part count
   - Router latency (p50, p95, p99)
   - SAM3 instance + ROI counts
   
4. Artifact:
   - .runtime_tmp/router_full_surface_eval.json
   - Plots: .runtime_tmp/router_risk_coverage_curves.png
```

**Trigger:** Monthly or post-router config change; on data/router_eval/ updates  
**Pass/Fail Criteria:** Crop FPR @ 95% TPR < policy threshold; part precision > threshold  
**Literature Anchor:**
- Selective prediction (Geifman & El-Yaniv 2019) – risk-coverage is the right metric
- GLIP / Grounding DINO papers on object-centric routing
- router_performance_literature_review.md § "Abstention Is A Feature"

---

### Tier 3: Hardening (Strategic Long-Term)

#### 3A. Artifact Manifest + Lineage Indexing

**Problem:** Training runs scatter artifacts across `runs/<crop>/`, `outputs/`, and `models/adapters/`. It's hard to link a deployed adapter back to its training config, dataset, and readiness verdict.

**Automation:**

```powershell
# scripts/index_artifact_lineage.py (enhancement)

# On every training run completion:
1. Create manifest entry in runs/_index/<run_id>.json:
   {
     "run_id": "grape_v2_2026-05-10_full",
     "crop": "grape",
     "part": "leaf",
     "dataset_key": "prepared_runtime_datasets/grape_full_may2026",
     "dataset_sha256": "abc123...",
     "config_used": "config/base.json + config/colab.json",
     "config_sha256": "def456...",
     "training_start": "2026-05-10T10:00:00Z",
     "training_end": "2026-05-10T14:30:00Z",
     "adapter_export_path": "models/adapters/grape/leaf/continual_sd_lora_adapter/",
     "readiness_verdict": "ready",
     "readiness_evidence_path": "runs/grape/v2_2026-05-10_full/production_readiness.json",
     "metrics": {
       "val_accuracy": 0.94,
       "test_accuracy": 0.92,
       "ood_fpr_at_95_tpr": 0.08
     },
     "lineage_parents": [
       "runs/_index/grape_v1_2026-04-15_ablation.json"  // if continualprevious checkpoint
     ]
   }

2. Update runs/_index/index.json with run_id + fingerprint
3. Generate .runtime_tmp/artifact_lineage_report.json:
   - For each deployed adapter, list full chain:
     - adapter → training run → dataset → config
     - Readiness evidence path
     - Metrics summary
```

**Trigger:** On training completion + daily cache refresh  
**Pass/Fail Criteria:** Every deployed adapter must have complete lineage chain  
**Literature Anchor:** NIST SSDF SP 800-218 (reproducible, traceable development)

---

#### 3B. Threshold Drift Detection

**Problem:** Router confidence distributions can shift across weeks as new images arrive; thresholds optimized for May data may not match June data.

**Automation:**

```powershell
# scripts/detect_router_threshold_drift.py

# On weekly basis (scheduled):
1. Collect router inference logs from recent runs:
   - Extract: crop_score, margin, ood_status, actual_crop for recent inferences
   
2. Compute confidence distribution histograms:
   - This week's distribution
   - Previous week's distribution
   - Previous month baseline (if available)
   
3. Statistical test (e.g., Wasserstein distance, KL divergence):
   - Alert if distribution shifts significantly (p<0.05)
   - Report: "Router confidence has drifted 15% vs. baseline"
   
4. Suggest action:
   - "Recommend re-calibration using scripts/calibrate_router_surface.py"
   - Link to fresh router_eval set if available
   
5. Artifact: .runtime_tmp/router_drift_report.json
```

**Trigger:** Weekly scheduled job; on major dataset update  
**Pass/Fail Criteria:** Alert (not fail) if drift > threshold; suggest recalibration  
**Literature Anchor:** Guo et al. 2017 on calibration drift; selective prediction under distribution shift

---

#### 3C. Notebook Output Validation

**Problem:** Notebooks can produce incomplete or malformed artifacts (e.g., Notebook 2 missing `confusion_matrices.json`).

**Automation:**

```powershell
# scripts/validate_notebook_outputs.py

# After each notebook run in CI or on demand:

1. Notebook 0 output check:
   - Verify data/prepared_runtime_datasets/<key>/ created
   - Check split counts match expected
   - Audit class list vs. plant_taxonomy.json

2. Notebook 1 output check:
   - Verify router_inference_log.json exists
   - Check router output payloads (crop, part, confidence fields)
   - Validate abstention reasons recorded

3. Notebook 2 output check:
   - Verify adapter exported to models/adapters/<crop>/<part>/
   - Check training_log.json exists + complete
   - Verify confusion_matrices.json, metric_gate.json, production_readiness.json
   - Validate readiness verdict logic
   - Verify calibration_state saved in adapter

4. Notebook 3 output check:
   - Verify smoke_test_report.json exists
   - Check all test images produced inference payloads
   - Validate no errors in inference log

5. Notebook 5 output check:
   - Verify router_calibration_sweep.json exists
   - Check that sweep covered expected thresholds
   - Validate risk-coverage curves generated

Emit report: .runtime_tmp/notebook_output_validation.json
```

**Trigger:** Post-notebook run (in CI if notebook runs added); on manual trigger  
**Pass/Fail Criteria:** All required artifacts present; schemas match contracts  
**Literature Anchor:** Contractual entrypoints (src/shared/contracts.py) are enforced

---

## Implementation Priority

### Quick Start

1. **1A. OOD Evidence Consistency**
   - Add to CI after training runs
   - Simple policy check against ood_recommendation.md

2. **1B. Router Calibration Regression**
   - Integrate with existing `calibrate_router_surface.py`
   - Add to CI for PR touching router code

3. **1C. Adapter Smoke Test CI**
   - Integrate `colab_adapter_smoke_test.py` into CI workflow
   - Run on adapter changes

4. **1D. Code Quality & Bug-Fix Automation**
   - Add CI lint/type/test gate (`ruff`, `mypy`, `pytest`)
   - Install pre-commit hooks and dependency update automation

### Medium Term

4. **2A. Dataset Integrity Monitoring**
   - Perceptual hashing + leakage detection
   - Add to dataset prep pipeline

5. **2B. Adapter Metadata Validation**
   - Checklist of required metadata fields
   - Audit calibration state completeness

6. **2C. Risk-Coverage Reporting**
   - Extend existing router eval script
   - Generate risk-coverage curves

### Long Term

7. **3A. Artifact Manifest + Lineage**
   - Index training runs, link to adapters
   - Build traceability dashboard

8. **3B. Threshold Drift Detection**
   - Statistical distribution monitoring
   - Weekly alerts + recalibration suggestions

9. **3C. Notebook Output Validation**
   - Per-notebook artifact schema checks
   - Add to notebook CI or trigger

---

## Literature Anchors

### Core Training & OOD References

| Topic | Paper | Key Insight | Repo Application |
|-------|-------|------------|------------------|
| **Class Imbalance** | Cui et al., 2019, CVPR | Effective number re-weighting improves few-class learning | `src/training/services/class_balance.py` |
| **Overconfidence** | Wei et al., 2022, ICML | LogitNorm regularization reduces train-time overconfidence | Integrated in loss |
| **Energy-Based OOD** | Liu et al., 2020, NeurIPS | Energy scores outperform raw softmax for OOD detection | OOD benchmark |
| **Mahalanobis Distance** | Lee et al., 2018 | Feature-space distance for anomaly detection | OOD detector |
| **Outlier Exposure** | Hendrycks et al., 2018 | Auxiliary outlier pool improves OOD generalization | Ready in `data/<crop>/oe/` |
| **OOD Eval Distortion** | Bitterwolf et al., 2023, ICML | Poor eval sets distort OOD conclusions; small clean set > large flawed set | ood_recommendation.md |

### Router & Inference References

| Topic | Paper | Key Insight | Repo Application |
|-------|-------|------------|------------------|
| **Object-Centric Routing** | Kirillov et al., 2024 (SAM3) | Prompt-guided segmentation better than whole-image for fine-grained tasks | SAM3 + BioCLIP routing |
| **Biological Vision** | Radford et al., 2021 (CLIP); Li et al., 2024 (BioCLIP-2.5) | Plant-specific embeddings improve zero-shot species/disease detection | BioCLIP-2.5 backbone |
| **Glocal Evidence** | LGNet (plant vision); Zhang et al. (deep anomaly) | Local symptom + global context carry complementary information | `global_crop_context_weight` |
| **Selective Prediction** | Geifman & El-Yaniv, 2019 | Explicit abstention optimizes risk-coverage trade-off | `router_uncertain`, `unknown_crop` |
| **Calibration** | Guo et al., 2017, ICML | Modern nets poorly calibrated; temperature scaling is effective baseline | Post-hoc calibration ready |
| **Risk-Coverage** | Selective prediction lit. | Accuracy alone misleads; risk-coverage curves are the right metric | Router eval reports both |
| **Conformal Prediction** | Angelopoulos & Bates, 2021 | Distribution-free set-prediction guarantees under exchangeability | Ready for multi-crop output |

### Validation & Reproducibility References

| Topic | Paper | Key Insight | Repo Application |
|-------|-------|------------|------------------|
| **Test Contamination** | Bitterwolf et al., 2023, ICML | Train/test overlap distorts evaluation; prevent leakage first | Leakage detection automation |
| **Reproducible ML** | NIST SSDF SP 800-218 | Traceable, auditable development; versioned artifacts | Manifest + lineage indexing |
| **Semantic Duplication** | Logistic insights from dedup lit. | Perceptual hashes catch transformed duplicates | Dataset integrity automation |

---

## Integration Checklist

### Canli Guncelleme Kurali

Bu dokuman yasayan bir rehberdir. `Phase 1`, `Phase 2`, `Phase 3` checklist maddeleri ve `Literature Anchors` tablosu; ilgili CI calistirmalari, notebook ciktilari, scheduled workflow sonuclari veya yeni kabul edilen literatur geldikce bu markdown icinde guncellenmelidir.

Bu dosya kullanilarak bir is tamamlandiginda, son adim olarak internette yeni yapilacak isler ve yeni literatur adaylari icin tarama yapilmali; bulunan uygun maddeler dogrudan bu rehbere islenmelidir.

- Yeni bir automation eklendiginde ilgili checklist maddesi isaretlenmeli ve kisa bir durum notu eklenmelidir.
- Yeni bir paper kabul edildiginde yalnizca `docs/SOTA_AUTOMATION_GUIDE.md` icindeki uygun literatur tablosu guncellenmelidir.
- Canonical kaynak sadece bu dosyadir; aday dosyasi tutulmamalidir.

### Phase 1 Checklist (Tier 1)

- [x] Implement `validate_ood_evidence_consistency.py`; add to CI post-training
- [x] Implement `validate_router_calibration_stability.py`; integrate into PR checks
- [x] Enhance `colab_adapter_smoke_test.py` with contract validation; add to CI
- [x] Document passing criteria in each script's docstring
- [x] Add all three scripts to [AGENTS.md](../AGENTS.md) validation defaults
 - [x] Implement CI lint/type/test gate (ruff, mypy, pytest); add pre-commit hooks
 - [x] Add Dependabot/scheduled action for dependency updates

**Automated SOTA updates:** The repository runs a scheduled workflow that queries arXiv for recent papers matching a small set of repo-relevant queries and helps refresh this guide directly. See `scripts/update_sota_references.py` and `.github/workflows/sota_auto_update.yml`.

#### Automated SOTA Literature Updates

- Purpose: keep the SOTA guidance and literature anchors fresh by surfacing recent, relevant papers for reviewer consideration, then updating this markdown directly.
- Files:
   - `scripts/update_sota_references.py` queries arXiv for configured keywords, refreshes a managed candidate-scan section inside this guide, and adds a lightweight repo-local bug/weak-point/improvement scan for triage.
   - If every configured arXiv query fails during a run, the updater preserves the previous managed candidate scan instead of replacing reviewer context with a transient outage report.
   - `.github/workflows/sota_auto_update.yml` runs weekly, runs the script, and pushes the resulting guide changes back to the current branch for review.
- Trigger: weekly schedule (workflow cron), or manual `workflow_dispatch`.
- Automatic push policy: the script itself does not push. The scheduled workflow may surface or commit changes for review, but it must not create a new branch, auto-merge, or bypass review before this guide is updated.
- Evidence scope: work derived from this guide should be literature-grounded where it changes ML methods, evaluation policy, threshold logic, or data-curation guidance. Repo wiring, CI scheduling, report formatting, and notebook/automation ergonomics are engineering adaptations; document them as current repo behavior or engineering inference instead of claiming they are directly literature-derived.
- Reviewer flow: the workflow or local run surfaces candidate literature, and reviewers should update the relevant Literature Anchors section in this file directly.
- Repo opportunity scan: the same run also surfaces local `TODO`/`FIXME`/`HACK`/`BUG`/`XXX` markers plus guide/workflow consistency risks as triage candidates. Treat these as prompts for review, not confirmed defects.
- Local test: run the script locally to generate candidate suggestions for this guide:

```bash
python scripts/update_sota_references.py --output docs/SOTA_AUTOMATION_GUIDE.md
```

Note: the script only suggests candidates inside this guide; human review is required before promoting any literature into the canonical Literature Anchors tables.

<!-- BEGIN SOTA AUTOMATION CANDIDATES -->
#### Latest Automated Candidate Scan

Generated: `2026-06-06T08:14:27Z`

These are machine-collected literature candidates for human review. They are not accepted repo guidance until a maintainer promotes them into the relevant Literature Anchors table above.

Candidate scan could not query all configured sources:

- `out-of-distribution detection`: 429 Client Error: Too Many Requests for url: https://export.arxiv.org/api/query?search_query=all%3Aout-of-distribution+detection&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending
- `energy based ood`: query timed out
- `mahalanobis ood`: query timed out

##### Self-Supervised Learning of Plant Image Representations

- Query: `bioclip`
- Published: `2026-04-30T07:41:49Z`
- Authors: Ilyass Moummad, Kawtar Zaher, Hervé Goëau, Jean-Christophe Lombardo, Pierre Bonnet, Alexis Joly
- Link: http://arxiv.org/abs/2604.27538v1
- Review note: Automated plant recognition plays a crucial role in biodiversity monitoring and conservation, yet current approaches rely heavily on supervised learning, which is limited by the availability of expert-labeled data. Self-supervised learning (SSL) offers a scalable alternative, but existing methods and training protocols are largely designed for coarse-grained visual tasks and may not transfer well to fine-grained domains such as plant species recognition. In this work, we investigate SSL for plant image representation learning. We show that commonly used augmentations in SSL pipelines - such as Gaussian blur, grayscale conversion, and solarization - are detrimental in the context of plant images, as they remove subtle discriminative cues essential for fine-grained recognition. We instead identify alternative transformations, including affine and posterization, that are better suited to this domain. We further demonstrate that training SimDINOv2 on the iNaturalist 2021 Plantae subset yields significantly stronger representations than training on ImageNet-1K, highlighting the importance of domain-specific data for SSL. Our findings are consistent across both ViT-Base and ViT-Large architectures. Moreover, our models achieve competitive performance and sometimes outperform strong supervised baselines Pl@ntCLEF and BioCLIP on downstream plant recognition tasks in few-shot settings. Overall, our results highlight the critical importance of domain-adapted augmentation strategies and dataset selection in self-supervised learning, and provide practical guidelines for building scalable models for biodiversity monitoring.


#### Repo Bug / Weak Point / Improvement Scan

No lightweight repo-local improvement signals found in the configured roots.

<!-- END SOTA AUTOMATION CANDIDATES -->

### Phase 2 Checklist (Tier 2)

- [x] Implement `monitor_dataset_integrity.py` with leakage detection
- [x] Implement `validate_adapter_metadata_completeness.py`
- [x] Extend `evaluate_router_part_surface.py` to full crop+part risk-coverage
- [x] Add reports to `.runtime_tmp/` with clear pass/fail indicators
- [x] Update skill guidance in `skills/aads-repo-hygiene/SKILL.md`

### Phase 3 Checklist (Tier 3)

- [x] Enhance `index_training_runs.py` with full lineage chain + manifest
- [x] Implement `detect_router_threshold_drift.py` with statistical tests
- [x] Implement `validate_notebook_outputs.py` with per-notebook checks
- [x] Build summary dashboard linking runs → adapters → readiness verdicts
- [x] Add weekly drift detection to `.github/workflows/ci.yml` schedule

---

## Appendix: Command Reference

### Run Automations Locally

```powershell
# Validate OOD evidence (Tier 1A)
.\scripts\python.cmd scripts/validate_ood_evidence_consistency.py \
  --runs-root runs/ \
  --output .runtime_tmp/ood_consistency_report.json

# Validate router calibration (Tier 1B)
.\scripts\python.cmd scripts/validate_router_calibration_stability.py \
  --router-eval-root data/router_eval/ \
  --output .runtime_tmp/router_calibration_stability_report.json

# Run adapter smoke tests (Tier 1C)
.\scripts\python.cmd scripts/colab_adapter_smoke_test.py \
  --adapter-root models/adapters/ \
  --output .runtime_tmp/adapter_smoke_test.json \
  --strict

# Monitor dataset integrity (Tier 2A)
.\scripts\python.cmd scripts/monitor_dataset_integrity.py \
  --root data/prepared_runtime_datasets \
  --output .runtime_tmp/dataset_integrity.json

# Validate adapter metadata (Tier 2B)
.\scripts\python.cmd scripts/validate_adapter_metadata_completeness.py \
  --adapter-root models/adapters/ \
  --output .runtime_tmp/adapter_metadata_completeness.json

# Index artifact lineage (Tier 3A)
.\scripts\python.cmd scripts/index_training_runs.py \
  --runs-root runs/ \
  --output-root .runtime_tmp/run_index

# Detect router drift (Tier 3B)
.\scripts\python.cmd scripts/detect_router_threshold_drift.py \
  --root runs \
  --output .runtime_tmp/router_drift_report.json

# Validate notebook outputs (Tier 3C)
.\scripts\python.cmd scripts/validate_notebook_outputs.py \
  --output-root outputs/colab_notebook_training \
  --output .runtime_tmp/notebook_output_validation.json
```

---

## Questions & Feedback

- **Is this aligned with your deployment pipeline?** Check [docs/user_guide/ood_readiness_guide.md](./user_guide/ood_readiness_guide.md) for readiness policy.
- **Which tier should I prioritize?** Start with Tier 1 (high ROI, low cost); Tier 2 adds foundation; Tier 3 is long-term hardening.
- **Where does this fit in AGENTS.md?** Each automation maps to a skill in `skills/aads-*`. See [../AGENTS.md](../AGENTS.md#routing-rules).
