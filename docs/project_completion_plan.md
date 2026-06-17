# Project Completion Plan

Last updated: 2026-06-16

This document is the completion contract for the June 2026 handoff. It turns the current repo state and owner answers into a concrete scope, milestone plan, deliverable list, and definition of done.

## Completion Goal

By the end of June 2026, the project should work as a demo product for plant-disease inference:

1. A user provides a plant photo.
2. The system identifies the plant and usable plant part through the router.
3. The system loads the correct plant/part adapter when available.
4. The system predicts the disease.
5. If the disease, crop, part, adapter, or input evidence is not reliable enough, the system returns an explicit unknown/abstain status instead of forcing a bad answer.

The final handoff is not a web or mobile application. It is a repo, Colab, PowerPoint, and live-demo handoff that a company or evaluator can run and understand.

## Final Delivery Strategy

The project is optimized for safe delivery, not maximum feature count.

- Demo surface: Colab Notebook 8, with fallback captured outputs/screenshots.
- Presentation language: Turkish narration with English technical terms.
- Supported final target set: 8 plant/part surfaces for tomato, strawberry, grape, and apricot, each with fruit and leaf coverage where adapters are available and reliable enough.
- Demo image source: a convincing mix of internet images, phone-captured images, and random/user-like photos.
- User guidance is part of the product: users will be told what kind of plant photos the system expects, but the demo still needs enough variation to prove practical reliability.
- Weak adapters should first get audit/fix attention. If not fixed in time, they should be labeled `low_confidence` or `experimental` instead of being presented as fully supported.

## In Scope

The final scope includes:

- Router-guided single-image inference through the maintained inference workflow and Notebook 8.
- Correct adapter resolution for the supported plant/part surfaces that already exist.
- Unknown/abstain behavior for unsupported crops, uncertain router output, missing adapters, non-plant input, and low-confidence/OOD adapter output.
- SAM3/Grounding DINO/router evidence as required supporting evidence for crop/part handoff and review flags.
- Notebook 16 as the maintained ROI/bbox evidence-gate ablation and review-signal surface.
- Colab notebook handoff for dataset preparation, training, adapter validation, router calibration, inference, and evidence-gate analysis.
- Dataset and training guide documentation.
- Architecture and handoff documentation.
- A curated live-demo checklist with expected outcomes.
- Repo hygiene: narrow validation checks, notebook import checks, lint where practical, and clear generated-vs-source boundaries.

## Out Of Scope

The following are explicitly out of scope for the June 2026 finish:

- A production web app.
- A mobile application.
- New adapter families or new model-family scouting unless an existing final blocker proves impossible to solve otherwise.
- Broad new dataset collection or new adapter training by default.
- Promoting Notebook 16 evidence-gate policies into runtime disease decisions unless the documented promotion gates pass.
- A full annotation platform.
- Treating generated outputs under `runs/`, `models/adapters/`, `outputs/`, `data/prepared_runtime_datasets/`, or `.runtime_tmp/` as maintained source.

## Key Technical Policy

Runtime inference remains conservative:

- Full-image adapter prediction is the disease decision when an adapter runs.
- Router and bbox/ROI evidence are required for handoff confidence and review signaling.
- ROI/bbox evidence should not override the disease prediction at runtime unless a separate validation decision promotes it.
- Unknown is a valid final answer when the system lacks enough evidence.

This avoids the main delivery risk: a live demo that looks confident while producing poor results.

## Literature-Grounded Rationale

The final plan follows four literature-backed principles:

1. Use a reject/unknown option when risk is too high. Selective classification frames this as a risk-coverage tradeoff: a system can improve reliability on answered examples by abstaining on unsafe examples instead of forcing every prediction.
2. Treat uncertainty and OOD detection as deployment requirements, not extras. Conformal prediction and OOD/OE work support explicit uncertainty handling for black-box vision models, especially when inputs can drift away from the training distribution.
3. Treat plant disease diagnosis as an open-set problem in the final demo. Real users can submit unsupported plants, unsupported plant parts, healthy plants, unknown diseases, non-plant images, or field images outside the trained distribution.
4. Keep local symptom evidence and global plant context separate until validated. Plant-disease literature supports combining local and global cues, but the current repo evidence shows ROI/bbox rules are not yet safe enough to override full-image adapter decisions at runtime.

Practical implication: the demo should be judged by correct answers plus correct abstentions, not only by raw top-1 disease accuracy.

## Deliverables

The final package must include:

| Deliverable | Required state |
|---|---|
| Inference pipeline | Single-image router-to-adapter inference works through the canonical workflow and Notebook 8. |
| Live demo checklist | A small curated set of supported, unknown, uncertain, and failure-path examples with expected statuses. |
| Colab notebooks | Maintained notebooks are runnable or clearly marked for their role: 0, 1, 2, 3, 5, 8, and 16. |
| Dataset and training guide | User can understand how to prepare data, train an adapter, validate export, and interpret readiness. |
| Architecture documentation | Router, adapter, OOD/readiness, and evidence-gate boundaries are clear. |
| Handoff guide | A new evaluator or company reviewer knows what to read first, which Colab notebook to run, which assets are generated, and what limitations remain. |
| Validation evidence | Narrow checks pass or any remaining blockers are written with exact status and workaround. |
| Presentation | Turkish PowerPoint narration with English technical terms explains goal, architecture, demo flow, results, limits, and company handoff path. |
| GitHub handoff | Repo state is clean enough to share, with generated/local-only paths clearly separated. |

## Required Demo Checklist Shape

The demo checklist must be written before final rehearsal. It should include enough images to convince an evaluator that performance is reliable under expected use. The default target is 40-60 images, with the right to reduce only if time is blocked by runtime or asset access.

| Case type | Expected behavior |
|---|---|
| Supported crop/part with known disease | Router accepts, adapter loads, disease prediction is shown with confidence/OOD evidence. |
| Supported crop/part but visually difficult example | Either correct disease or explicit uncertainty/review; no crash and no hidden failure. |
| Supported crop with unsupported/ambiguous part | Router or runtime returns uncertainty instead of forcing the wrong adapter. |
| Unsupported crop | `unknown_crop` or equivalent abstain status. |
| Non-plant image | Non-plant rejection when the input guard is enabled, or a documented fallback status if disabled. |
| Missing adapter path | `adapter_unavailable` or equivalent status, not an unhandled exception. |
| Unknown or out-of-distribution disease | Unknown/OOD/review status when calibrated evidence is unsafe. |
| External dependency unavailable | Documented fallback output or screenshot for presentation continuity. |

This checklist is the first final-delivery artifact because it converts "demo success" into observable pass/fail behavior.

## Failure Definition

For final delivery, these are failures:

- The system assigns the wrong disease to an example it should be able to classify.
- The system returns unknown for a disease it should confidently know, unless the image violates documented input guidance.
- The system maps an unknown disease or unsupported condition onto a known disease label.
- The system crashes or requires code edits during the live demo.
- The system presents an experimental or low-confidence adapter as fully supported.

Correct abstention is not a failure when the input is unsupported, ambiguous, out of distribution, missing an adapter, or outside the documented photo guidance.

## Definition Of Done

The project can be called complete when all of the following are true:

1. A supported plant photo can produce a structured inference payload with crop, part/router summary, adapter prediction, confidence/OOD evidence, and final status.
2. Unknown or unsafe cases return explicit abstain statuses instead of misleading disease labels.
3. The live demo checklist passes without manual code edits between examples.
4. Notebook 8 is the canonical live inference surface.
5. Notebook 16 remains report-only unless its promotion gate passes.
6. Colab notebooks and repo docs agree on maintained surfaces and generated/local-only paths.
7. Required validation commands either pass or have documented, literal blockers.
8. PowerPoint and GitHub handoff are aligned with the same scope.

## Proposed Acceptance Gates

These are practical demo-product gates. They can be tightened if final time allows.

| Gate | Target |
|---|---|
| Demo stability | No crashes across the curated live-demo checklist. |
| Supported crop/part routing | At least 90% correct or explicitly abstained on the curated supported demo set. |
| Disease demo behavior | Target 90% correct on answerable supported demo examples; if time or adapter quality blocks this, 80% is the minimum acceptable threshold with explicit limitations. |
| Unknown behavior | Unsupported, non-plant, missing-adapter, or uncertain cases do not force a disease label. |
| Coverage transparency | The final demo reports how many cases were answered versus abstained/reviewed. |
| Repo hygiene | Notebook import validation, config validation, organization audit, and targeted tests pass or have documented blockers. |
| Handoff clarity | README/docs tell a new user exactly which notebook/command to use for demo, training, and validation. |

## Milestones

### M0 - Scope Freeze (2026-06-16)

Goal: freeze final project boundaries.

- Keep web/mobile apps out of scope.
- Keep new adapter training out of the default path.
- Prioritize demo correctness and handoff.
- Add this completion plan as the planning source of truth.

Exit criteria:

- This document is present and linked from the docs map.
- `PROJECT_STATE.md` records the completion-plan decision.

Concrete tasks:

- [x] Record final scope and out-of-scope items in this document.
- [x] Record safe-delivery strategy and failure definition.
- [x] Link the plan from `docs/README.md`.
- [x] Record durable state in `PROJECT_STATE.md`.

### M1 - Demo Inference Contract (2026-06-17 to 2026-06-19)

Goal: make the live inference path explicit and testable.

- Identify the exact Notebook 8 or script command for live demo.
- Confirm adapter root and supported crop/part inventory.
- Confirm status behavior for supported, unknown crop, uncertain router, missing adapter, and non-plant examples.
- Create the demo checklist file and make every case type explicit.
- Write the user photo guidance that explains expected framing, plant visibility, supported crop/part set, and when unknown is expected.

Exit criteria:

- One command/notebook path is the declared demo surface.
- Demo expected outputs are written before rehearsal.
- The checklist reports answer coverage separately from correctness.
- The eight final supported crop/part surfaces are marked as `supported`, `low_confidence`, or `experimental`.

Concrete tasks:

- [x] Fill `docs/demo_checklist.md` with 40-60 candidate image rows.
- [x] Confirm the exact Notebook 8 parameter cells and run path for demo.
- [x] Confirm adapter root and available adapter bundle for each of the eight target surfaces.
- [x] Define expected behavior for every checklist image before running inference.
- [x] Write final user photo guidance in `docs/demo_checklist.md` and reuse it in the handoff guide.
- [x] Decide where fallback screenshots/outputs will be stored.

### M2 - Demo Reliability And Evidence (2026-06-20 to 2026-06-22)

Goal: reduce the risk of bad live-demo predictions.

- Run the curated demo checklist.
- Record failures literally: router, adapter, OOD, evidence/review, asset missing, or environment.
- Use Notebook 16 reports as review/evidence support, not as unvalidated runtime overrides.
- Fix only narrow blockers needed for demo stability.
- If the demo fails because of a bad adapter class, prefer an explicit abstain/review policy or documented limitation before risky late retraining.
- Audit/fix weak adapters first; if they cannot be stabilized quickly, mark them `low_confidence` or `experimental`.

Exit criteria:

- Demo checklist passes or remaining blockers have exact workarounds.
- Unknown/abstain behavior is visible for unsafe cases.

Concrete tasks:

- [ ] Run Notebook 8 on every row in `docs/demo_checklist.md`. Local start on 2026-06-16 reached `demo_001` and stopped on `dependency_access` because `facebook/sam3` gated Hugging Face access was unavailable. The image requirement was expanded to at least 500 rows; 96 disease-focused internet images were downloaded into `.runtime_tmp/final_demo_images/internet_expansion/`, and a generated supported-disease manifest guarantees 10 images for each of the 37 non-healthy supported disease classes across the eight adapters. Full coverage asset audit is 512/514 file-ready; only fallback screenshots `demo_047` and `demo_048` remain missing.
- [ ] Fill actual status, crop, part, disease, confidence/OOD, pass/fail, and failure bucket.
- [ ] Summarize answered, abstained/reviewed, failed, and per-target pass counts.
- [ ] Fix only failures that block the final demo or handoff.
- [ ] Re-run failed rows after each fix.
- [ ] Label each target surface as `supported`, `low_confidence`, or `experimental`.
- [ ] Capture fallback screenshots/outputs for at least one successful case and one unknown/review case.

### M3 - Handoff Documentation (2026-06-23 to 2026-06-25)

Goal: make the project understandable to an evaluator or company handoff reviewer.

- Refresh README/docs routing if needed.
- Verify dataset and training guide flow.
- Verify architecture documentation points to maintained surfaces.
- Add a dedicated handoff guide covering demo notebook, supported targets, asset paths, generated/local-only paths, validation commands, known limitations, and company handoff notes.

Exit criteria:

- A new user can find demo, training, validation, and generated-output boundaries from docs.
- The handoff guide is sufficient without reading internal planning notes.

Concrete tasks:

- [ ] Complete `docs/handoff_guide.md` with final supported target labels and demo evidence.
- [ ] Link `docs/handoff_guide.md`, `docs/demo_checklist.md`, `docs/final_validation_checklist.md`, and `docs/presentation_outline.md` from `docs/README.md`.
- [ ] Confirm README and docs map route new users to demo, training, validation, and generated-output boundaries.
- [ ] Add exact Colab/runtime access notes needed for demo.
- [ ] Record known limitations and external dependency risks.

### M4 - Presentation And Rehearsal (2026-06-26 to 2026-06-28)

Goal: prepare final presentation and live-demo flow.

- Build PowerPoint around problem, architecture, demo flow, results, limitations, and handoff.
- Rehearse the live demo from a clean terminal/notebook state.
- Capture fallback screenshots or outputs in case live GPU/API access fails.
- Keep Turkish narration with English technical terms.

Exit criteria:

- Presentation and live-demo script tell the same story as the repo.
- Fallback evidence exists for risky external dependencies.

Concrete tasks:

- [ ] Build slides from `docs/presentation_outline.md`.
- [ ] Add one architecture diagram.
- [ ] Add one Notebook 8 output screenshot.
- [ ] Add demo checklist result summary.
- [ ] Add supported-target status table.
- [ ] Rehearse the live demo from a clean Colab session.
- [ ] Keep fallback screenshots ready for external dependency failure.

### M5 - Final Buffer And Freeze (2026-06-29 to 2026-06-30)

Goal: avoid late scope expansion and package the final state.

- Run final narrow validation.
- Do not introduce broad refactors or new model experiments.
- Commit/push final documentation and code changes.
- Write exact known issues instead of trying risky late fixes.

Exit criteria:

- Final repo state is handoff-ready.
- Remaining risks are documented and defensible.

Concrete tasks:

- [ ] Run required checks from `docs/final_validation_checklist.md`.
- [ ] Fill the final status table in `docs/final_validation_checklist.md`.
- [ ] Confirm no unplanned generated outputs are staged.
- [ ] Confirm final supported/low-confidence/experimental labels are consistent across docs and slides.
- [ ] Commit and push final documentation/code changes when requested.
- [ ] Stop feature expansion unless it fixes a final demo blocker.

## Immediate Next 7 Days

1. Create the demo checklist and choose demo image cases.
2. Run Notebook 8 or the equivalent inference command against the checklist.
3. Fix only live-demo blockers in router, adapter loading, status payloads, or docs.
4. Confirm unknown/abstain behavior.
5. Label the eight target plant/part surfaces as supported, low-confidence, or experimental.
6. Decide whether any final adapter audit is required after the demo path is stable.
7. Start the PowerPoint only after the demo story is stable.

## Main Risks

| Risk | Mitigation |
|---|---|
| Inference gives poor live-demo results | Use a curated checklist, explicit unknown status, and narrow fixes before presentation work. |
| Company handoff is hard | Keep canonical surfaces small: README, docs map, Notebook 8, training guide, OOD guide, architecture overview. |
| External gated tools fail during final week | Keep fallback screenshots/outputs and do not depend on live gated access for the final story. |
| Scope expands again | Use this document to reject web/mobile/new-model work unless it fixes a final blocker. |
| Notebook 16 is mistaken for production runtime | Keep it report-only unless promotion gates pass. |
| Random user-like images lower measured performance | Document photo guidance, separate answerable supported cases from out-of-scope cases, and report abstention coverage honestly. |

## Open Decisions

These should be resolved during M1:

- Exact demo checklist file location: `docs/demo_checklist.md`.
- Exact list of supported crop/part adapters available for final live demo: eight target surfaces are inventoried in `docs/demo_checklist.md`; current demo root is `runs/` because `models/adapters/` is empty in this workspace snapshot.
- Exact final validation command set for the handoff run: `docs/final_validation_checklist.md`.
- Exact handoff guide filename and routing from README/docs: `docs/handoff_guide.md`, linked from `docs/README.md`.
- Whether Notebook 16 is mentioned briefly as review-signal research or omitted from the main presentation story: mention briefly as report-only review/evidence support, not as production runtime.

## Literature References

- [Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks"](https://arxiv.org/abs/1705.08500): risk-coverage framing and reject-option rationale.
- [Geifman and El-Yaniv, "SelectiveNet"](https://proceedings.mlr.press/v97/geifman19a.html): integrated reject option and selective prediction framing.
- [Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"](https://arxiv.org/abs/2107.07511): uncertainty sets and distribution-free uncertainty handling for black-box models.
- [Hendrycks, Mazeika, and Dietterich, "Deep Anomaly Detection with Outlier Exposure"](https://arxiv.org/abs/1812.04606): OOD/anomaly detection through auxiliary outlier data.
- [The impact of fine-tuning paradigms on unknown plant diseases](https://pmc.ncbi.nlm.nih.gov/articles/PMC11297179/): plant-disease OSR/OOD benchmarking context.
- [Open-set domain adaptation for tomato disease recognition](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2021.758027/full): unknown plant disease and domain-shift handling.
- [Local and Global Feature-Aware Dual-Branch Networks for Plant Disease Recognition](https://spj.science.org/doi/10.34133/plantphenomics.0208): local symptom and global plant-context evidence.
