# Notebook 0 Upgrade Plan: Metadata-Free Risk-Aware Data Prep

## Summary

Upgrade Notebook 0's grouped-prep pipeline so it handles third-party datasets without provenance metadata by using stronger proxy signals instead of true metadata. The first implementation should stay conservative and mostly automatic: detect duplicate/derivative/source-style risks, triage likely label errors, keep uncertain items out of canonical `val/test`, and expose the outcome through light notebook summaries plus new audit artifacts.

This plan keeps the current Notebook 0 role and runtime dataset contract intact. It extends the existing grouped audit and materialization flow rather than replacing it.

## Implementation Changes

### 1. Strengthen metadata-free grouping and risk detection

Extend the grouped-prep audit in `scripts/prepare_grouped_runtime_dataset.py` to add a new source-style risk layer on top of the existing exact-hash / pHash / DINOv3 / BioCLIP family grouping.

Required behavior:

- Derive weak provenance proxies from observable cues only:
  - filename and path tokens
  - watermark / stock-site / screenshot hints
  - border/layout traits
  - aspect-ratio and resolution buckets
  - compression / web-export style cues where practical
  - existing `source_hint` and `source_like_group` signals
- Build a source-style grouping pass that clusters images likely originating from the same external package, screenshot workflow, or web source style even when they are not near-duplicates.
- Promote these groups into split assignment constraints for canonical evaluation, similar in spirit to current family bundling.
- Keep the existing grouped-family split policy and canonical-eval behavior, but make it harder for visually related or source-style-related samples to leak across `continual` vs `val/test`.

Defaults:

- Conservative mode.
- Source-style risk does not delete samples; it affects split eligibility and routing.
- Risky but usable samples default to `continual` only.

### 2. Add label-noise triage to Notebook 0 audit outputs

Add a new audit-time label review stage that works without metadata.

Required behavior:

- Run a lightweight label-noise triage pass after grouped families are built.
- Use metadata-free signals only, combining:
  - cross-class duplicate/conflict findings already present
  - borderline same-class review clusters
  - representation-neighbor disagreement against local class neighborhoods
  - confusion-prone samples identified by nearest-neighbor class inconsistency
- Score each sample for likely mislabel / ambiguity risk and classify into:
  - `clear`
  - `train_only_risk`
  - `review_candidate`
  - `blocking_conflict`
- Auto-route `train_only_risk` samples to `continual` only.
- Emit a small, high-signal manual review queue for `review_candidate`.
- Keep `blocking_conflict` behavior materialization-blocking when the signal is strong enough, aligned with current cross-class duplicate blocking.

New artifacts/interfaces:

- `label_review_candidates.csv`
- `label_risk_summary.json`
- additional row-level fields in the grouped manifest / family manifest such as:
  - `label_risk_level`
  - `label_risk_reason`
  - `train_only_routed`
- These fields are additive only and must not break existing notebook consumers.

### 3. Make split routing explicitly conservative for canonical eval

Update materialization planning so canonical `val/test` contain only the safest evaluation candidates.

Required behavior:

- `val/test` eligibility should require all of:
  - family-canonical status
  - not synthetic
  - not eval-quality-risk
  - not source-style-risked for canonical eval
  - not label-risked beyond `clear`
- Samples failing those conditions remain eligible for `continual` unless they are blocking conflicts.
- If a class loses too many eval-eligible units after the new filters, the class-health report should reflect that and materialization should block exactly as it does today for insufficient evaluation families/bundles.
- Preserve the current "low review burden" goal by auto-routing uncertain-but-not-severe samples instead of forcing review in the common case.

### 4. Keep Notebook 0 UX light and artifact-driven

Expose the new behavior with minimal notebook surface changes.

Required behavior:

- Keep the notebook mostly unchanged structurally.
- Add a short summary cell or emitted summary section covering:
  - source-style risk counts
  - label review queue size
  - number of samples auto-routed to `continual` only
  - whether canonical eval splits remain healthy after the stricter rules
- Update guided prep artifacts so the new files appear in the "start here" flow.
- Do not add heavy interactive review UI in v1.

### 5. Documentation and contract updates

Update maintained docs to describe the new metadata-free prep behavior accurately.

Required behavior:

- Document that Notebook 0 now uses proxy provenance signals when source metadata is unavailable.
- Clarify that:
  - canonical `val/test` are intentionally stricter than `continual`
  - risky third-party samples may be retained for training but excluded from benchmark splits
  - label triage is heuristic and review-assisted, not ground truth
- Keep claims framed as repo behavior plus literature-backed rationale, not as paper-faithful reproduction.

## Test Plan

Add or update narrow tests around the grouped-prep path.

Required scenarios:

- Source-style-related images are bundled or routed so they do not split across canonical eval boundaries.
- Samples with synthetic / eval-quality / source-style / label-risk flags are kept out of `val/test` and routed to `continual`.
- Strong cross-class conflict still blocks materialization.
- Classes with insufficient safe eval families after new filtering still block with clear class-health output.
- New artifact files are generated and listed in guided prep outputs.
- Existing consumers of `proposed_split_manifest.json` and materialized `split_manifest.json` continue to work with additive fields present.
- Notebook import validation and grouped-prep unit tests remain green.

## Assumptions and Defaults

- Datasets usually arrive without reliable provenance metadata.
- v1 should focus on Notebook 0 audit/materialization and include label-noise triage, but not a post-training feedback loop.
- The default operating mode is conservative.
- The default review burden is low: automatic routing first, short review queues second.
- Uncertain but non-blocking label-risk samples go to `continual` only by default.
- Notebook UX stays light; most detail lives in artifacts, manifests, and guided summaries.
