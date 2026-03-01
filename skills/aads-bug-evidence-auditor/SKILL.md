---
name: aads-bug-evidence-auditor
description: Perform fact-only bug and problem audits across AADS-ULoRA by requiring concrete evidence for every claim, separating confirmed defects from evidence-backed risks, and forbidding guesswork or hallucinated findings. Use when asked to review code for bugs, validate suspected defects, or produce evidence-first problem reports before implementation.
---

# AADS Bug Evidence Auditor

## Workflow

1. Capture repository state.
- Run `git status --short --branch`.
- Run `git diff --name-only`.
- Run `git log --oneline -n 8`.

2. Perform whole-repository static sweep.
- Inspect `src/`, `tests/`, `scripts/`, and `config/`.
- Trace call paths for suspected contradictions.
- Record exact file and line references for each observation.

3. Build the evidence ledger.
- For every candidate issue, record:
  - location (`path:line`)
  - concrete artifact (code contradiction, failing command output, or contract mismatch)
  - exact reproduction command
  - expected vs actual behavior
- Discard any candidate without at least one concrete artifact.

4. Select and run targeted runtime checks for flagged surfaces.
- Use existing validation maps from:
  - `skills/aads-status-triage/references/status-checklist.md`
  - `skills/aads-coder/references/coding-playbook.md`
- Run only module-relevant checks needed to confirm or reject candidates.
- Avoid broad suites unless module coupling makes targeted validation insufficient.

5. Emit findings with strict output contract.
- Return only verified defects and evidence-backed risks.
- Keep findings tied to concrete artifacts and reproducible commands.

## Evidence Standard

- Require at least one concrete artifact for every claim.
- Accept only:
  - deterministic code contradiction with call-path proof
  - failing test or command output
  - schema/config contract mismatch with exact key and path evidence
- Reject inference-only statements and speculative language.
- State coverage limits explicitly when verification cannot be completed.

## Risk Classification Rules

- `Confirmed Defect`:
  - include reproducible proof or deterministic contradiction proof.
  - do not use uncertain wording such as "maybe" or "probably".
- `Evidence-Backed Risk`:
  - include direct evidence.
  - include one explicit missing proof step.
  - include one exact next verification command.
- Do not report any item that lacks evidence and a clear verification path.

## Runtime Check Selection

- Start from static evidence and map each finding to its owning module.
- Choose the smallest validation command set that can confirm behavior.
- Prefer targeted unit/module checks before integration checks.
- Promote to broader checks only when cross-module coupling is directly evidenced.

## Output Contract

Return sections in this order:

1. `Confirmed Defects`
- Use fields: `id`, `severity`, `location`, `claim`, `evidence`, `repro`, `expected_vs_actual`, `confidence`.

2. `Evidence-Backed Risks`
- Use fields: `id`, `severity`, `location`, `claim`, `direct_evidence`, `missing_proof`, `next_verification_command`, `confidence`.

3. `Coverage Performed`
- List scanned surfaces and runtime checks actually executed.

4. `Unverified Areas`
- List areas not proven and why proof is missing.

5. `Suggested Next Checks`
- List minimal follow-up commands needed to close unverified areas.

If no defects are confirmed, state this explicitly and include coverage limits.

## Failure/Unknown Handling

- If a command fails, capture stderr/stdout evidence and classify the finding as unverified until reproduced.
- If environment constraints block runtime proof, keep item in `Evidence-Backed Risks` with missing proof and exact next command.
- If evidence conflicts across sources, report the conflict and do not promote to confirmed defect.

## References

- Read `references/evidence-protocol.md` for severity criteria, evidence formats, and required finding fields.
- Reuse validation mappings from:
  - `skills/aads-status-triage/references/status-checklist.md`
  - `skills/aads-coder/references/coding-playbook.md`
