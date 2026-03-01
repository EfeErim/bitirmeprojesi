# Evidence Protocol

## Severity Classes

- `P0`:
  - Causes data loss, corrupted outputs, security-critical failure, or hard crash in core runtime paths.
  - Blocks safe usage of primary pipeline behavior.
- `P1`:
  - Breaks a major feature or produces materially wrong results in normal workflows.
  - No safe workaround exists or workaround is operationally costly.
- `P2`:
  - Causes incorrect behavior in constrained scenarios, partial feature degradation, or unstable edge-path behavior.
  - Workaround exists with moderate operational impact.
- `P3`:
  - Low-impact defect such as non-critical diagnostics mismatch, minor UX/reporting issue, or maintainability risk with bounded runtime effect.

## Accepted Evidence Types

1. Deterministic code contradiction with call-path proof.
- Show contradictory logic and the execution path that reaches it.
- Include exact file and line references.

2. Failing test or command output.
- Include command, key output snippet, and failure assertion/error.
- Keep command reproducible from repository root.

3. Schema/config contract mismatch with exact keys/paths.
- Show expected contract key/path and actual key/path.
- Cite owning source files for both sides of the contract.

## Mandatory Fields For Confirmed Defects

Every `Confirmed Defect` entry must include:

- `id`: Stable identifier (for example `BUG-001`).
- `severity`: One of `P0`, `P1`, `P2`, `P3`.
- `location`: `path:line` (or multiple paths if cross-file).
- `claim`: Concrete defect statement.
- `evidence`: Direct artifact proving the claim.
- `repro`: Exact command or deterministic proof procedure.
- `expected_vs_actual`: Short expected and actual behavior contrast.
- `confidence`: `high` for confirmed defects.

## Mandatory Fields For Evidence-Backed Risks

Every `Evidence-Backed Risk` entry must include:

- `id`
- `severity`
- `location`
- `claim`
- `direct_evidence`
- `missing_proof`
- `next_verification_command`
- `confidence`

Constraints:

- Do not use broad speculative language.
- Keep `missing_proof` specific to one unresolved verification step.
- Keep `next_verification_command` executable without additional interpretation.

## Validation Mapping Source Of Truth

Do not duplicate full module-to-test maps here.
Use existing repository references for command selection:

- `skills/aads-status-triage/references/status-checklist.md`
- `skills/aads-coder/references/coding-playbook.md`

Select the smallest command set needed to prove or reject each candidate finding.
