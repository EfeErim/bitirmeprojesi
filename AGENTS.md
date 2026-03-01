## Skills

Project-local skills for `d:\bitirme projesi`.

### Available skills

- aads-status-triage: Triage repository state, map impacted surfaces, and define minimal safe validation before edits. (file: skills/aads-status-triage/SKILL.md)
- aads-bug-evidence-auditor: Audit the repository for bugs and problems using concrete evidence only, separating confirmed defects from evidence-backed risks. (file: skills/aads-bug-evidence-auditor/SKILL.md)
- aads-coder: Implement scoped code changes with module-aware validation and minimal blast radius. (file: skills/aads-coder/SKILL.md)
- aads-orchestrator: Sequence cross-track work, dependencies, and integration validation gates. (file: skills/aads-orchestrator/SKILL.md)
- aads-architect: Design architecture and contract changes with compatibility classes and migration planning. (file: skills/aads-architect/SKILL.md)
- aads-config-pipeline-guardrails: Guard config and pipeline contracts with compatibility-focused validation and migration notes. (file: skills/aads-config-pipeline-guardrails/SKILL.md)
- aads-router-ood-guardrails: Update router and OOD behavior while preserving policy regressions and Phase 5 performance guardrails. (file: skills/aads-router-ood-guardrails/SKILL.md)
- aads-training-lifecycle: Update phase1/2/3 training and lifecycle paths while preserving freeze/parity invariants. (file: skills/aads-training-lifecycle/SKILL.md)
- aads-colab-doc-sync: Keep Colab notebooks, canonical scripts, and docs entrypoints synchronized. (file: skills/aads-colab-doc-sync/SKILL.md)

### Skill Selection Matrix

| Request focus | Primary skill | Secondary skill (if needed) |
|---|---|---|
| Need current state, impact map, or test scope | `aads-status-triage` | `aads-orchestrator` |
| Fact-based bug/problem review with evidence-first findings | `aads-bug-evidence-auditor` | `aads-status-triage` |
| Direct code implementation | `aads-coder` | domain-specific skill |
| Cross-module sequencing and handoffs | `aads-orchestrator` | `aads-status-triage` |
| Interface/config/contract redesign | `aads-architect` | `aads-orchestrator` |
| Config/pipeline contract and schema changes | `aads-config-pipeline-guardrails` | `aads-architect` |
| Router, ROI, policy, OOD behavior/perf | `aads-router-ood-guardrails` | `aads-coder` |
| Phase1/2/3 training lifecycle updates | `aads-training-lifecycle` | `aads-coder` |
| Notebook/script/docs path synchronization | `aads-colab-doc-sync` | `aads-status-triage` |
| Core/dataset/pipeline/eval/monitoring updates | `aads-coder` | `aads-status-triage` |

### Invocation and Sequencing Rules

- Named invocation is supported: `$aads-coder`, `$aads-bug-evidence-auditor`, `$aads-orchestrator`, etc.
- Use the minimal set of skills needed for the request.
- For broad tasks, default sequence is:
  1. `aads-status-triage`
  2. `aads-architect` (only if interfaces/contracts change)
  3. Domain implementation skill(s)
  4. `aads-orchestrator` to consolidate cross-track validation and risks
- For review-oriented bug and problem audits, use:
  1. `aads-status-triage`
  2. `aads-bug-evidence-auditor`
  3. Implementation skill(s) only after evidence is established
- Prefer canonical `scripts/` paths in outputs; treat root wrappers as compatibility aliases.
