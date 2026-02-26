# Orchestration Matrix

## Track Routing

| Work type | Primary skill | Minimum gate |
|---|---|---|
| Repo status + impact map | `aads-status-triage` | `git status --short --branch` + scope summary |
| Direct code implementation | `aads-coder` | Touched module tests |
| Router/OOD behavior and perf | `aads-router-ood-guardrails` | Router tests + policy regression (+ perf guardrails if runtime path changed) |
| Training phase behavior | `aads-training-lifecycle` | Training unit tests + Colab smoke training |
| Docs/notebook path updates | `aads-colab-doc-sync` | Markdown links + notebook import validation |
| Contracts/interfaces design | `aads-architect` | Interface diff + compatibility notes |
| Config/pipeline contract changes | `aads-config-pipeline-guardrails` | Validation unit tests + configuration integration check |

## Default Execution Order

1. Status triage
2. Architecture decision (if interfaces change)
3. Domain implementation tracks (parallel when independent)
4. Domain-specific validation
5. Integration checks
6. Final summary

## Integration Gate Bundle

```bash
python scripts/run_python_sanity_bundle.py
python scripts/run_policy_regression_bundle.py
pytest tests/integration -v --runintegration
```

Use this full bundle only for broad, cross-module changes.

## Sequencing Rules

- Run independent implementation tracks in parallel only when they do not touch shared files.
- Run docs/notebook synchronization after behavior-changing tracks are complete.
- Promote expensive gates only when a track changes runtime behavior, config contracts, or user-facing paths.
