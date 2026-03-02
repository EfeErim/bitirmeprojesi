---
name: aads-colab-doc-sync
description: Keep Colab notebooks, canonical scripts, and repository documentation synchronized for AADS-ULoRA. Use when notebook flow, script entrypoints, docs navigation, or compatibility alias guidance changes.
---

# AADS Colab and Docs Sync

## Workflow

1. Confirm canonical entrypoints.
- Treat `colab_notebooks/archive/v6_superseded_2026-03-02/` as archived/superseded assets.
- Use active notebook references only when new canonical notebooks are explicitly introduced.

2. Synchronize docs and scripts.
- Keep `README.md`, `scripts/README.md`, and `docs/REPO_FILE_RELATIONS.md` aligned with notebook/script changes.
- Keep `docs/README.md` and notebook index pages aligned with renamed or re-prioritized entrypoints.

3. Preserve compatibility policy.
- Prefer canonical `scripts/...` entrypoints; do not rely on removed root wrapper aliases.

4. Validate documentation and import surfaces.
- Run markdown link checks and notebook import validation.
- Run Colab environment checks when relevant.

5. Emit synchronization summary.
- List every file synchronized and any intentional deprecations.

## Output Contract

Return sections in this order:
1. Changed user entrypoints
2. Synced docs/scripts/notebooks
3. Validation commands and results
4. Compatibility/deprecation notes
5. Remaining doc drift risks

## References

- Read `references/colab-doc-sync-checklist.md` for synchronization points.
