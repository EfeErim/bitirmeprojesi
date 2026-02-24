# Phase 2 Implementation: Config Source-of-Truth & Dependency Consolidation

Date: 2026-02-24  
Scope: Resolve config model ambiguity and consolidate Colab dependency manifests.

## Decisions Implemented

1. Config source-of-truth is now explicitly treated as:
   - `config/base.json` for baseline runtime settings
   - `config/<environment>.json` (e.g. `config/colab.json`) for environment overrides

2. Legacy split config files (`router-config.json`, `ood-config.json`, `monitoring-config.json`, `security-config.json`) remain optional backward-compatibility inputs if present.

3. Canonical Colab dependency manifest is now `requirements_colab.txt` at repository root.

4. `colab_notebooks/requirements_colab.txt` is retained only as a compatibility mirror and now points to the canonical root file.

## Changes Applied

### Configuration manager alignment
- Updated `src/core/config_manager.py`:
  - `load_all_configs()` now skips missing legacy split config files silently instead of warning for each absent file.
  - `validate_merged_config()` required section checks updated to current repo scope: `training`, `router`, `ood`.

### Dependency consolidation
- Updated `requirements_colab.txt`:
  - Expanded into canonical Colab dependency set.
  - Removed stale API/database-only packages that no longer match current repo scope.
  - Removed duplicates and normalized package naming.

- Updated `colab_notebooks/requirements_colab.txt`:
  - Replaced duplicated static list with:
    - `-r ../requirements_colab.txt`

### Installer compatibility
- Updated `scripts/install_colab.py` (`create_requirements_file`):
  - Uses canonical `requirements_colab.txt` when available.
  - Falls back to embedded defaults only if canonical file is missing.
  - Preserves compatibility with existing tests and non-repo temporary workspaces.

### Documentation alignment
- Updated `colab_notebooks/README.md`:
  - Clarified canonical vs mirror dependency file responsibilities.

## Validation Notes

- This phase is non-breaking by design:
  - No behavior change to core training/inference logic.
  - Backward compatibility retained for legacy split config files and notebook dependency paths.

## Outcome

- Config model ambiguity reduced.
- Colab dependency drift risk reduced (single source-of-truth).
- Installer behavior now follows canonical dependency source.
- Notebook compatibility maintained.
