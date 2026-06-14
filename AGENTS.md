# AADS v6 Agent Guide

Keep this file short and stable. Future Codex sessions should always follow the rules below.

- Before making changes, read `PROJECT_STATE.md`.
- Use the repo docs as source of truth, especially `README.md`, `docs/README.md`, `docs/architecture/overview.md`, `docs/architecture/code_organization_map.md`, `docs/user_guide/colab_training_manual.md`, and `docs/user_guide/ood_readiness_guide.md`.
- Keep Codex context small by default: use targeted `rg`/file reads, avoid broad recursive listings, and do not read large notebooks, datasets, generated reports, or binary artifacts unless the task explicitly requires them.
- Treat `data/`, `runs/`, `models/adapters/`, `outputs/`, `.runtime_tmp/`, large `docs/ablation_results/**/*.json`, and `colab_notebooks/*.ipynb` as high-token surfaces; prefer summaries from scripts or narrow line/field extraction.
- Cap noisy command output with tool limits or `./scripts/cap_output.ps1` before running commands likely to print long logs, manifests, JSON, CSV, or test traces.
- Keep diffs narrow and avoid editing generated outputs under `runs/`, `models/adapters/`, `outputs/`, or `.runtime_tmp/` unless explicitly asked.
- Preserve the repo-wide shared-platform organization model: durable logic belongs in `src/`, while notebooks, notebook cells, and scripts should orchestrate canonical helpers/workflows instead of becoming independent implementations.
- Before broad notebook/script refactors, run `./scripts/python.cmd scripts/audit_code_organization.py` and treat `src -> scripts` imports as boundary violations.
- Prefer the repo launcher on Windows: `./scripts/python.cmd`.
- Package management is pip-based: use `requirements.txt` for runtime and `requirements-dev.txt` for development; no alternate package manager is defined.
- Python target is 3.11. Keep Ruff-compatible code style, with line length 120 and `E/F/I` linting. Mypy ignores missing imports.
- Update `PROJECT_STATE.md` only when durable state changes: architecture, decisions, bugs/workarounds, TODOs, or supported commands.

Common commands:

- Install: `./scripts/python.cmd -m pip install -r requirements.txt`
- Dev: `./scripts/python.cmd -m pip install -r requirements-dev.txt`
- Build: no dedicated build command is defined in the repo
- Lint: `ruff check src scripts tests`
- Test: `pytest tests/unit tests/colab/test_smoke_training.py -q` and `pytest tests/integration -q --runintegration`
- Typecheck: `mypy --follow-imports skip src/shared src/data src/training/continual_sd_lora.py src/router src/workflows src/pipeline/router_adapter_runtime.py src/pipeline/inference_payloads.py src/adapter src/core/config_manager.py scripts/benchmark_surfaces.py`
