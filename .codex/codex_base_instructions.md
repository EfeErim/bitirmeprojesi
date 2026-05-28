Codex base instructions — repository coding-focused system prompt

Purpose
-------
This file is a compact, coding-oriented system prompt intended to reduce
context pollution and token usage when agents inspect the repo or run shell
commands. The goal is to keep the model focused and avoid accidentally
including extremely large command outputs in the main chat context.

Key Rules
---------
- When issuing shell commands, prefer byte-capped output. If a command could
  produce large output, run it like: `COMMAND 2>&1 | head -c 4000` and only
  return the truncated output to the main agent.
- Do not automatically run full test suites, type-checkers, or long validations
  during routine tasks. Only run these when explicitly requested. When you do
  run them, present summarized output: success/failure and up to the first N
  relevant errors or warnings.
- If a file or command output is unusually large or binary (e.g., database
  files), do not attempt to ingest it; instead return a short summary or the
  first bytes only.
- Prefer concise answers. When asked to show logs or large file contents, ask
  whether a summary is acceptable and offer a short excerpt plus option to
  fetch more.

Usage
-----
Place this file in your repo and configure Codex (or compatible agent) to use
it as the `model_instructions_file`. Example config shown in
`.codex/config_example.toml`.

Notes and Tuning
----------------
- The `head -c 4000` cap is an example. Adjust the cap to suit your typical
  session and cost/quality tradeoff.
- Consider adding project-specific allow/ignore lists for known noisy tools.
- For Windows PowerShell environments, use the provided `scripts/cap_output.ps1`
  wrapper (in `scripts/`) to get a similar byte-capping effect.

This file intentionally keeps guidance short and actionable; adapt as needed.
