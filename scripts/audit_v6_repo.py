#!/usr/bin/env python3
"""Run v6 repository audit and emit manifest/coverage/report artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

LEGACY_PATTERN = r"DoRA|CoNeC|phase1|phase3|load_in_4bit|nf4|QLoRA"
MISSING_V55_PATH = "docs/reports/v55/V55_FINAL_STATUS_REPORT.md"
GENERATED_RELATIONS_DOC = "docs/REPO_FILE_RELATIONS_DETAILED.md"
GENERATED_V6_REPORTS_GLOB = "!docs/reports/v6/**"
LEGACY_TOKEN_GUARD_EXCLUDES = (
    "!**/archive/**",
    "!scripts/audit_v6_repo.py",
    f"!{GENERATED_RELATIONS_DOC}",
    GENERATED_V6_REPORTS_GLOB,
)
ARCHIVE_REFERENCE_GUARD_EXCLUDES = (
    "!**/archive/**",
    "!scripts/audit_v6_repo.py",
    "!scripts/run_test_suites.py",
)


def git_command(root: Path, *args: str) -> list[str]:
    return ["git", "-c", f"safe.directory={root.resolve().as_posix()}", *args]


def rg_command(pattern: str, *paths: str, exclude_globs: tuple[str, ...]) -> list[str]:
    command = ["rg", "-n", pattern, *paths]
    for glob in exclude_globs:
        command.extend(["-g", glob])
    return command


def run_cmd(command: list[str], cwd: Path, timeout: int = 1800) -> dict:
    started = dt.datetime.now(dt.timezone.utc)
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        status = "pass" if proc.returncode == 0 else "fail"
        rc = proc.returncode
        out = proc.stdout
        err = proc.stderr
        note = ""
    except subprocess.TimeoutExpired:
        status = "unverified"
        rc = None
        out = ""
        err = ""
        note = f"Timed out after {timeout}s"
    except FileNotFoundError:
        status = "unverified"
        rc = None
        out = ""
        err = ""
        note = f"Executable not found: {command[0]}"
    ended = dt.datetime.now(dt.timezone.utc)
    return {
        "command": " ".join(command),
        "status": status,
        "returncode": rc,
        "stdout": out[:24000],
        "stderr": err[:24000],
        "notes": note,
        "duration_sec": max((ended - started).total_seconds(), 0.0),
    }


def run_rg_no_match(command: list[str], cwd: Path, timeout: int = 600) -> dict:
    result = run_cmd(command, cwd=cwd, timeout=timeout)
    if result["status"] == "unverified":
        return result
    rc = result["returncode"]
    # rg return codes: 0=found, 1=not found, 2=error
    if rc == 1:
        result["status"] = "pass"
        result["notes"] = "No matches found."
    elif rc == 0:
        result["status"] = "fail"
        result["notes"] = "Unexpected matches found."
    else:
        result["status"] = "unverified"
        result["notes"] = f"Unexpected rg return code: {rc}"
    return result


def git_ls_files(root: Path) -> list[str]:
    proc = subprocess.run(
        git_command(root, "ls-files"),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def parse_rg_locations(text: str) -> list[dict]:
    items: list[dict] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        items.append(
            {
                "path": parts[0].replace("\\", "/"),
                "line": int(parts[1]),
                "snippet": parts[2] if len(parts) > 2 else "",
            }
        )
    return items


def ext(path: str) -> str:
    return Path(path).suffix.lower()


def file_class(path: str) -> str:
    suffix = ext(path)
    if suffix == ".py":
        return "python"
    if suffix == ".ipynb":
        return "notebook"
    if suffix in {".md", ".txt"}:
        return "documentation"
    if suffix in {".json", ".yaml", ".yml", ".ini", ".toml"}:
        return "configuration"
    if suffix in {".sh"}:
        return "shell"
    if suffix == "":
        return "meta"
    return "other"


def owner_surface(path: str) -> str:
    if path.startswith("src/router/"):
        return "router"
    if path.startswith("src/ood/"):
        return "ood"
    if path.startswith("src/training/"):
        return "training"
    if path.startswith("src/adapter/"):
        return "adapter"
    if path.startswith("src/pipeline/") or path.startswith("src/dataset/"):
        return "pipeline"
    if path.startswith("src/core/") or path.startswith("config/"):
        return "core/config"
    if path.startswith("src/monitoring/") or path.startswith("src/debugging/") or path.startswith("src/utils/"):
        return "monitoring/debugging/utils"
    if path.startswith("docs/") or path.startswith("colab_notebooks/") or path.startswith("scripts/"):
        return "docs/notebooks/scripts"
    if path.startswith("tests/") or path.startswith(".github/"):
        return "tests/ci/meta"
    if path.endswith(".md") or path.endswith(".ipynb"):
        return "docs/notebooks/scripts"
    return "tests/ci/meta"


def dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def checks_for_file(path: str, v55_ref_files: set[str]) -> list[str]:
    checks = ["archive_reference_guard"]
    if (
        path.startswith("src/")
        or path.startswith("config/")
        or path.startswith("colab_notebooks/")
        or path.startswith("scripts/")
        or path.startswith("docs/")
        or path in {"README.md", "PLAN.md", "AGENTS.md"}
    ):
        checks.append("legacy_token_guard_non_archive")
    if path.endswith(".py"):
        checks.append("py_compile_batch")
        if path.startswith("src/") or path.startswith("tests/") or path.startswith("config/"):
            checks.append("suite_unit_colab_integration")
        if path.startswith("src/router/"):
            checks.append("policy_regression_bundle")
        if path.startswith("scripts/") or "/" not in path:
            checks.append("python_sanity_bundle")
        if "/" not in path and path not in {"setup.py", "sitecustomize.py"}:
            checks.append("root_alias_contract")
    if path.startswith("config/") and ext(path) in {".json", ".yaml", ".yml"}:
        checks.extend(["config_parse", "suite_unit_validation"])
    if path.endswith(".ipynb"):
        checks.extend(["notebook_structure", "notebook_import_validation", "suite_colab_smoke"])
    if path.endswith(".md"):
        checks.append("markdown_links")
    if path in v55_ref_files:
        checks.append("missing_v55_report_reference_check")
    return dedupe(checks)


def run_internal_py_compile(root: Path, py_files: list[str]) -> tuple[dict, list[dict]]:
    failures: list[dict] = []
    ok = 0
    skipped_missing = 0
    for rel in py_files:
        full = root / rel
        if not full.exists():
            skipped_missing += 1
            continue
        try:
            source = full.read_text(encoding="utf-8")
            compile(source, rel, "exec")
            ok += 1
        except SyntaxError as exc:
            failures.append(
                {
                    "path": rel,
                    "line": int(exc.lineno or 1),
                    "message": str(exc.msg or "SyntaxError"),
                }
            )
        except Exception as exc:
            failures.append({"path": rel, "line": 1, "message": str(exc)})
    result = {
        "command": "internal: compile(source, path, 'exec')",
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "stdout": json.dumps(
            {
                "py_files": len(py_files),
                "compiled_ok": ok,
                "skipped_missing": skipped_missing,
                "failures": failures,
            },
            indent=2,
        )[:24000],
        "stderr": "",
        "notes": "In-process syntax validation.",
        "duration_sec": 0.0,
    }
    return result, failures


def run_internal_notebook_structure(root: Path, notebooks: list[str]) -> tuple[dict, list[dict]]:
    failures: list[dict] = []
    token_re = re.compile(LEGACY_PATTERN)
    for rel in notebooks:
        full = root / rel
        try:
            raw = full.read_text(encoding="utf-8-sig")
            payload = json.loads(raw)
            if not isinstance(payload.get("cells"), list):
                failures.append({"path": rel, "line": 1, "message": "cells is not a list"})
                continue
            if int(payload.get("nbformat", 0)) < 4:
                failures.append({"path": rel, "line": 1, "message": "nbformat < 4"})
                continue
            if token_re.search(raw):
                failures.append({"path": rel, "line": 1, "message": "legacy token found"})
            if "archive/v5_legacy" in raw:
                failures.append({"path": rel, "line": 1, "message": "archive path reference found"})
        except Exception as exc:
            failures.append({"path": rel, "line": 1, "message": f"notebook parse failed: {exc}"})
    result = {
        "command": "internal: notebook json + token/path guard checks",
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "stdout": json.dumps({"notebooks": len(notebooks), "failures": failures}, indent=2)[:24000],
        "stderr": "",
        "notes": "",
        "duration_sec": 0.0,
    }
    return result, failures


def run_internal_config_parse(root: Path, config_files: list[str]) -> tuple[dict, list[dict]]:
    failures: list[dict] = []
    parsed = 0
    yaml_mod = None
    try:
        import yaml as yaml_mod  # type: ignore[assignment]
    except Exception:
        yaml_mod = None
    for rel in config_files:
        full = root / rel
        try:
            if ext(rel) == ".json":
                json.loads(full.read_text(encoding="utf-8"))
            elif ext(rel) in {".yaml", ".yml"}:
                if yaml_mod is None:
                    raise RuntimeError("PyYAML is not installed")
                yaml_mod.safe_load(full.read_text(encoding="utf-8"))
            parsed += 1
        except Exception as exc:
            failures.append({"path": rel, "line": 1, "message": str(exc)})
    result = {
        "command": "internal: json.loads/yaml.safe_load for config files",
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "stdout": json.dumps({"configs": len(config_files), "parsed_ok": parsed, "failures": failures}, indent=2)[:24000],
        "stderr": "",
        "notes": "",
        "duration_sec": 0.0,
    }
    return result, failures


def run_internal_root_alias(active: list[str]) -> tuple[dict, list[dict]]:
    failures: list[dict] = []
    root_py = [p for p in active if "/" not in p and p.endswith(".py")]
    for rel in root_py:
        if rel in {"setup.py", "sitecustomize.py"}:
            continue
        if f"scripts/{rel}" not in active:
            failures.append({"path": rel, "line": 1, "message": f"missing scripts/{rel} counterpart"})
    result = {
        "command": "internal: root wrapper alias contract check",
        "status": "pass" if not failures else "fail",
        "returncode": 0 if not failures else 1,
        "stdout": json.dumps({"root_py_files": root_py, "failures": failures}, indent=2)[:24000],
        "stderr": "",
        "notes": "",
        "duration_sec": 0.0,
    }
    return result, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full v6 repository audit.")
    parser.add_argument("--manifest-out", default="plans/v6_repo_audit_manifest.json")
    parser.add_argument("--coverage-out", default="plans/v6_repo_audit_coverage.csv")
    parser.add_argument("--report-out", default="docs/reports/v6/V6_FULL_REPO_AUDIT.md")
    parser.add_argument("--commands-out", default="plans/v6_repo_audit_command_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / args.manifest_out
    coverage_path = root / args.coverage_out
    report_path = root / args.report_out
    commands_path = root / args.commands_out

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)

    tracked = git_ls_files(root)
    active = [p for p in tracked if "/archive/" not in p]
    archive = [p for p in tracked if "/archive/" in p]
    checks: dict[str, dict] = {}

    checks["snapshot_git_status"] = run_cmd(git_command(root, "status", "--short", "--branch"), cwd=root, timeout=120)
    checks["snapshot_git_diff"] = run_cmd(git_command(root, "diff", "--name-only"), cwd=root, timeout=120)
    checks["snapshot_git_log"] = run_cmd(git_command(root, "log", "--oneline", "-n", "8"), cwd=root, timeout=120)

    v55_scan = run_rg_no_match(
        [
            "rg",
            "-n",
            re.escape(MISSING_V55_PATH),
            "skills",
            "docs",
            "scripts",
            "src",
            "tests",
            "README.md",
            "PLAN.md",
            "AGENTS.md",
            "-g",
            "!**/archive/**",
            "-g",
            "!scripts/audit_v6_repo.py",
            "-g",
            "!docs/reports/v6/**",
        ],
        cwd=root,
        timeout=240,
    )
    checks["missing_v55_report_reference_scan"] = v55_scan
    v55_locs = parse_rg_locations(v55_scan["stdout"]) if v55_scan["status"] == "fail" else []
    v55_ref_files = {item["path"] for item in v55_locs}
    missing_v55 = not (root / MISSING_V55_PATH).exists()
    checks["missing_v55_report_reference_check"] = {
        "command": f"internal: exists({MISSING_V55_PATH}) + rg scan",
        "status": "fail" if (missing_v55 and len(v55_locs) > 0) else "pass",
        "returncode": 1 if (missing_v55 and len(v55_locs) > 0) else 0,
        "stdout": json.dumps({"missing_path": MISSING_V55_PATH, "missing": missing_v55, "references": v55_locs}, indent=2)[:24000],
        "stderr": "",
        "notes": "",
        "duration_sec": 0.0,
    }

    py_result, py_failures = run_internal_py_compile(root, [p for p in active if p.endswith(".py")])
    checks["py_compile_batch"] = py_result
    cfg_result, cfg_failures = run_internal_config_parse(
        root,
        [p for p in active if p.startswith("config/") and ext(p) in {".json", ".yaml", ".yml"}],
    )
    checks["config_parse"] = cfg_result
    nb_result, nb_failures = run_internal_notebook_structure(root, [p for p in active if p.endswith(".ipynb")])
    checks["notebook_structure"] = nb_result
    alias_result, alias_failures = run_internal_root_alias(active)
    checks["root_alias_contract"] = alias_result

    legacy_guard_command = rg_command(
        LEGACY_PATTERN,
        "src",
        "config",
        "colab_notebooks",
        "scripts",
        "docs",
        exclude_globs=LEGACY_TOKEN_GUARD_EXCLUDES,
    )
    checks["legacy_token_guard_non_archive"] = run_rg_no_match(legacy_guard_command, cwd=root, timeout=600)
    legacy_locs = parse_rg_locations(checks["legacy_token_guard_non_archive"]["stdout"]) if checks["legacy_token_guard_non_archive"]["status"] == "fail" else []

    checks["suite_unit_validation"] = run_cmd([sys.executable, "scripts/run_test_suites.py", "--suite", "unit/validation"], cwd=root, timeout=1800)
    checks["notebook_import_validation"] = run_cmd([sys.executable, "scripts/validate_notebook_imports.py"], cwd=root, timeout=1200)
    checks["suite_colab_smoke"] = run_cmd([sys.executable, "scripts/run_test_suites.py", "--suite", "colab/smoke"], cwd=root, timeout=1800)
    checks["markdown_links"] = run_cmd([sys.executable, "scripts/check_markdown_links.py", "--root", "."], cwd=root, timeout=1800)
    checks["python_sanity_bundle"] = run_cmd([sys.executable, "scripts/run_python_sanity_bundle.py"], cwd=root, timeout=1800)
    checks["suite_unit_colab_integration"] = run_cmd(
        [
            sys.executable,
            "scripts/run_test_suites.py",
            "--suite",
            "unit",
            "--suite",
            "colab",
            "--suite",
            "integration/core",
        ],
        cwd=root,
        timeout=7200,
    )
    checks["policy_regression_bundle"] = run_cmd([sys.executable, "scripts/run_policy_regression_bundle.py"], cwd=root, timeout=3600)
    archive_guard_command = [
        "rg",
        "-n",
        r"archive/v5_legacy|src/archive|docs/archive|tests/archive|plans/archive",
        "src",
        "tests",
        "scripts",
        "config",
        "colab_notebooks",
        "-g",
        "*.py",
        "-g",
        "*.ipynb",
        "-g",
        "*.json",
        "-g",
        "*.yaml",
        "-g",
        "*.yml",
    ]
    for glob in ARCHIVE_REFERENCE_GUARD_EXCLUDES:
        archive_guard_command.extend(["-g", glob])
    checks["archive_reference_guard"] = run_rg_no_match(archive_guard_command, cwd=root, timeout=600)
    archive_locs = parse_rg_locations(checks["archive_reference_guard"]["stdout"]) if checks["archive_reference_guard"]["status"] == "fail" else []

    manifest_files = []
    for path in tracked:
        scope = "archive" if "/archive/" in path else "active"
        manifest_files.append(
            {
                "path": path,
                "scope": scope,
                "extension": ext(path),
                "file_class": file_class(path),
                "owner_surface": owner_surface(path),
                "checks": checks_for_file(path, v55_ref_files=v55_ref_files) if scope == "active" else [],
            }
        )
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": {"tracked_total": len(tracked), "active_total": len(active), "archive_total": len(archive)},
        "files": manifest_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fail_files = {item["path"] for item in py_failures + cfg_failures + nb_failures + legacy_locs}
    risk_files = {item["path"] for item in archive_locs + v55_locs + alias_failures}

    command_risk_surfaces = {
        "suite_unit_validation": ["config/"],
        "notebook_import_validation": ["colab_notebooks/", "colab_bootstrap.ipynb"],
        "suite_colab_smoke": ["colab_notebooks/", "colab_bootstrap.ipynb"],
        "markdown_links": ["docs/", "README.md", "PLAN.md", "SECURITY.md"],
        "python_sanity_bundle": [
            "scripts/",
            "scripts/test_dynamic_taxonomy.py",
            "scripts/test_pipeline_final_check.py",
            "scripts/validate_notebook_imports.py",
        ],
        "suite_unit_colab_integration": ["src/", "tests/", "config/"],
        "policy_regression_bundle": ["src/router/", "config/"],
    }

    command_unverified_surfaces = {
        key: prefixes
        for key, prefixes in command_risk_surfaces.items()
        if checks[key]["status"] == "unverified"
    }
    command_failed_surfaces = {
        key: prefixes
        for key, prefixes in command_risk_surfaces.items()
        if checks[key]["status"] == "fail"
    }

    coverage_rows = []
    for item in manifest_files:
        if item["scope"] != "active":
            continue
        path = item["path"]
        file_checks = item["checks"]
        status = "pass"

        if path in fail_files:
            status = "fail"
        else:
            for prefixes in command_unverified_surfaces.values():
                if any(path.startswith(prefix) or path == prefix for prefix in prefixes):
                    status = "unverified"
                    break

        if status == "pass":
            if path in risk_files:
                status = "risk"
            else:
                for prefixes in command_failed_surfaces.values():
                    if any(path.startswith(prefix) or path == prefix for prefix in prefixes):
                        status = "risk"
                        break

        if status == "pass" and any(checks.get(cid, {"status": "pass"})["status"] == "unverified" for cid in file_checks):
            status = "unverified"

        coverage_rows.append(
            {
                "path": path,
                "owner_surface": item["owner_surface"],
                "file_class": item["file_class"],
                "status": status,
                "checks": "|".join(file_checks),
                "evidence_pointer": f"{args.commands_out}::{'|'.join(file_checks)}",
                "notes": "",
            }
        )

    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "owner_surface", "file_class", "status", "checks", "evidence_pointer", "notes"],
        )
        writer.writeheader()
        writer.writerows(coverage_rows)

    commands_path.write_text(
        json.dumps({"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checks": checks}, indent=2),
        encoding="utf-8",
    )

    confirmed_defects = []
    evidence_backed_risks = []
    bug_id = 1
    for group in (py_failures, cfg_failures, nb_failures):
        for item in group:
            confirmed_defects.append(
                {
                    "id": f"BUG-{bug_id:03d}",
                    "severity": "P1",
                    "location": f"{item['path']}:{item['line']}",
                    "claim": "File-level structural check failed.",
                    "evidence": item["message"],
                    "repro": f"{sys.executable} scripts/audit_v6_repo.py",
                    "expected_vs_actual": "Expected file to pass structural validation; failure detected.",
                    "confidence": "high",
                }
            )
            bug_id += 1
    for item in legacy_locs:
        confirmed_defects.append(
            {
                "id": f"BUG-{bug_id:03d}",
                "severity": "P1",
                "location": f"{item['path']}:{item['line']}",
                "claim": "Forbidden legacy token appears in non-archive file.",
                "evidence": item["snippet"],
                "repro": " ".join(legacy_guard_command),
                "expected_vs_actual": "Expected no legacy tokens in active surfaces; match found.",
                "confidence": "high",
            }
        )
        bug_id += 1

    risk_id = 1
    if checks["missing_v55_report_reference_check"]["status"] == "fail":
        v55_locations = [f"{item['path']}:{item['line']}" for item in v55_locs]
        evidence_backed_risks.append(
            {
                "id": f"RISK-{risk_id:03d}",
                "severity": "P2",
                "location": ", ".join(v55_locations) if v55_locations else "unknown:1",
                "claim": "Active instruction references missing non-archive v55 report path.",
                "direct_evidence": f"{MISSING_V55_PATH} is missing but referenced by active files.",
                "missing_proof": "Operational impact outside audit flow is not runtime-proven.",
                "next_verification_command": "rg -n \"docs/reports/v55/V55_FINAL_STATUS_REPORT.md\" skills docs scripts src tests README.md PLAN.md AGENTS.md -g \"!**/archive/**\" -g \"!scripts/audit_v6_repo.py\"",
                "confidence": "medium",
            }
        )
        risk_id += 1
    for item in archive_locs:
        evidence_backed_risks.append(
            {
                "id": f"RISK-{risk_id:03d}",
                "severity": "P2",
                "location": f"{item['path']}:{item['line']}",
                "claim": "Active file references archive surface.",
                "direct_evidence": item["snippet"],
                "missing_proof": "Runtime dependency impact not isolated in this sweep.",
                "next_verification_command": f"rg -n \"archive/v5_legacy|src/archive|docs/archive|tests/archive|plans/archive\" \"{item['path']}\"",
                "confidence": "medium",
            }
        )
        risk_id += 1
    for check_id in [
        "suite_unit_validation",
        "notebook_import_validation",
        "suite_colab_smoke",
        "markdown_links",
        "python_sanity_bundle",
        "suite_unit_colab_integration",
        "policy_regression_bundle",
    ]:
        if checks[check_id]["status"] != "pass":
            evidence_backed_risks.append(
                {
                    "id": f"RISK-{risk_id:03d}",
                    "severity": "P1",
                    "location": "scripts:1",
                    "claim": f"Audit command `{check_id}` did not pass.",
                    "direct_evidence": f"Exit={checks[check_id]['returncode']} notes={checks[check_id]['notes']}",
                    "missing_proof": "Root-cause path mapping is not isolated in this aggregate run.",
                    "next_verification_command": checks[check_id]["command"],
                    "confidence": "medium",
                }
            )
            risk_id += 1

    by_status = {name: len([row for row in coverage_rows if row["status"] == name]) for name in ["pass", "fail", "risk", "unverified"]}
    lines = [
        "# V6 Full Repository Audit",
        "",
        f"Generated at (UTC): `{dt.datetime.now(dt.timezone.utc).isoformat()}`",
        "",
        "## Baseline Snapshot",
        "",
        f"- Tracked files: `{len(tracked)}`",
        f"- Active files: `{len(active)}`",
        f"- Archive files: `{len(archive)}`",
        f"- Coverage rows: `{len(coverage_rows)}`",
        f"- Coverage summary: pass=`{by_status['pass']}` fail=`{by_status['fail']}` risk=`{by_status['risk']}` unverified=`{by_status['unverified']}`",
        "",
        "## Confirmed Defects",
        "",
    ]
    if not confirmed_defects:
        lines.append("- None.")
    else:
        for item in confirmed_defects:
            lines.extend(
                [
                    f"- `{item['id']}`",
                    f"  - severity: {item['severity']}",
                    f"  - location: {item['location']}",
                    f"  - claim: {item['claim']}",
                    f"  - evidence: {item['evidence']}",
                    f"  - repro: {item['repro']}",
                    f"  - expected_vs_actual: {item['expected_vs_actual']}",
                    f"  - confidence: {item['confidence']}",
                ]
            )
    lines.extend(["", "## Evidence-Backed Risks", ""])
    if not evidence_backed_risks:
        lines.append("- None.")
    else:
        for item in evidence_backed_risks:
            lines.extend(
                [
                    f"- `{item['id']}`",
                    f"  - severity: {item['severity']}",
                    f"  - location: {item['location']}",
                    f"  - claim: {item['claim']}",
                    f"  - direct_evidence: {item['direct_evidence']}",
                    f"  - missing_proof: {item['missing_proof']}",
                    f"  - next_verification_command: {item['next_verification_command']}",
                    f"  - confidence: {item['confidence']}",
                ]
            )
    lines.extend(["", "## Coverage Performed", ""])
    for cid, result in checks.items():
        lines.append(f"- `{cid}`: status={result['status']} returncode={result['returncode']} duration={result['duration_sec']:.1f}s")
    lines.extend(["", "## Unverified Areas", ""])
    unver = [cid for cid, result in checks.items() if result["status"] == "unverified"]
    if not unver:
        lines.append("- None.")
    else:
        for cid in unver:
            lines.append(f"- `{cid}`: {checks[cid]['notes'] or 'No additional details'}")
    lines.extend(["", "## Suggested Next Checks", ""])
    if not evidence_backed_risks:
        lines.append("- None.")
    else:
        for item in evidence_backed_risks:
            lines.append(f"- `{item['id']}`: `{item['next_verification_command']}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{args.manifest_out}`",
            f"- Coverage: `{args.coverage_out}`",
            f"- Command results: `{args.commands_out}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {args.manifest_out}")
    print(f"[OK] {args.coverage_out}")
    print(f"[OK] {args.report_out}")
    print(f"[OK] {args.commands_out}")
    print(
        f"[SUMMARY] pass={by_status['pass']} fail={by_status['fail']} "
        f"risk={by_status['risk']} unverified={by_status['unverified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
