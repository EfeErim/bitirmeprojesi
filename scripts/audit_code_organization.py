"""Audit repo-wide code organization boundaries.

This guard keeps the shared platform model explicit: durable code lives under
`src/`, operational wrappers live under `scripts/`, and notebooks stay thin.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(".runtime_tmp/code_organization_audit.json")


@dataclass(frozen=True)
class FileRecord:
    path: str
    category: str
    line_count: int


@dataclass(frozen=True)
class Finding:
    severity: str
    path: str
    message: str


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_python_files(root: Path) -> Iterable[Path]:
    for base in ("src", "scripts", "tests"):
        base_path = root / base
        if not base_path.exists():
            continue
        yield from sorted(
            path
            for path in base_path.rglob("*.py")
            if "__pycache__" not in path.parts
        )


def iter_notebooks(root: Path) -> Iterable[Path]:
    notebook_root = root / "colab_notebooks"
    if not notebook_root.exists():
        return
    yield from sorted(notebook_root.glob("*.ipynb"))


def categorize_path(path: Path, root: Path) -> str:
    rel = _relative(path, root)
    if rel.startswith("src/core/"):
        return "core"
    if rel.startswith("src/shared/"):
        return "shared"
    if rel.startswith("src/workflows/"):
        return "workflow"
    if rel.startswith("src/pipeline/"):
        return "runtime"
    if rel.startswith(("src/data/", "src/ood/", "src/router/", "src/adapter/")):
        return "domain"
    if rel.startswith("src/training/services/"):
        return "service"
    if rel.startswith("src/training/"):
        return "domain"
    if rel.startswith("src/app/"):
        return "cli"
    if rel.startswith("src/"):
        return "domain"
    if rel.startswith("scripts/notebook_cells/"):
        return "notebook_cell"
    if rel.startswith("scripts/notebook_helpers/"):
        return "notebook_helper"
    if rel.startswith(("scripts/validate_", "scripts/check_", "scripts/monitor_")):
        return "validation"
    if rel.startswith("scripts/"):
        return "cli"
    if rel.startswith("tests/"):
        return "test"
    if rel.startswith("colab_notebooks/"):
        return "notebook"
    return "other"


def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except UnicodeDecodeError:
        return len(path.read_text(encoding="utf-8-sig").splitlines())


def _imported_top_level_modules(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".", 1)[0])
    return modules


def _audit_import_boundaries(path: Path, root: Path) -> list[Finding]:
    rel = _relative(path, root)
    imports = set(_imported_top_level_modules(path))
    findings: list[Finding] = []
    if rel.startswith("src/"):
        forbidden = sorted(imports.intersection({"scripts", "tests", "colab_notebooks"}))
        for module in forbidden:
            findings.append(
                Finding(
                    severity="error",
                    path=rel,
                    message=f"`src` modules must not import `{module}`; move shared logic into `src` instead.",
                )
            )
    if rel.startswith("src/shared/"):
        forbidden_shared = sorted(imports.intersection({"torch", "transformers", "open_clip", "PIL"}))
        for module in forbidden_shared:
            findings.append(
                Finding(
                    severity="warning",
                    path=rel,
                    message=f"`src/shared` imports heavy/domain dependency `{module}`; keep shared utilities lightweight.",
                )
            )
    return findings


def _audit_size(path: Path, root: Path, line_count: int) -> list[Finding]:
    rel = _relative(path, root)
    category = categorize_path(path, root)
    findings: list[Finding] = []
    if category in {"cli", "notebook_cell", "notebook_helper", "validation"} and line_count > 800:
        findings.append(
            Finding(
                severity="warning",
                path=rel,
                message=(
                    f"{category} file has {line_count} lines; consider extracting reusable logic into `src` "
                    "or smaller testable helpers."
                ),
            )
        )
    return findings


def _audit_notebook(path: Path, root: Path, max_code_cell_lines: int) -> tuple[FileRecord, list[Finding]]:
    rel = _relative(path, root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return FileRecord(path=rel, category="notebook", line_count=0), [
            Finding(severity="error", path=rel, message=f"Notebook JSON could not be parsed: {exc}")
        ]
    code_lines = 0
    findings: list[Finding] = []
    for index, cell in enumerate(payload.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source_lines = source if isinstance(source, list) else str(source).splitlines()
        cell_line_count = len(source_lines)
        code_lines += cell_line_count
        if cell_line_count > max_code_cell_lines:
            findings.append(
                Finding(
                    severity="warning",
                    path=rel,
                    message=(
                        f"Code cell {index} has {cell_line_count} lines; move durable logic into "
                        "`scripts/notebook_helpers` or `src`."
                    ),
                )
            )
    return FileRecord(path=rel, category="notebook", line_count=code_lines), findings


def build_report(root: Path = REPO_ROOT, *, max_code_cell_lines: int = 120) -> dict[str, Any]:
    root = root.resolve()
    files: list[FileRecord] = []
    findings: list[Finding] = []

    for path in iter_python_files(root):
        line_count = _line_count(path)
        files.append(FileRecord(path=_relative(path, root), category=categorize_path(path, root), line_count=line_count))
        findings.extend(_audit_import_boundaries(path, root))
        findings.extend(_audit_size(path, root, line_count))

    for path in iter_notebooks(root):
        record, notebook_findings = _audit_notebook(path, root, max_code_cell_lines)
        files.append(record)
        findings.extend(notebook_findings)

    category_counts: dict[str, int] = {}
    for record in files:
        category_counts[record.category] = category_counts.get(record.category, 0) + 1
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "status": "fail" if error_count else "pass",
        "file_count": len(files),
        "category_counts": dict(sorted(category_counts.items())),
        "error_count": error_count,
        "warning_count": warning_count,
        "files": [asdict(record) for record in files],
        "findings": [asdict(finding) for finding in findings],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-code-cell-lines", type=int, default=120)
    parser.add_argument("--no-write", action="store_true", help="Print the summary without writing the JSON report.")
    args = parser.parse_args(argv)

    report = build_report(args.root, max_code_cell_lines=args.max_code_cell_lines)
    if not args.no_write:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = args.root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        "code_organization "
        f"status={report['status']} files={report['file_count']} "
        f"errors={report['error_count']} warnings={report['warning_count']}"
    )
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
