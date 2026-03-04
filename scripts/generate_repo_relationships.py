#!/usr/bin/env python3
"""Generate repository relationship artifacts.

Governance intent:
- `docs/REPO_FILE_RELATIONS.md` is the canonical human-maintained summary.
- `docs/REPO_FILE_RELATIONS_DETAILED.md` is a generated detailed artifact.
- This script is the canonical owner for generating the detailed artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple


INTERNAL_IMPORT_ROOTS = ("src", "tests", "scripts", "config")
CANONICAL_RELATIONS_SUMMARY_DOC = "docs/REPO_FILE_RELATIONS.md"
CANONICAL_RELATIONS_DETAILED_DOC = "docs/REPO_FILE_RELATIONS_DETAILED.md"
CANONICAL_RELATIONS_JSON_SNAPSHOT = "docs/reports/repository_relationships_snapshot.json"
LOW_SIGNAL_EXTENSIONS = {".pdf", ".zip", ".log"}
TEXT_FRIENDLY_EXTENSIONS = {
    ".py",
    ".md",
    ".ipynb",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".ini",
    ".sh",
    ".tex",
    ".log",
    ".example",
    ".gitignore",
    ".coveragerc",
    ".gitattributes",
    "",
}


@dataclass(frozen=True, order=True)
class Edge:
    target: str
    relation_type: str
    confidence: str
    evidence: str


@dataclass(frozen=True, order=True)
class IncomingEdge:
    source: str
    relation_type: str
    confidence: str
    evidence: str


@dataclass
class FileNode:
    path: str
    type: str
    size: int
    hash: str
    category: str
    purpose: str
    outgoing_edges: List[Edge] = field(default_factory=list)
    incoming_edges: List[IncomingEdge] = field(default_factory=list)


class GraphBuilder:
    """Mutable graph helper with deduplicated edge insertion."""

    def __init__(self, nodes: Dict[str, FileNode]) -> None:
        self.nodes = nodes
        self._adj: Dict[str, Set[Edge]] = defaultdict(set)

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        confidence: str,
        evidence: str,
    ) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        if source == target and relation_type != "mirror_of":
            return
        edge = Edge(
            target=target,
            relation_type=relation_type,
            confidence="explicit" if confidence == "explicit" else "inferred",
            evidence=evidence.strip()[:300],
        )
        self._adj[source].add(edge)

    def finalize(self) -> None:
        incoming: Dict[str, Set[IncomingEdge]] = defaultdict(set)
        for source, edges in self._adj.items():
            ordered_edges = sorted(edges, key=lambda e: (e.target, e.relation_type, e.confidence, e.evidence))
            self.nodes[source].outgoing_edges = ordered_edges
            for edge in ordered_edges:
                incoming[edge.target].add(
                    IncomingEdge(
                        source=source,
                        relation_type=edge.relation_type,
                        confidence=edge.confidence,
                        evidence=edge.evidence,
                    )
                )
        for path, node in self.nodes.items():
            node.incoming_edges = sorted(
                incoming.get(path, set()),
                key=lambda e: (e.source, e.relation_type, e.confidence, e.evidence),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate repository file relationship map (detailed generated artifact). "
            f"Human-maintained summary remains {CANONICAL_RELATIONS_SUMMARY_DOC}."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root to scan (default: current directory).")
    parser.add_argument(
        "--output",
        default=CANONICAL_RELATIONS_DETAILED_DOC,
        help=(
            "Output markdown path for generated detailed artifact "
            f"(default: {CANONICAL_RELATIONS_DETAILED_DOC})."
        ),
    )
    parser.add_argument(
        "--json-output",
        default="",
        help=(
            "Optional JSON snapshot path (default: disabled; canonical optional path: "
            f"{CANONICAL_RELATIONS_JSON_SNAPSHOT})."
        ),
    )
    parser.add_argument(
        "--exclude-dirs",
        default=".venv,.git",
        help="Comma-separated directory names to exclude everywhere in the tree.",
    )
    parser.add_argument(
        "--deep-infer",
        dest="deep_infer",
        action="store_true",
        default=True,
        help="Enable deep inference pass for low-signal files (default: enabled).",
    )
    parser.add_argument(
        "--no-deep-infer",
        dest="deep_infer",
        action="store_false",
        help="Disable deep inference pass.",
    )
    return parser.parse_args()


def to_posix_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_files(root: Path, exclude_dirs: Set[str]) -> Iterator[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in exclude_dirs)
        current_path = Path(current)
        if any(part in exclude_dirs for part in current_path.relative_to(root).parts):
            continue
        for filename in sorted(files):
            file_path = current_path / filename
            rel_parts = file_path.relative_to(root).parts
            if any(part in exclude_dirs for part in rel_parts):
                continue
            yield file_path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_probably_text(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in TEXT_FRIENDLY_EXTENSIONS:
        return True
    if ext in LOW_SIGNAL_EXTENSIONS:
        return False
    try:
        with path.open("rb") as f:
            sample = f.read(2048)
    except OSError:
        return False
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    non_printable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return non_printable / max(len(sample), 1) < 0.15


def read_text_safe(path: Path) -> Optional[str]:
    if not is_probably_text(path):
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def infer_category(rel_path: str) -> str:
    parts = rel_path.split("/")
    return parts[0] if len(parts) > 1 else "root"


def first_markdown_heading(text: str) -> Optional[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


def extract_py_docstring(path: Path) -> Optional[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    first = doc.strip().splitlines()[0].strip()
    return first[:160] if first else None


def extract_ipynb_heading(path: Path) -> Optional[str]:
    text = read_text_safe(path)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        heading = first_markdown_heading(source)
        if heading:
            return heading[:160]
    return None


def infer_purpose(path: Path, rel_path: str) -> str:
    ext = path.suffix.lower()
    category = infer_category(rel_path)
    if rel_path == "README.md":
        return "Primary project overview and quick-start entrypoint."
    if category == "src":
        doc = extract_py_docstring(path) if ext == ".py" else None
        return doc or "Core runtime source module for training/inference pipeline."
    if category == "tests":
        doc = extract_py_docstring(path) if ext == ".py" else None
        return doc or "Automated test surface covering behavior, regressions, or integration."
    if category == "scripts":
        doc = extract_py_docstring(path) if ext == ".py" else None
        return doc or "Operational utility script used for setup, checks, or diagnostics."
    if category == "docs":
        text = read_text_safe(path)
        heading = first_markdown_heading(text) if text else None
        return (heading + ".") if heading else "Documentation artifact."
    if category == "colab_notebooks":
        heading = extract_ipynb_heading(path)
        return (heading + ".") if heading else "Colab notebook workflow artifact."
    if category == "config":
        return "Configuration contract or runtime settings file."
    if category == "skills":
        return "Project-local agent skill definition or reference asset."
    if category == "plans":
        return "Planning artifact for implementation sequencing and status."
    if category == "logs":
        return "Execution log artifact."
    if ext == ".pdf":
        return "Binary documentation or report artifact (PDF)."
    if ext == ".zip":
        return "Archived binary artifact."
    if ext == ".py":
        doc = extract_py_docstring(path)
        return doc or "Python utility or compatibility surface."
    if ext == ".md":
        text = read_text_safe(path)
        heading = first_markdown_heading(text) if text else None
        return (heading + ".") if heading else "Markdown documentation artifact."
    return "Repository file artifact."


def build_inventory(
    root: Path,
    exclude_dirs: Set[str],
    stable_generated_paths: Optional[Set[str]] = None,
) -> Dict[str, FileNode]:
    stable_generated_paths = stable_generated_paths or set()
    nodes: Dict[str, FileNode] = {}
    for file_path in iter_files(root, exclude_dirs):
        rel = to_posix_rel(file_path, root)
        ext = file_path.suffix.lower() or "<noext>"
        digest = (
            "0" * 64
            if rel in stable_generated_paths
            else file_hash(file_path)
        )
        nodes[rel] = FileNode(
            path=rel,
            type=ext,
            size=file_path.stat().st_size,
            hash=digest,
            category=infer_category(rel),
            purpose=infer_purpose(file_path, rel),
        )
    return dict(sorted(nodes.items(), key=lambda kv: kv[0]))


def build_module_index(nodes: Dict[str, FileNode]) -> Dict[str, str]:
    module_to_path: Dict[str, str] = {}
    for rel_path in nodes:
        if not rel_path.endswith(".py"):
            continue
        no_ext = rel_path[:-3]
        parts = no_ext.split("/")
        module_name = ".".join(parts[:-1]) if parts[-1] == "__init__" else ".".join(parts)
        module_to_path[module_name] = rel_path
    return module_to_path


def resolve_module(module_name: str, module_to_path: Dict[str, str]) -> Optional[str]:
    current = module_name
    while current:
        if current in module_to_path:
            return module_to_path[current]
        if "." not in current:
            break
        current = current.rsplit(".", 1)[0]
    return None


def resolve_relative_import_module(source_path: str, module: Optional[str], level: int) -> Optional[str]:
    if level <= 0:
        return module
    source_parts = source_path[:-3].split("/")
    package_parts = source_parts[:-1]
    base_len = len(package_parts) - (level - 1)
    if base_len <= 0:
        return None
    base_parts = package_parts[:base_len]
    return ".".join(base_parts + module.split(".")) if module else ".".join(base_parts)


def resolve_relative_path_reference(raw_target: str, source_path: str, nodes: Dict[str, FileNode]) -> Optional[str]:
    cleaned = raw_target.strip().strip("'\"")
    cleaned = cleaned.split("#", 1)[0].split("?", 1)[0].strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("\\", "/")
    if cleaned.startswith("file://"):
        cleaned = cleaned[len("file://") :]
    candidates: List[str] = []
    if cleaned in nodes:
        candidates.append(cleaned)
    source_parent = Path(source_path).parent
    rel_candidate = (source_parent / cleaned).as_posix()
    if rel_candidate in nodes:
        candidates.append(rel_candidate)
    trimmed = cleaned.lstrip("./")
    if trimmed in nodes:
        candidates.append(trimmed)
    for candidate in candidates:
        if candidate in nodes:
            return candidate
    return None


def extract_python_structure(
    root: Path,
    nodes: Dict[str, FileNode],
    graph: GraphBuilder,
    module_to_path: Dict[str, str],
    warnings: List[str],
) -> None:
    wrapper_location_re = re.compile(r'Canonical script location:\s*([^\s]+)')
    wrapper_join_re = re.compile(r'["\']scripts["\']\s*/\s*["\']([^"\']+\.py)["\']')

    for rel_path in nodes:
        if not rel_path.endswith(".py"):
            continue
        text = read_text_safe(root / rel_path) or ""
        try:
            tree = ast.parse(text)
        except Exception as exc:
            warnings.append(f"python-parse-failed:{rel_path}:{exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod.split(".", 1)[0] not in INTERNAL_IMPORT_ROOTS:
                        continue
                    target = resolve_module(mod, module_to_path)
                    if target:
                        graph.add_edge(rel_path, target, "imports", "explicit", f"import {mod}")
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module
                if node.level > 0:
                    module_name = resolve_relative_import_module(rel_path, node.module, node.level)
                if not module_name or module_name.split(".", 1)[0] not in INTERNAL_IMPORT_ROOTS:
                    continue

                base_target = resolve_module(module_name, module_to_path)
                if base_target:
                    graph.add_edge(
                        rel_path,
                        base_target,
                        "imports",
                        "explicit",
                        f"from {module_name} import ...",
                    )

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    submodule = f"{module_name}.{alias.name}"
                    sub_target = resolve_module(submodule, module_to_path)
                    if sub_target:
                        graph.add_edge(
                            rel_path,
                            sub_target,
                            "imports",
                            "explicit",
                            f"from {module_name} import {alias.name}",
                        )

        canonical_match = wrapper_location_re.search(text)
        if canonical_match:
            target = resolve_relative_path_reference(canonical_match.group(1), rel_path, nodes)
            if target:
                graph.add_edge(
                    rel_path,
                    target,
                    "compatibility_alias",
                    "explicit",
                    "canonical script location marker",
                )

        if "runpy.run_path" in text:
            for match in wrapper_join_re.finditer(text):
                target = resolve_relative_path_reference(f"scripts/{match.group(1)}", rel_path, nodes)
                if target:
                    graph.add_edge(
                        rel_path,
                        target,
                        "delegates_to_script",
                        "explicit",
                        "runpy.run_path wrapper dispatch",
                    )


def extract_notebook_relationships(
    root: Path,
    nodes: Dict[str, FileNode],
    graph: GraphBuilder,
    module_to_path: Dict[str, str],
    warnings: List[str],
) -> None:
    from_re = re.compile(r"\bfrom\s+(src(?:\.[A-Za-z_]\w*)+)\s+import\s+([A-Za-z0-9_*,\s]+)")
    import_re = re.compile(r"\bimport\s+(src(?:\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))")
    run_re = re.compile(r"^\s*%run\s+([^\s]+)", re.MULTILINE)
    path_re = re.compile(r"(?:\.\.?/)?(?:src|scripts|config|docs|tests|colab_notebooks)/[\w./\\-]+")

    for rel_path in nodes:
        if not rel_path.endswith(".ipynb"):
            continue
        text = read_text_safe(root / rel_path)
        if text is None:
            continue
        try:
            payload = json.loads(text)
        except Exception as exc:
            warnings.append(f"notebook-parse-failed:{rel_path}:{exc}")
            continue

        chunks: List[str] = []
        for cell in payload.get("cells", []):
            if not isinstance(cell, dict):
                continue
            source = cell.get("source", [])
            chunks.append("".join(source) if isinstance(source, list) else str(source))
        merged = "\n".join(chunks)

        for match in from_re.finditer(merged):
            module_name = match.group(1)
            target = resolve_module(module_name, module_to_path)
            if target:
                graph.add_edge(rel_path, target, "imports", "explicit", f"notebook from-import {module_name}")
            for imported_name in [n.strip() for n in match.group(2).split(",")]:
                if not imported_name or imported_name == "*":
                    continue
                submodule = f"{module_name}.{imported_name}"
                sub_target = resolve_module(submodule, module_to_path)
                if sub_target:
                    graph.add_edge(rel_path, sub_target, "imports", "explicit", f"notebook from-import {submodule}")

        for match in import_re.finditer(merged):
            module_name = match.group(1)
            target = resolve_module(module_name, module_to_path)
            if target:
                graph.add_edge(rel_path, target, "imports", "explicit", f"notebook import {module_name}")

        for match in run_re.finditer(merged):
            target = resolve_relative_path_reference(match.group(1), rel_path, nodes)
            if target:
                graph.add_edge(
                    rel_path,
                    target,
                    "runs_notebook_or_script",
                    "explicit",
                    f"%run {match.group(1)}",
                )

        for match in path_re.finditer(merged):
            target = resolve_relative_path_reference(match.group(0), rel_path, nodes)
            if target:
                graph.add_edge(
                    rel_path,
                    target,
                    "references_path",
                    "explicit",
                    f"path reference {match.group(0)}",
                )


def extract_markdown_links(root: Path, nodes: Dict[str, FileNode], graph: GraphBuilder) -> None:
    inline_link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    ref_link_re = re.compile(r"^\[[^\]]+\]:\s*([^\s]+)", re.MULTILINE)

    for rel_path in nodes:
        if not rel_path.endswith(".md"):
            continue
        text = read_text_safe(root / rel_path)
        if not text:
            continue

        for match in inline_link_re.finditer(text):
            raw = match.group(1).strip()
            if raw.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = resolve_relative_path_reference(raw, rel_path, nodes)
            if target:
                graph.add_edge(rel_path, target, "links_to", "explicit", f"markdown link {raw}")

        for match in ref_link_re.finditer(text):
            raw = match.group(1).strip()
            if raw.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = resolve_relative_path_reference(raw, rel_path, nodes)
            if target:
                graph.add_edge(rel_path, target, "links_to", "explicit", f"reference link {raw}")


def extract_workflow_relationships(root: Path, nodes: Dict[str, FileNode], graph: GraphBuilder) -> None:
    workflow_files = [
        rel
        for rel in nodes
        if rel.startswith(".github/workflows/") and (rel.endswith(".yml") or rel.endswith(".yaml"))
    ]
    python_cmd_re = re.compile(r"\bpython\s+([^\s\"'`;|&]+)")
    pytest_cmd_re = re.compile(r"\bpytest\s+([^\s\"'`;|&]+)")

    for rel_path in workflow_files:
        text = read_text_safe(root / rel_path)
        if not text:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            for match in python_cmd_re.finditer(stripped):
                arg = match.group(1).strip()
                if not arg.endswith(".py"):
                    continue
                target = resolve_relative_path_reference(arg, rel_path, nodes)
                if target:
                    graph.add_edge(rel_path, target, "invokes", "explicit", f"workflow command: {stripped}")

            for match in pytest_cmd_re.finditer(stripped):
                arg = match.group(1).strip()
                target = resolve_relative_path_reference(arg, rel_path, nodes)
                if target:
                    graph.add_edge(rel_path, target, "invokes_tests", "explicit", f"workflow command: {stripped}")
                    continue
                full_dir = root / arg
                if full_dir.exists() and full_dir.is_dir():
                    for test_file in sorted(full_dir.rglob("test_*.py")):
                        rel_test = to_posix_rel(test_file, root)
                        if rel_test in nodes:
                            graph.add_edge(
                                rel_path,
                                rel_test,
                                "invokes_tests",
                                "explicit",
                                f"workflow command: {stripped}",
                            )


def extract_config_usage(root: Path, nodes: Dict[str, FileNode], graph: GraphBuilder) -> None:
    config_paths = [p for p in nodes if p.startswith("config/")]
    if not config_paths:
        return
    text_cache: Dict[str, Optional[str]] = {rel: read_text_safe(root / rel) for rel in nodes}
    for source, text in text_cache.items():
        if not text:
            continue
        for cfg in config_paths:
            if source == cfg:
                continue
            if cfg in text or cfg.replace("/", "\\") in text:
                graph.add_edge(source, cfg, "uses_config", "explicit", f"mentions config path {cfg}")


def extract_test_coverage_mapping(
    root: Path,
    nodes: Dict[str, FileNode],
    graph: GraphBuilder,
    module_to_path: Dict[str, str],
    warnings: List[str],
) -> None:
    src_file_paths = [p for p in nodes if p.startswith("src/") and p.endswith(".py")]
    for rel_path in nodes:
        if not rel_path.startswith("tests/") or not rel_path.endswith(".py"):
            continue
        text = read_text_safe(root / rel_path) or ""
        try:
            tree = ast.parse(text)
        except Exception as exc:
            warnings.append(f"test-parse-failed:{rel_path}:{exc}")
            continue

        explicit_targets: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod.startswith("src."):
                        target = resolve_module(mod, module_to_path)
                        if target:
                            explicit_targets.add(target)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module
                if mod and mod.startswith("src."):
                    target = resolve_module(mod, module_to_path)
                    if target:
                        explicit_targets.add(target)
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        sub_target = resolve_module(f"{mod}.{alias.name}", module_to_path)
                        if sub_target:
                            explicit_targets.add(sub_target)

        for target in sorted(explicit_targets):
            graph.add_edge(rel_path, target, "validates", "explicit", "test imports target module")

        parts = rel_path.split("/")
        area_hint: Optional[str] = None
        if len(parts) >= 4 and parts[0] == "tests" and not parts[2].startswith("test_"):
            area_hint = parts[2]
        if area_hint:
            area_targets = [p for p in src_file_paths if p.startswith(f"src/{area_hint}/")]
            for target in area_targets:
                graph.add_edge(
                    rel_path,
                    target,
                    "validates",
                    "inferred",
                    f"area mapping tests/{parts[1]}/{area_hint} -> src/{area_hint}",
                )
            continue

        stem_tokens = [tok for tok in Path(rel_path).stem.replace("test_", "").split("_") if len(tok) >= 4]
        for token in stem_tokens:
            candidates = [src for src in src_file_paths if token in Path(src).stem.lower()]
            if 0 < len(candidates) <= 5:
                for target in candidates:
                    graph.add_edge(
                        rel_path,
                        target,
                        "validates",
                        "inferred",
                        f"filename token match '{token}'",
                    )


def extract_deep_inference(root: Path, nodes: Dict[str, FileNode], graph: GraphBuilder) -> None:
    root_readme = "README.md" if "README.md" in nodes else None
    text_cache: Dict[str, Optional[str]] = {rel: read_text_safe(root / rel) for rel in nodes}

    def has_explicit_links(rel_path: str) -> bool:
        node = nodes[rel_path]
        return any(e.confidence == "explicit" for e in node.outgoing_edges) or any(
            e.confidence == "explicit" for e in node.incoming_edges
        )

    low_signal_paths = [p for p, node in nodes.items() if node.type in LOW_SIGNAL_EXTENSIONS or node.category == "logs"]
    for artifact in low_signal_paths:
        basename = Path(artifact).name
        for source, text in text_cache.items():
            if not text or source == artifact:
                continue
            if basename in text:
                graph.add_edge(
                    source,
                    artifact,
                    "mentions_artifact",
                    "inferred",
                    f"text mention of '{basename}'",
                )

        if has_explicit_links(artifact):
            continue
        top = artifact.split("/")[0] if "/" in artifact else "root"
        owner_candidates = [f"{top}/README.md", "docs/README.md", root_readme or ""]
        for candidate in owner_candidates:
            if candidate and candidate in nodes and candidate != artifact:
                graph.add_edge(
                    artifact,
                    candidate,
                    "owned_by_context",
                    "inferred",
                    f"context owner fallback for low-signal artifact in '{top}'",
                )
                break


def extract_mirror_and_aliases(root: Path, nodes: Dict[str, FileNode], graph: GraphBuilder) -> None:
    hash_groups: Dict[str, List[str]] = defaultdict(list)
    for path, node in nodes.items():
        hash_groups[node.hash].append(path)

    for digest, members in sorted(hash_groups.items(), key=lambda kv: kv[0]):
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                graph.add_edge(left, right, "mirror_of", "explicit", f"sha256 match {digest[:12]}")
                graph.add_edge(right, left, "mirror_of", "explicit", f"sha256 match {digest[:12]}")

    if "colab_bootstrap.ipynb" in nodes and "colab_notebooks/colab_bootstrap.ipynb" in nodes:
        if nodes["colab_bootstrap.ipynb"].hash == nodes["colab_notebooks/colab_bootstrap.ipynb"].hash:
            graph.add_edge(
                "colab_bootstrap.ipynb",
                "colab_notebooks/colab_bootstrap.ipynb",
                "compatibility_mirror",
                "explicit",
                "root mirror of canonical notebook",
            )
            graph.add_edge(
                "colab_notebooks/colab_bootstrap.ipynb",
                "colab_bootstrap.ipynb",
                "compatibility_mirror",
                "explicit",
                "canonical notebook mirrored to root compatibility path",
            )

    mirror = "colab_notebooks/requirements_colab.txt"
    canonical = "requirements_colab.txt"
    if mirror in nodes and canonical in nodes:
        text = read_text_safe(root / mirror) or ""
        if "-r ../requirements_colab.txt" in text:
            graph.add_edge(
                mirror,
                canonical,
                "compatibility_alias",
                "explicit",
                "mirror requirements file includes canonical dependency list",
            )


def confidence_counter(nodes: Dict[str, FileNode]) -> Counter:
    counts: Counter = Counter()
    for node in nodes.values():
        for edge in node.outgoing_edges:
            counts[edge.confidence] += 1
    return counts


def relation_counter(nodes: Dict[str, FileNode]) -> Counter:
    counts: Counter = Counter()
    for node in nodes.values():
        for edge in node.outgoing_edges:
            counts[edge.relation_type] += 1
    return counts


def category_counter(nodes: Dict[str, FileNode]) -> Counter:
    counts: Counter = Counter()
    for node in nodes.values():
        counts[node.category] += 1
    return counts


def type_counter(nodes: Dict[str, FileNode]) -> Counter:
    counts: Counter = Counter()
    for node in nodes.values():
        counts[node.type] += 1
    return counts


def build_directory_matrix(nodes: Dict[str, FileNode]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for node in nodes.values():
        for edge in node.outgoing_edges:
            matrix[node.category][nodes[edge.target].category] += 1
    return {
        src: {dst: matrix[src][dst] for dst in sorted(matrix[src])}
        for src in sorted(matrix)
    }


def top_relationship_edges(matrix: Dict[str, Dict[str, int]], limit: int = 18) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    for src, row in matrix.items():
        for dst, count in row.items():
            if src != dst and count > 0:
                rows.append((src, dst, count))
    rows.sort(key=lambda t: (-t[2], t[0], t[1]))
    return rows[:limit]


def format_human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{int(value)} {units[idx]}" if idx == 0 else f"{value:.2f} {units[idx]}"


def format_markdown(
    nodes: Dict[str, FileNode],
    matrix: Dict[str, Dict[str, int]],
    exclude_dirs: Sequence[str],
    deep_infer_enabled: bool,
    warnings: Sequence[str],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    type_counts = type_counter(nodes)
    cat_counts = category_counter(nodes)
    relation_counts = relation_counter(nodes)
    conf_counts = confidence_counter(nodes)
    matrix_categories = sorted(cat_counts.keys())
    top_edges = top_relationship_edges(matrix)

    lines: List[str] = [
        "# Repository File Relationships (Detailed)",
        "",
        f"- Generated at (UTC): `{generated_at}`",
        "- Scan root: `.`",
        f"- Excluded directories: `{','.join(exclude_dirs)}`",
        f"- Deep inference: `{'enabled' if deep_infer_enabled else 'disabled'}`",
        f"- Total files scanned: `{len(nodes)}`",
        "",
        "## Legend",
        "",
        "- `relation_type`: semantic connection from source file to target file.",
        "- `confidence=explicit`: parsed from direct syntax/link/command/path evidence.",
        "- `confidence=inferred`: heuristic or contextual relationship.",
        "",
        "## Global Metrics",
        "",
        "### File Counts by Type",
        "",
        "| Type | Count |",
        "|---|---:|",
    ]
    for file_type, count in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{file_type}` | {count} |")

    lines += ["", "### File Counts by Category", "", "| Category | Count |", "|---|---:|"]
    for category, count in sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{category}` | {count} |")

    lines += ["", "### Edge Counts by Relation Type", "", "| Relation Type | Count |", "|---|---:|"]
    for relation, count in sorted(relation_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{relation}` | {count} |")

    lines += ["", "### Edge Counts by Confidence", "", "| Confidence | Count |", "|---|---:|"]
    for confidence, count in sorted(conf_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{confidence}` | {count} |")

    lines += ["", "## High-Level Relationship Map", "", "```mermaid", "graph LR"]
    node_labels: Dict[str, str] = {}
    for category in matrix_categories:
        node_id = re.sub(r"[^A-Za-z0-9_]", "_", category.upper()) or "ROOT"
        if node_id[0].isdigit():
            node_id = f"N_{node_id}"
        node_labels[category] = node_id
        lines.append(f"  {node_id}[{category}]")
    for src, dst, count in top_edges:
        lines.append(f"  {node_labels[src]} -->|{count}| {node_labels[dst]}")
    lines += ["```", "", "## Directory-Level Relationship Matrix", ""]

    lines.append("| Source \\ Target | " + " | ".join(f"`{c}`" for c in matrix_categories) + " |")
    lines.append("|---|" + "|".join("---:" for _ in matrix_categories) + "|")
    for src in matrix_categories:
        row = [str(matrix.get(src, {}).get(dst, 0)) for dst in matrix_categories]
        lines.append(f"| `{src}` | " + " | ".join(row) + " |")

    lines += ["", "## Per-File Relationship Catalog", ""]

    weak_or_unresolved: List[str] = []
    for rel_path, node in nodes.items():
        lines += [
            f"### `{rel_path}`",
            "",
            f"- `type`: `{node.type}`",
            f"- `size`: `{node.size}` bytes ({format_human_size(node.size)})",
            f"- `sha256`: `{node.hash}`",
            f"- `category`: `{node.category}`",
            f"- `purpose`: {node.purpose}",
            "",
        ]

        out_explicit = sum(1 for e in node.outgoing_edges if e.confidence == "explicit")
        out_inferred = sum(1 for e in node.outgoing_edges if e.confidence == "inferred")
        lines.append(f"- `outgoing_edges`: {len(node.outgoing_edges)} (explicit={out_explicit}, inferred={out_inferred})")
        if node.outgoing_edges:
            for edge in node.outgoing_edges:
                lines.append(
                    f"  - [`{edge.relation_type}` | `{edge.confidence}`] -> `{edge.target}` | evidence: {edge.evidence}"
                )
        else:
            lines.append("  - none")
        lines.append("")

        in_explicit = sum(1 for e in node.incoming_edges if e.confidence == "explicit")
        in_inferred = sum(1 for e in node.incoming_edges if e.confidence == "inferred")
        lines.append(f"- `incoming_edges`: {len(node.incoming_edges)} (explicit={in_explicit}, inferred={in_inferred})")
        if node.incoming_edges:
            for edge in node.incoming_edges:
                lines.append(
                    f"  - [`{edge.relation_type}` | `{edge.confidence}`] <- `{edge.source}` | evidence: {edge.evidence}"
                )
        else:
            lines.append("  - none")
        lines.append("")

        confidences = [e.confidence for e in node.outgoing_edges] + [e.confidence for e in node.incoming_edges]
        if not confidences or all(conf == "inferred" for conf in confidences):
            weak_or_unresolved.append(rel_path)

    lines += ["## Unresolved or Weakly Linked Files", ""]
    if weak_or_unresolved:
        lines.extend(f"- `{path}`" for path in weak_or_unresolved)
    else:
        lines.append("- none")

    lines += [
        "",
        "## Regeneration",
        "",
        "```bash",
        "python scripts/generate_repo_relationships.py --output docs/REPO_FILE_RELATIONS_DETAILED.md "
        "--json-output docs/reports/repository_relationships_snapshot.json",
        "```",
        "",
        "## Parser Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- `{w}`" for w in warnings)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def to_json_payload(
    nodes: Dict[str, FileNode],
    exclude_dirs: Sequence[str],
    deep_infer_enabled: bool,
) -> Dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": {
            "root": ".",
            "exclude_dirs": list(exclude_dirs),
            "deep_infer": deep_infer_enabled,
            "total_files": len(nodes),
        },
        "metrics": {
            "type_counts": dict(type_counter(nodes)),
            "category_counts": dict(category_counter(nodes)),
            "relation_type_counts": dict(relation_counter(nodes)),
            "confidence_counts": dict(confidence_counter(nodes)),
        },
        "files": [
            {
                "path": node.path,
                "type": node.type,
                "size": node.size,
                "hash": node.hash,
                "category": node.category,
                "purpose": node.purpose,
                "outgoing_edges": [asdict(edge) for edge in node.outgoing_edges],
                "incoming_edges": [asdict(edge) for edge in node.incoming_edges],
            }
            for _, node in nodes.items()
        ],
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = (root / args.output).resolve()
    json_path = (root / args.json_output).resolve() if args.json_output else None
    exclude_dirs = [d.strip() for d in args.exclude_dirs.split(",") if d.strip()]
    exclude_set = set(exclude_dirs)

    stable_generated_paths: Set[str] = set()
    if output_path.is_relative_to(root):
        stable_generated_paths.add(output_path.relative_to(root).as_posix())
    if json_path and json_path.is_relative_to(root):
        stable_generated_paths.add(json_path.relative_to(root).as_posix())

    nodes = build_inventory(root, exclude_set, stable_generated_paths)
    module_to_path = build_module_index(nodes)
    graph = GraphBuilder(nodes)
    warnings: List[str] = []

    # Ordered extraction passes aligned with the implementation plan.
    extract_python_structure(root, nodes, graph, module_to_path, warnings)
    extract_notebook_relationships(root, nodes, graph, module_to_path, warnings)
    extract_markdown_links(root, nodes, graph)
    extract_workflow_relationships(root, nodes, graph)
    extract_config_usage(root, nodes, graph)
    graph.finalize()

    extract_test_coverage_mapping(root, nodes, graph, module_to_path, warnings)
    graph.finalize()

    if args.deep_infer:
        extract_deep_inference(root, nodes, graph)
        graph.finalize()

    extract_mirror_and_aliases(root, nodes, graph)
    graph.finalize()

    markdown = format_markdown(
        nodes=nodes,
        matrix=build_directory_matrix(nodes),
        exclude_dirs=exclude_dirs,
        deep_infer_enabled=args.deep_infer,
        warnings=warnings,
    )
    ensure_parent(output_path)
    output_path.write_text(markdown, encoding="utf-8", newline="\n")

    if json_path:
        ensure_parent(json_path)
        json_payload = to_json_payload(nodes, exclude_dirs, args.deep_infer)
        json_path.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print(f"Scanned files: {len(nodes)}")
    print(f"Markdown output: {output_path}")
    if json_path:
        print(f"JSON output: {json_path}")
    print(f"Warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
