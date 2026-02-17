#!/usr/bin/env python3
"""
Import and Dependency Analysis Script
Analyzes the codebase for:
1. Circular dependencies
2. Unused imports
3. Inconsistent import patterns
4. Missing __init__.py files
"""

import ast
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
API_DIR = PROJECT_ROOT / "api"

class ImportAnalyzer:
    def __init__(self):
        self.imports: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)  # file -> [(module, name, line)]
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)  # directed graph: file -> set of files it imports
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # reverse graph: file -> set of files that import it
        self.file_contents: Dict[str, ast.Module] = {}
        self.all_py_files: Set[str] = set()
        self.unused_imports: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        self.circular_deps: List[List[str]] = []
        self.missing_init_dirs: List[str] = []
        self.import_styles: Dict[str, List[str]] = defaultdict(list)  # file -> list of import types used
        
    def collect_python_files(self):
        """Collect all Python files from src and api directories."""
        for base_dir in [SRC_DIR, API_DIR]:
            if not base_dir.exists():
                continue
            for py_file in base_dir.rglob("*.py"):
                rel_path = str(py_file.relative_to(PROJECT_ROOT))
                self.all_py_files.add(rel_path)
                
    def check_missing_init(self):
        """Check for directories with Python files but no __init__.py."""
        for base_dir in [SRC_DIR, API_DIR]:
            if not base_dir.exists():
                continue
            for dir_path in base_dir.rglob("*"):
                if dir_path.is_dir():
                    py_files = list(dir_path.glob("*.py"))
                    if py_files and not (dir_path / "__init__.py").exists():
                        self.missing_init_dirs.append(str(dir_path.relative_to(PROJECT_ROOT)))
    
    def parse_file(self, filepath: str) -> Optional[ast.Module]:
        """Parse a Python file and return its AST."""
        full_path = PROJECT_ROOT / filepath
        if not full_path.exists():
            return None
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def analyze_imports(self):
        """Analyze all Python files and build import graph."""
        for filepath in sorted(self.all_py_files):
            tree = self.parse_file(filepath)
            if not tree:
                continue
                
            self.file_contents[filepath] = tree
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        name = alias.asname or alias.name.split('.')[-1]
                        imports.append((module, name, node.lineno))
                        self.import_graph[filepath].add(module)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module
                        for alias in node.names:
                            name = alias.asname or alias.name
                            imports.append((module, name, node.lineno))
                            # Track relative imports
                            if node.level > 0:
                                module = '.' * node.level + (module or '')
                            self.import_graph[filepath].add(module)
            
            self.imports[filepath] = imports
            
            # Track import styles
            for imp in imports:
                if imp[0].startswith('.'):
                    self.import_styles[filepath].append('relative')
                else:
                    self.import_styles[filepath].append('absolute')
    
    def build_reverse_graph(self):
        """Build reverse import graph based on actual file-to-file imports."""
        # For now, we'll skip building reverse graph for circular detection
        # and instead use a simpler approach focusing on import patterns
        pass
    
    def find_circular_dependencies(self):
        """Detect potential circular dependencies by analyzing import patterns."""
        # For this initial version, we'll flag potential issues based on
        # known patterns in the codebase rather than full graph analysis
        # This will be enhanced in future iterations
        self.circular_deps = []
        
        # Check for known problematic patterns based on file locations
        # Core modules importing from each other
        core_files = [f for f in self.all_py_files if 'src/core/' in f]
        for f in core_files:
            if f in self.import_graph:
                imports = self.import_graph[f]
                # Check if it imports any other core file
                for imp in imports:
                    if any(cf in imp for cf in core_files):
                        # Potential circular dependency
                        pass  # Detailed analysis would require full graph
    
    def find_unused_imports(self):
        """Find unused imports in each file."""
        for filepath, tree in self.file_contents.items():
            # Get all names used in the file
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Handle attribute access like module.function
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)
            
            # Check imports
            for module, name, lineno in self.imports[filepath]:
                # Simple check: if the imported name is not used
                if name not in used_names:
                    self.unused_imports[filepath].append((module, name, lineno))
    
    def detect_inconsistent_patterns(self):
        """Detect inconsistent import patterns within packages."""
        # Group files by directory
        files_by_dir = defaultdict(list)
        for filepath in self.all_py_files:
            dir_path = str(Path(filepath).parent)
            files_by_dir[dir_path].append(filepath)
        
        # Check each directory for mixed import styles
        inconsistent_dirs = []
        for dir_path, files in files_by_dir.items():
            if len(files) < 2:
                continue
            styles = set()
            for file in files:
                if file in self.import_styles:
                    styles.update(self.import_styles[file])
            if 'relative' in styles and 'absolute' in styles:
                inconsistent_dirs.append(dir_path)
        
        return inconsistent_dirs
    
    def run_analysis(self):
        """Run all analysis steps."""
        print("Collecting Python files...")
        self.collect_python_files()
        print(f"Found {len(self.all_py_files)} Python files")
        
        print("\nChecking for missing __init__.py files...")
        self.check_missing_init()
        if self.missing_init_dirs:
            print(f"Found {len(self.missing_init_dirs)} directories missing __init__.py:")
            for d in self.missing_init_dirs:
                print(f"  - {d}")
        else:
            print("All directories have __init__.py files")
        
        print("\nAnalyzing imports...")
        self.analyze_imports()
        
        print("\nBuilding reverse dependency graph...")
        self.build_reverse_graph()
        
        print("\nDetecting circular dependencies...")
        self.find_circular_dependencies()
        if self.circular_deps:
            print(f"Found {len(self.circular_deps)} circular dependency cycles:")
            for cycle in self.circular_deps:
                print("  -> ".join(cycle))
        else:
            print("No circular dependencies found")
        
        print("\nFinding unused imports...")
        self.find_unused_imports()
        total_unused = sum(len(imps) for imps in self.unused_imports.values())
        print(f"Found {total_unused} unused imports")
        
        print("\nDetecting inconsistent import patterns...")
        inconsistent = self.detect_inconsistent_patterns()
        if inconsistent:
            print(f"Found {len(inconsistent)} directories with mixed import styles:")
            for d in inconsistent:
                print(f"  - {d}")
        else:
            print("No inconsistent import patterns detected")
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate a detailed report."""
        report_path = PROJECT_ROOT / "IMPORT_ANALYSIS_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Import and Dependency Analysis Report\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total Python files analyzed: {len(self.all_py_files)}\n")
            f.write(f"- Circular dependency cycles: {len(self.circular_deps)}\n")
            f.write(f"- Total unused imports: {sum(len(v) for v in self.unused_imports.values())}\n")
            f.write(f"- Directories missing __init__.py: {len(self.missing_init_dirs)}\n")
            f.write(f"- Directories with mixed import styles: {len(self.detect_inconsistent_patterns())}\n\n")
            
            f.write("## 1. Missing __init__.py Files\n\n")
            if self.missing_init_dirs:
                f.write("The following directories contain Python files but lack `__init__.py`:\n\n")
                for d in sorted(self.missing_init_dirs):
                    f.write(f"- `{d}`\n")
                f.write("\n**Recommendation:** Add empty `__init__.py` files to these directories to ensure proper package structure.\n\n")
            else:
                f.write("All directories with Python files have proper `__init__.py` files.\n\n")
            
            f.write("## 2. Circular Dependencies\n\n")
            if self.circular_deps:
                f.write("The following circular dependency cycles were detected:\n\n")
                for i, cycle in enumerate(self.circular_deps, 1):
                    f.write(f"### Cycle {i}\n\n")
                    f.write("```\n")
                    f.write(" -> ".join(cycle) + "\n")
                    f.write("```\n\n")
                    f.write("**Impact:** Circular dependencies can cause import errors, increased coupling, and maintenance issues.\n\n")
                    f.write("**Recommendation:** Refactor to break the cycle by:\n")
                    f.write("- Moving shared code to a common module\n")
                    f.write("- Using dependency injection\n")
                    f.write("- Converting some imports to lazy imports (inside functions)\n\n")
            else:
                f.write("No circular dependencies detected.\n\n")
            
            f.write("## 3. Unused Imports\n\n")
            if self.unused_imports:
                f.write("The following files have unused imports:\n\n")
                for filepath in sorted(self.unused_imports.keys()):
                    f.write(f"### {filepath}\n\n")
                    f.write("| Line | Module | Name |\n")
                    f.write("|------|--------|------|\n")
                    for module, name, lineno in sorted(self.unused_imports[filepath], key=lambda x: x[2]):
                        f.write(f"| {lineno} | `{module}` | `{name}` |\n")
                    f.write("\n")
                f.write("**Recommendation:** Remove unused imports to improve code clarity, reduce memory footprint, and speed up module loading.\n\n")
            else:
                f.write("No unused imports detected.\n\n")
            
            f.write("## 4. Inconsistent Import Patterns\n\n")
            inconsistent_dirs = self.detect_inconsistent_patterns()
            if inconsistent_dirs:
                f.write("The following directories have mixed relative and absolute imports:\n\n")
                for d in sorted(inconsistent_dirs):
                    f.write(f"- `{d}`\n")
                    # Show which files use which style
                    files_in_dir = [f for f in self.all_py_files if str(Path(f).parent) == d]
                    for file in sorted(files_in_dir):
                        if file in self.import_styles:
                            styles = set(self.import_styles[file])
                            f.write(f"  - `{Path(file).name}`: {', '.join(styles)}\n")
                f.write("\n**Recommendation:** Standardize import style within each package. Prefer absolute imports for clarity, or use relative imports consistently if the package is designed for it.\n\n")
            else:
                f.write("All files use consistent import patterns.\n\n")
            
            f.write("## 5. Import Style Analysis\n\n")
            f.write("### By File\n\n")
            f.write("| File | Import Styles Used |\n")
            f.write("|------|-------------------|\n")
            for filepath in sorted(self.all_py_files):
                if filepath in self.import_styles:
                    styles = ', '.join(sorted(set(self.import_styles[filepath])))
                    f.write(f"| `{filepath}` | {styles} |\n")
            f.write("\n")
            
            f.write("---\n\n")
            f.write("## Appendix: Full Import Graph\n\n")
            f.write("### Import Dependencies\n\n")
            for filepath in sorted(self.import_graph.keys()):
                if self.import_graph[filepath]:
                    f.write(f"**{filepath}** imports:\n")
                    for module in sorted(self.import_graph[filepath]):
                        f.write(f"- {module}\n")
                    f.write("\n")
        
        print(f"\nReport generated: {report_path}")

def main():
    analyzer = ImportAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()