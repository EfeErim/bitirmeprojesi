# Comprehensive Codebase Analysis Strategy

## Project Overview
**Project**: AADS-ULoRA v5.5 - Agricultural AI Development System
**Type**: Python backend with API, ML pipeline, mobile app
**Structure**: 
- `src/` - Core Python package with adapters, core, dataset, debugging, evaluation, middleware, monitoring, ood, pipeline, router, security, training, utils, visualization
- `api/` - FastAPI endpoints and middleware
- `tests/` - Unit and integration tests
- `config/` - JSON configuration files
- `mobile/` - Android application
- `docker/`, `monitoring/`, `docs/`, `scripts/`, `benchmarks/`, `colab_notebooks/`, `archive/`

---

## 1. File Organization Issues

### 1.1 Duplicate Files Across Directories
**Target Areas:**
- Compare `src/middleware/` vs `api/middleware/`
- Compare `src/monitoring/metrics.py` vs `api/metrics.py`
- Compare `src/utils/metrics.py` vs `src/evaluation/metrics.py`
- Check for duplicate `validation.py` files: `src/core/validation.py`, `api/validation.py`

**Key Patterns:**
- Files with identical or similar names in parallel directories
- Similar import patterns suggesting duplication
- Functions with identical signatures across files

**Tools/Commands:**
```bash
# Find files with same names in different directories
find src -type f -name "*.py" | xargs -I {} basename {} | sort | uniq -d
find api -type f -name "*.py" | xargs -I {} basename {} | sort | uniq -d

# Compare specific suspected duplicates
diff -u src/middleware/auth.py api/middleware/auth.py
diff -u src/middleware/caching.py api/middleware/caching.py
diff -u src/middleware/compression.py api/middleware/compression.py
diff -u src/middleware/rate_limit.py api/middleware/rate_limit.py
diff -u src/middleware/audit.py api/middleware/audit.py

# Find files with similar content (hash-based)
find src -type f -name "*.py" -exec md5sum {} \; | sort | uniq -w32 -d
find api -type f -name "*.py" -exec md5sum {} \; | sort | uniq -w32 -d
```

**Expected Output:**
```
DUPLICATE_FILES:
- src/middleware/auth.py vs api/middleware/auth.py: 95% similarity
- src/middleware/caching.py vs api/middleware/caching.py: 98% similarity
...
```

### 1.2 Misplaced Files
**Target Areas:**
- Test files outside `tests/` directory (e.g., `test_package_import.py`, `run_tests_directly.py`, `run_tests_without_pytest.py`)
- Configuration files in root vs `config/`
- Scripts that should be in `scripts/` but are in root
- Mobile app code that may have backend dependencies

**Key Patterns:**
- Files starting with `test_` or ending with `_test.py` outside `tests/`
- Configuration JSON/INI/YAML files in root directory
- Python scripts in root that are not entry points

**Tools/Commands:**
```bash
# Find test files outside tests directory
find . -type f -name "test_*.py" ! -path "./tests/*" ! -path "./.venv/*"
find . -type f -name "*_test.py" ! -path "./tests/*" ! -path "./.venv/*"

# Find configuration files in root
find . -maxdepth 1 -type f \( -name "*.json" -o -name "*.ini" -o -name "*.yaml" -o -name "*.yml" \) ! -name ".env.example"

# List all Python files in root (excluding standard project files)
find . -maxdepth 1 -type f -name "*.py" ! -name "setup.py" ! -name "sitecustomize.py"
```

**Expected Output:**
```
MISPLACED_FILES:
- test_package_import.py (should move to tests/unit/ or delete)
- run_tests_directly.py (should move to scripts/ or delete)
- run_tests_without_pytest.py (should move to scripts/ or delete)
```

### 1.3 Naming Inconsistencies
**Target Areas:**
- Module naming: snake_case vs camelCase
- Test file naming: `test_*.py` vs `*_test.py`
- Configuration key naming patterns
- Class naming conventions

**Key Patterns:**
- Mixed naming styles in same directory
- Inconsistent test naming (both `test_*.py` and `*_test.py` exist)
- Files that don't follow project conventions

**Tools/Commands:**
```bash
# Check for mixed test naming in tests/
ls tests/unit/ | grep -E "^test_" | wc -l
ls tests/unit/ | grep -E "_test\.py$" | wc -l

# Find files not following snake_case
find src -type f -name "*.py" | grep -v -E "^[a-z][a-z0-9_]*\.py$"
find api -type f -name "*.py" | grep -v -E "^[a-z][a-z0-9_]*\.py$"

# Check class naming (should be PascalCase)
grep -r "class [a-z]" src/ api/ --include="*.py" | head -20
```

**Expected Output:**
```
NAMING_ISSUES:
- tests/unit/test_adapter_comprehensive.py (comprehensive suffix inconsistent)
- tests/unit/verify_optimizations_simple.py (not prefixed with test_)
```

### 1.4 Orphaned Files
**Target Areas:**
- Files not imported anywhere
- Test files without corresponding implementation
- Configuration files not referenced
- Documentation that doesn't match current state

**Key Patterns:**
- Files with no imports/references in the codebase
- Old test files for deprecated features
- Backup files (`.bak`, `backup.log`)
- Archive directories that may be outdated

**Tools/Commands:**
```bash
# Find all Python files and check imports (requires Python analysis)
# Use a script to build import graph and find unreferenced files

# Find backup/log files
find . -type f \( -name "*.bak" -o -name "*.log" -o -name "*backup*" \) ! -path "./.venv/*"

# Check archive directory
ls -la archive/
```

**Expected Output:**
```
ORPHANED_FILES:
- backup.log (should be deleted or moved to logs/)
- archive/development-plans/dinov3-integration-plan.md (outdated?)
```

---

## 2. Code Redundancies

### 2.1 Duplicate Code Blocks or Functions
**Target Areas:**
- `src/middleware/` vs `api/middleware/` implementations
- `src/utils/metrics.py` vs `src/evaluation/metrics.py`
- `src/core/config_manager.py` vs `config/` usage patterns
- Validation logic across `src/core/validation.py`, `api/validation.py`, `src/adapter/`

**Key Patterns:**
- Functions with identical or near-identical bodies
- Repeated error handling patterns
- Duplicate logging configurations
- Similar data transformation logic

**Tools/Commands:**
```bash
# Use a code similarity detection tool (e.g., jscpd, simian)
# For Python, use pysonar2 or custom script with AST analysis

# Simple grep-based approach for common patterns
grep -r "def " src/middleware/ api/middleware/ | cut -d: -f1,2 | sort | uniq -d

# Find similar function signatures
grep -rh "def .*(" src/ api/ | sort | uniq -c | sort -nr | head -20

# Check for duplicate imports
grep -rh "^import \|^from " src/middleware/ api/middleware/ | sort | uniq -d
```

**Expected Output:**
```
DUPLICATE_CODE:
- Function authenticate() appears in src/middleware/auth.py and api/middleware/auth.py (98% match)
- Function calculate_metrics() appears in src/utils/metrics.py and src/evaluation/metrics.py (95% match)
```

### 2.2 Similar Implementations Across Modules
**Target Areas:**
- All `*_test.py` files for similar test patterns
- Configuration loading in different modules
- Database connection handling
- Error handling and logging setup

**Key Patterns:**
- Copy-pasted code with minor variations
- Repeated try-except blocks
- Similar class structures with different names
- Repeated docstrings

**Tools/Commands:**
```bash
# Find repeated exception handling patterns
grep -r "except Exception" src/ api/ --include="*.py" | wc -l

# Find similar logging setup
grep -r "logging\.getLogger" src/ api/ --include="*.py" | cut -d: -f2 | sort | uniq -c

# Use AST-based similarity detection (custom script needed)
python scripts/detect_similar_code.py --dirs src,api --threshold 0.8
```

**Expected Output:**
```
SIMILAR_IMPLEMENTATIONS:
- 15 files use identical try-except-log pattern
- 8 files have similar logging configuration blocks
```

### 2.3 Repeated Configuration Patterns
**Target Areas:**
- Hardcoded configuration values in code
- Repeated dictionary structures
- Duplicate environment variable names
- Similar FastAPI app initialization

**Key Patterns:**
- Hardcoded paths, URLs, or constants
- Repeated `Dict[str, Any]` patterns
- Multiple places defining the same config keys
- Duplicate pydantic model fields

**Tools/Commands:**
```bash
# Find hardcoded strings that look like configs
grep -r "http://\|https://\|localhost:\|0.0.0.0" src/ api/ --include="*.py" | grep -v "test"

# Find repeated dict patterns
grep -r "{" src/ api/ --include="*.py" -A 2 | grep -E "^\s+\"" | sort | uniq -d

# Check for duplicate pydantic models
grep -r "class.*BaseModel" src/ api/ --include="*.py" -A 5 | grep "Field(" | sort | uniq -d
```

**Expected Output:**
```
CONFIG_PATTERNS:
- Database URL hardcoded in 3 locations
- API response format repeated in 5 endpoint files
```

### 2.4 Unused Code
**Target Areas:**
- Imports that are never used
- Functions/classes that are never called
- Parameters that are always default
- Dead code paths

**Key Patterns:**
- `# TODO`, `# FIXME`, `# XXX` comments
- Import statements with `# noqa` or `# unused`
- Functions with only `pass` or `raise NotImplementedError`
- Variables assigned but never read

**Tools/Commands:**
```bash
# Use linters to find unused imports/code
flake8 --select=F401 --isolated src/ api/
mypy --warn-unused-ignores src/ api/

# Find TODO/FIXME comments
grep -r "TODO\|FIXME\|XXX\|HACK" src/ api/ --include="*.py"

# Find NotImplementedError
grep -r "NotImplementedError" src/ api/ --include="*.py"

# Use vulture for dead code detection
pip install vulture
vulture src/ api/ --min-confidence 80
```

**Expected Output:**
```
UNUSED_CODE:
- Unused imports: 47 instances
- Dead functions: 12
- TODO comments: 23
```

---

## 3. Import and Dependency Issues

### 3.1 Circular Dependencies
**Target Areas:**
- `src/core/` modules importing from each other
- `src/pipeline/` importing from `src/router/` and vice versa
- `api/endpoints/` importing from `api/middleware/` and middleware importing endpoints
- Cross-dependencies between `src/adapter/`, `src/utils/`, `src/evaluation/`

**Key Patterns:**
- Import chains that form cycles
- Late imports inside functions (workaround for circular deps)
- Import errors at runtime

**Tools/Commands:**
```bash
# Use pydeps or pydependency to visualize dependencies
pip install pydeps
pydeps src --max-bacon=2 --output=deps.svg

# Use snakefood to detect circular dependencies
pip install snakefood
sfood src/ | sfood-checker

# Manual check with Python script
python scripts/check_circular_imports.py --paths src,api
```

**Expected Output:**
```
CIRCULAR_DEPENDENCIES:
- src/core/config_manager.py -> src/core/validation.py -> src/core/config_manager.py
- api/endpoints/crops.py -> api/middleware/auth.py -> api/endpoints/crops.py
```

### 3.2 Unused Imports
**Target Areas:**
- All Python files in `src/` and `api/`
- Test files that import unused fixtures
- Configuration modules

**Key Patterns:**
- Standard library imports not used
- Third-party imports with `# noqa` comment
- Local imports that are never referenced

**Tools/Commands:**
```bash
# Use flake8 with pyflakes
flake8 --select=F401 src/ api/

# Use autoflake to identify and remove
autoflake --check-only --remove-all-unused-imports src/ api/

# Use pylint
pylint --disable=all --enable=unused-import src/ api/
```

**Expected Output:**
```
UNUSED_IMPORTS:
- src/middleware/auth.py: import json (unused)
- api/endpoints/crops.py: from datetime import datetime (unused)
...
```

### 3.3 Inconsistent Import Patterns
**Target Areas:**
- Mix of absolute and relative imports
- Different ordering (stdlib, third-party, local)
- Some files using `from x import *`
- Inconsistent use of `__init__.py` for namespace packages

**Key Patterns:**
- Both `from src.utils import` and `from ..utils import` in same package
- Import statements not grouped by type
- Missing `__init__.py` files in subdirectories

**Tools/Commands:**
```bash
# Check for relative vs absolute imports
grep -r "from \.\." src/ api/ --include="*.py" | wc -l
grep -r "from src\." src/ --include="*.py" | wc -l

# Check isort compliance
isort --check-only src/ api/

# Find missing __init__.py
find src -type d -exec test -e "{}/__init__.py" \; -print
find api -type d -exec test -e "{}/__init__.py" \; -print
```

**Expected Output:**
```
IMPORT_INCONSISTENCIES:
- src/adapter/ uses relative imports, src/core/ uses absolute imports
- 12 directories missing __init__.py
```

### 3.4 Missing `__init__.py` Files
**Target Areas:**
- All subdirectories in `src/`, `api/`, `tests/`
- Check if project uses namespace packages (PEP 420) or regular packages

**Key Patterns:**
- Directories with Python files but no `__init__.py`
- Import errors when running tests

**Tools/Commands:**
```bash
# Find directories with Python files but no __init__.py
find src -name "*.py" -type f | xargs -I {} dirname {} | sort | uniq | while read d; do [ -f "$d/__init__.py" ] || echo "$d"; done
find api -name "*.py" -type f | xargs -I {} dirname {} | sort | uniq | while read d; do [ -f "$d/__init__.py" ] || echo "$d"; done
```

**Expected Output:**
```
MISSING_INIT_FILES:
- src/debugging/
- src/visualization/
```

---

## 4. Configuration Inconsistencies

### 4.1 Duplicate Configuration Values
**Target Areas:**
- `config/*.json` files for overlapping keys
- Hardcoded values in code that match config values
- Environment variable names repeated across files

**Key Patterns:**
- Same key with different values in different config files
- Config keys that appear in multiple files
- Values that should be centralized (e.g., API version, model names)

**Tools/Commands:**
```bash
# Extract all keys from JSON configs
for f in config/*.json; do echo "=== $f ==="; jq -r 'keys[]' "$f" 2>/dev/null; done | sort | uniq -c

# Compare config files pairwise
for f1 in config/*.json; do
  for f2 in config/*.json; do
    [ "$f1" \< "$f2" ] && echo "Comparing $f1 vs $f2" && jq --argfile f1 "$f1" --argfile f2 "$f2" -n '$f1 | keys_unsorted as $k1 | $f2 | keys_unsorted as $k2 | ($k1 - $k2) as $diff | "Keys only in \($f1): \($diff)"'
  done
done

# Find hardcoded config-like values in code
grep -r "DEBUG\|LOG_LEVEL\|DATABASE_URL\|REDIS_URL" src/ api/ --include="*.py" | grep -v "os.environ\|config\."
```

**Expected Output:**
```
DUPLICATE_CONFIG_KEYS:
- database.host appears in base.json, development.json, production.json
- api.version appears in base.json, staging.json, production.json
```

### 4.2 Conflicting Settings Between Environments
**Target Areas:**
- `config/development.json` vs `config/production.json`
- `config/staging.json` vs `config/test.json`
- Security settings across environments

**Key Patterns:**
- Different values for same key in different environments
- Debug/verbose settings enabled in production
- Inconsistent security configurations

**Tools/Commands:**
```bash
# Compare specific keys across configs
jq -r '.debug' config/*.json
jq -r '.log_level' config/*.json
jq -r '.database' config/*.json

# Generate diff report
python scripts/compare_configs.py --dir config/ --output config_diff_report.md
```

**Expected Output:**
```
CONFIG_CONFLICTS:
- debug: true in development.json, false in production.json (expected)
- log_level: DEBUG in development.json, INFO in production.json (expected)
- database.port: 5432 in development.json, 5433 in production.json (unexpected mismatch)
```

### 4.3 Unused Configuration Options
**Target Areas:**
- Keys in config files that are never accessed in code
- Environment variables defined but not used
- Deprecated config options

**Key Patterns:**
- Config keys with no corresponding `config.get()` or attribute access
- Comments like "TODO: use this config"
- Old configuration formats

**Tools/Commands:**
```bash
# Extract all config key accesses in code
grep -rh "config\[" src/ api/ --include="*.py" | grep -o "'.*'" | sort | uniq
grep -rh "config\." src/ api/ --include="*.py" | grep -o "\..*" | cut -d. -f2 | sort | uniq

# Compare with available keys
python scripts/find_unused_configs.py --config-dir config/ --code-dirs src,api
```

**Expected Output:**
```
UNUSED_CONFIG_OPTIONS:
- config/base.json: "experimental_features" (never accessed)
- config/production.json: "legacy_mode" (deprecated)
```

---

## 5. Test Issues

### 5.1 Redundant or Overlapping Tests
**Target Areas:**
- `tests/unit/test_*.py` files with similar coverage
- `test_adapter.py` vs `test_adapter_comprehensive.py`
- `test_router.py` vs `test_router_comprehensive.py` vs `test_router_minimal.py`
- `test_ood.py` vs `test_ood_comprehensive.py`
- `test_pipeline_comprehensive.py` vs integration tests

**Key Patterns:**
- Multiple test files testing same functions
- Similar test names across files
- Overlapping fixtures and mocks

**Tools/Commands:**
```bash
# Find test files with similar names
ls tests/unit/ | sort

# Extract test function names and check duplicates
grep -rh "^def test_" tests/unit/ | sed 's/:def /:/' | cut -d: -f2 | sort | uniq -c | sort -nr | head -20

# Compare test coverage
pytest --cov=src --cov-report=html tests/unit/
# Check html report for overlapping coverage
```

**Expected Output:**
```
REDUNDANT_TESTS:
- test_adapter.py and test_adapter_comprehensive.py: 80% overlapping test cases
- test_router.py, test_router_comprehensive.py, test_router_minimal.py: duplicate coverage
```

### 5.2 Duplicate Test Fixtures
**Target Areas:**
- `tests/fixtures/sample_data.py` vs `tests/fixtures/test_fixtures.py`
- Fixtures defined in multiple test files
- Conftest fixtures that duplicate module-level fixtures

**Key Patterns:**
- Same fixture name with different implementations
- Similar sample data structures repeated
- Fixtures that could be parameterized

**Tools/Commands:**
```bash
# Find all fixture definitions
grep -rh "^@pytest.fixture" tests/ --include="*.py" | cut -d: -f1 | sort | uniq -c

# Check for duplicate fixture names
grep -rh "^def .*(" tests/conftest.py tests/fixtures/*.py | sed 's/^def //' | cut -d'(' -f1 | sort | uniq -d

# Find similar data structures
diff tests/fixtures/sample_data.py tests/fixtures/test_fixtures.py
```

**Expected Output:**
```
DUPLICATE_FIXTURES:
- fixture "sample_crop_data" in sample_data.py and test_fixtures.py (identical)
- 5 fixtures could be consolidated into parameterized fixtures
```

### 5.3 Missing Test Coverage
**Target Areas:**
- Critical modules: `src/core/pipeline_manager.py`, `src/router/vlm_pipeline.py`
- Error handling paths
- Edge cases and validation
- Integration scenarios

**Key Patterns:**
- Modules with no corresponding test file
- Functions with no tests
- Missing tests for exception handling

**Tools/Commands:**
```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing tests/

# Find modules with < 50% coverage
pytest --cov=src --cov-report=json tests/ | python -c "import json, sys; data=json.load(open('coverage.json')); print([k for k,v in data['totals']['covered'].items() if v/data['totals']['num_statements'][k] < 0.5])"

# List files with no tests
find src -name "*.py" -type f | while read f; do bn=$(basename "$f" .py); if ! find tests -name "test_${bn}.py" -o -name "*_${bn}.py" | grep -q .; then echo "$f"; fi; done
```

**Expected Output:**
```
MISSING_COVERAGE:
- src/visualization/visualization.py: 0% coverage
- src/debugging/monitoring.py: 12% coverage
- src/ood/dynamic_thresholds.py: 35% coverage (needs more edge case tests)
```

### 5.4 Inefficient Test Patterns
**Target Areas:**
- Tests with excessive mocking
- Slow integration tests in unit suite
- Tests that could be parameterized but aren't
- Repeated setup/teardown code

**Key Patterns:**
- `time.sleep()` in tests
- Large fixture scopes where smaller would suffice
- Tests that hit real APIs/DBs instead of mocking
- Duplicate assertion patterns

**Tools/Commands:**
```bash
# Find tests with sleep
grep -r "time\.sleep" tests/ --include="*.py"

# Find tests using real connections
grep -r "requests\.\|urllib\|httpx" tests/ --include="*.py" | grep -v "mock"

# Check for parameterization opportunities
grep -rh "def test_" tests/unit/ | sed 's/def //' | cut -d'(' -f1 | sort | uniq -c | awk '$1 > 1 {print $2}'

# Count assertions per test
grep -rh "assert " tests/unit/test_*.py | wc -l
```

**Expected Output:**
```
INEFFICIENT_PATTERNS:
- 8 tests use time.sleep() (should use async/waiting)
- 12 tests make real HTTP requests (should mock)
- test_router.py has 45 similar assertions (could parameterize)
```

---

## 6. Architectural Violations

### 6.1 Mixed Responsibilities in Modules
**Target Areas:**
- `src/middleware/` modules that also contain business logic
- `api/endpoints/` that directly access database instead of using service layer
- `src/utils/` modules that do more than utilities
- `src/core/` modules that mix configuration, validation, and execution

**Key Patterns:**
- Files > 500 lines with multiple concerns
- Direct database access in API endpoints
- Business logic in middleware
- Mix of training, inference, and evaluation code

**Tools/Commands:**
```bash
# Find large files
find src api -name "*.py" -type f -exec wc -l {} + | sort -rn | head -20

# Check for database imports in endpoints
grep -l "sqlalchemy\|psycopg2\|redis" api/endpoints/*.py

# Check for business logic in middleware
grep -l "model\|train\|predict" src/middleware/*.py api/middleware/*.py

# Analyze imports to detect layering violations
python scripts/analyze_responsibilities.py --paths src,api
```

**Expected Output:**
```
RESPONSIBILITY_ISSUES:
- api/endpoints/crops.py: 450 lines, mixes validation, DB access, and business logic
- src/middleware/auth.py: contains user management logic (should be in separate service)
- src/utils/metrics.py: 600 lines, contains training metrics, evaluation metrics, and visualization
```

### 6.2 Violations of Separation of Concerns
**Target Areas:**
- API layer directly calling training code
- Configuration mixed with business logic
- Monitoring code scattered across modules
- Security checks in multiple places

**Key Patterns:**
- Endpoints importing from `src.training`
- Direct file system access in core modules
- Multiple places handling the same error
- Inconsistent use of dependency injection

**Tools/Commands:**
```bash
# Check for direct training imports in API
grep -r "from src.training" api/ --include="*.py"

# Check for file I/O in core
grep -r "open(" src/core/ --include="*.py" | grep -v "with open" | head

# Find security checks scattered
grep -r "security\|auth\|permission" src/ api/ --include="*.py" | grep -v "middleware" | grep -v "security.py" | head

# Check for dependency injection usage
grep -r "Depends(" api/endpoints/ --include="*.py" | wc -l
grep -r "class.*:" src/ api/ --include="*.py" | grep -v "pydantic" | wc -l
```

**Expected Output:**
```
SEPARATION_ISSUES:
- api/endpoints/diagnose.py imports from src.training.phase1_training (violation)
- src/core/config_manager.py reads files directly (should abstract)
- Security validation in 5 different locations (should centralize)
```

### 6.3 Inconsistent Patterns Across Similar Components
**Target Areas:**
- All endpoint files in `api/endpoints/`
- All middleware in `src/middleware/` and `api/middleware/`
- Test files for similar features
- Configuration loading patterns

**Key Patterns:**
- Different response formats across endpoints
- Inconsistent error handling
- Different validation approaches
- Mixed use of sync/async

**Tools/Commands:**
```bash
# Check endpoint patterns
grep -l "async def" api/endpoints/*.py | wc -l
grep -l "def " api/endpoints/*.py | wc -l

# Check response models
grep -h "response_model" api/endpoints/*.py | sort | uniq -c

# Check error handling
grep -h "HTTPException\|raise" api/endpoints/*.py | head -20

# Compare middleware patterns
diff -u src/middleware/auth.py api/middleware/auth.py | head -50
```

**Expected Output:**
```
INCONSISTENT_PATTERNS:
- 3 endpoints use async, 2 use sync (inconsistent)
- Response formats: JSONResponse, dict, pydantic models mixed
- Error handling: HTTPException, JSONResponse, plain dict all used
```

---

## Analysis Execution Plan

### Phase 1: Static Analysis (Day 1)
1. Run all grep/find commands to collect baseline data
2. Generate import dependency graphs
3. Run linters (flake8, mypy, pylint) and collect warnings
4. Run vulture for dead code detection
5. Check for circular dependencies with sfood or custom script

### Phase 2: Dynamic Analysis (Day 2)
1. Run test suite with coverage
2. Profile test execution times
3. Check for import time issues
4. Validate configuration loading

### Phase 3: Deep Dive (Day 3)
1. Manual review of high-priority files (large files, duplicates)
2. AST-based code similarity analysis
3. Architecture review with dependency graphs
4. Interview-style code walkthrough (if team available)

### Phase 4: Reporting (Day 4)
1. Compile all findings into structured report
2. Prioritize issues by impact and effort
3. Create actionable recommendations
4. Generate visualizations (dependency graphs, heat maps)

---

## Deliverables

### Primary Report: `CODEBASE_ANALYSIS_REPORT.md`
```markdown
# Codebase Analysis Report

## Executive Summary
- Total files analyzed: X
- Lines of code: Y
- Major issues found: Z
- Priority breakdown: High (N), Medium (M), Low (L)

## Detailed Findings by Category

### 1. File Organization
#### 1.1 Duplicate Files
- [High] src/middleware/auth.py duplicates api/middleware/auth.py (95% similarity)
  - Impact: Maintenance burden, bug fixes need to be applied twice
  - Recommendation: Consolidate into single module, refactor imports

#### 1.2 Misplaced Files
- [Medium] test_package_import.py in root should be in tests/unit/
...

### 2. Code Redundancies
...

### 3. Import Issues
...

### 4. Configuration Issues
...

### 5. Test Issues
...

### 6. Architectural Violations
...

## Appendices
- A: Full File Inventory
- B: Import Dependency Graph
- C: Test Coverage Report
- D: Configuration Matrix
- E: Raw Tool Outputs
```

### Secondary Reports
- `import_dependency_graph.png` - Visual dependency map
- `test_coverage_heatmap.html` - Interactive coverage view
- `config_comparison_matrix.csv` - All config files comparison
- `duplicate_code_report.json` - Machine-readable duplicate findings
- `circular_dependencies.txt` - List of import cycles

---

## Tools Summary

### Required Tools (Install)
```bash
pip install flake8 mypy pylint isort autoflake vulture pydeps snakefood pytest pytest-cov
```

### Custom Scripts (Create in scripts/)
- `check_circular_imports.py` - Detect import cycles
- `find_unused_configs.py` - Match config keys with code usage
- `compare_configs.py` - Generate config diff matrix
- `analyze_responsibilities.py` - Detect mixed responsibilities
- `detect_similar_code.py` - AST-based similarity detection

### One-Line Commands
See individual sections above for specific commands.

---

## Success Criteria

✅ All duplicate files identified and quantified  
✅ All circular dependencies mapped  
✅ Test coverage report generated with gaps highlighted  
✅ Configuration inconsistencies documented  
✅ Architectural violations cataloged with examples  
✅ Actionable recommendations provided for each issue  
✅ Complete report with prioritization delivered

---

## Notes

- This analysis should be run from the project root: `d:/bitirme projesi`
- Some tools may need Python 3.9+ (check `python --version`)
- The `src/` directory is the main package; `api/` is the FastAPI layer
- Configuration is JSON-based in `config/`
- Tests use pytest with fixtures in `tests/conftest.py`
- Mobile app (Android/Kotlin) is out of scope for Python analysis but file organization should be checked

---

**Next Steps After Analysis:**
1. Review findings with team
2. Prioritize fixes based on impact
3. Create refactoring tickets
4. Establish coding standards to prevent recurrence
5. Set up automated checks in CI/CD
