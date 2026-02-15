# AADS-ULoRA v5.5

**Agricultural AI Development System - ULoRA v5.5**

A production-ready, multi-crop disease detection system using independent crop adapters with dynamic OOD (Out-of-Distribution) detection and **crop+part routing** for precise disease identification.

## Project Restructuring

### Overview
The project has undergone significant reorganization to improve maintainability, reduce complexity, and establish a cleaner foundation for future development. This restructuring consolidates multiple version directories and eliminates redundant files.

### Key Changes

#### 1. Desktop.ini Removal
- Eliminated all system-generated desktop.ini placeholder files across project directories
- Removed from: `api/`, `config/`, `mobile/`, `src/`, `tests/`, `docs/`, `documents/`, `lit_review/`, and subdirectories
- This reduces clutter and prevents confusion with actual project files

#### 2. Configuration Consolidation
- Merged multiple `.gitattributes` and `.gitignore` files into unified `config/` directory
- Consolidated environment configuration files:
  - `config/base.json` - Base configuration
  - `config/development.json` - Development overrides
  - `config/production.json` - Production settings
- Created `config/adapter_spec_v55.json` for adapter specifications
- Standardized `config/pytest.ini` for test configuration

#### 3. Code Refactoring
- **API Structure**: Updated endpoints to match unified codebase
  - `api/endpoints/crops.py` - Crop management
  - `api/endpoints/diagnose.py` - Disease diagnosis
  - `api/endpoints/feedback.py` - User feedback collection
  - `api/endpoints/monitoring.py` - System monitoring
- **Middleware**: Consolidated security, caching, rate limiting, and compression middleware
- **Core Components**: Refactored adapter, pipeline, router, and OOD detection modules
- **Utility Extraction**: Moved common functions to `src/utils/` for reusability

#### 4. Test Suite Updates
- Updated comprehensive test files for unified structure
- Added new test cases for consolidated configuration
- Maintained coverage for: adapters, OOD detection, pipelines, routers, and validation

### File Deletions (Consolidation)

#### Version Directories Removed
- `versions/v5.5.0-baseline/` - Baseline version
- `versions/v5.5.1-ood/` - OOD detection version
- `versions/v5.5.4-dinov3/` - Dinov3 integration version

#### Configuration Files Removed
- Multiple `.gitattributes` files (now unified)
- Multiple `.gitignore` files (now unified)
- `README_STAGE3.md` files (duplicate documentation)

#### Documentation Files Removed
- `colab_notebooks/README.md` (duplicate)
- `current/README.md` (duplicate)
- Multiple `.tex` files (consolidated into compiled PDFs)

#### Implementation Files Removed
- `requirements_optimized.txt` (multiple locations)
- `setup_optimized.py` (multiple locations)

#### Mobile Application Files Removed
- `versions/v5.5.0-baseline/mobile/android/` (complete Android project)
- Consolidated mobile code references

#### Literature Review Files Removed
- `versions/v5.5.0-baseline/lit_review/` (complete directory)
- Centralized literature review in root `lit_review/` directory

### Directory Structure

```
d:/bitirme projesi/
├── api/                    # FastAPI backend
│   ├── endpoints/         # REST API endpoints
│   ├── middleware/        # Request middleware
│   ├── database.py
│   ├── graceful_shutdown.py
│   ├── main.py
│   ├── metrics.py
│   └── validation.py
├── config/                # Configuration files
│   ├── base.json
│   ├── development.json
│   ├── production.json
│   ├── adapter_spec_v55.json
│   └── pytest.ini
├── src/                   # Core source code
│   ├── adapter/          # Crop adapters
│   ├── dataset/          # Data preparation
│   ├── debugging/        # Debug utilities
│   ├── evaluation/       # Metrics and evaluation
│   ├── ood/              # OOD detection
│   ├── pipeline/         # Processing pipelines
│   ├── router/           # Crop routing logic
│   ├── training/         # Training scripts
│   ├── utils/            # Shared utilities
│   └── visualization/    # Visualization tools
├── tests/                 # Test suites
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── conftest.py
├── docs/                  # Documentation
├── docker/                # Containerization
├── monitoring/            # Prometheus/Grafana
├── colab_notebooks/       # Jupyter notebooks
├── lit_review/            # Literature review
├── plans/                 # Implementation plans
├── benchmarks/            # Performance benchmarks
├── demo/                  # Demo application
└── [configuration files] # Root-level configs
```

## Benefits

- **Simplified Maintenance**: Single codebase eliminates version confusion
- **Clean Structure**: Removed system files and duplicates
- **Standardized Configuration**: Centralized config management
- **Improved Testing**: Unified test structure with comprehensive coverage
- **Better Documentation**: Consolidated and up-to-date docs

## Technical Specifications

- **Framework**: FastAPI with async support
- **ML Backend**: PyTorch with Vision Transformers
- **Adaptation**: ULoRA (Unified Low-Rank Adaptation)
- **OOD Detection**: Mahalanobis distance + prototype-based methods
- **Routing**: Dynamic crop+part routing for specialized diagnosis
- **Configuration**: JSON-based with environment overrides

## Status

Production-ready for Uyumsoft ZiraiTakip integration. Version 5.5.0. Last updated: February 2026.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment: Copy `config/development.json` to `config/local.json`
3. Run API: `python -m api.main`
4. Access docs: `http://localhost:8000/docs`

## Documentation

- [API Reference](docs/api_reference.md)
- [Architecture Overview](docs/architecture.md)
- [Crop Router Technical Guide](CROP_ROUTER_TECHNICAL_GUIDE.md)
- [Rollback Guide](ROLLBACK_GUIDE.md)
- [Comprehensive Codebase Evaluation](COMPREHENSIVE_CODEBASE_EVALUATION.md)