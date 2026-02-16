# Migration Guide: Consolidated AADS-ULoRA v5.5 Structure

## Overview

This guide helps users transition from the previous multi-version structure to the new consolidated AADS-ULoRA v5.5 architecture. The consolidation simplifies deployment, reduces complexity, and improves maintainability.

## What Changed

### Before Consolidation
- Multiple version directories: `versions/v5.5.0-baseline/`, `versions/v5.5.1-ood/`, etc.
- Complex version switching with `current/` symlink
- Version management scripts and backup systems
- Duplicate documentation and configuration files

### After Consolidation
- **Single unified codebase** - all active code in root directories
- **Simplified configuration** - centralized in `config/` directory
- **No version switching** - one definitive version
- **Cleaner structure** - removed redundant directories
- **Git-based versioning** - use Git history for version tracking

## Key Directory Changes

### Removed Directories
- `versions/` - All version-specific directories
- `backups/` - Backup system (replaced by Git branches)
- `current/` - Symlink to active version
- `version_management/` - Version switching scripts
- `visualization/` - Moved to `src/visualization/`

### Current Structure
```
d:/bitirme projesi/
├── api/                    # FastAPI backend
├── config/                 # All configuration files
├── src/                    # Core source code
├── tests/                  # Test suite
├── docs/                   # Documentation
├── docker/                 # Docker configurations
├── scripts/                # Utility scripts
├── benchmarks/             # Performance benchmarks
├── demo/                   # Demo applications
└── [root-level files]      # README.md, requirements.txt, etc.
```

## Migration Steps

### For Developers

#### 1. Update Your Local Repository

If you have an old version cloned:

```bash
# Backup your current work
git branch my-backup

# Fetch latest changes
git fetch origin

# Reset to new consolidated structure
git reset --hard origin/main

# Clean up untracked files
git clean -fd
```

#### 2. Update Environment Configuration

Old configuration files in version directories are no longer used. Use:

```bash
# Copy example environment
cp .env.example .env

# Edit .env with your settings
# - API_HOST and API_PORT
# - Model paths (if using custom models)
# - Logging configuration
```

#### 3. Update Import Paths

If you had code importing from version-specific paths:

**OLD:**
```python
from versions.v5.5.3-performance.src.adapter import CropAdapter
```

**NEW:**
```python
from src.adapter import CropAdapter
```

#### 4. Configuration Files

All configuration is now in the `config/` directory:

- `config/base.json` - Base configuration
- `config/development.json` - Development overrides
- `config/production.json` - Production settings
- `config/adapter-spec.json` - Model adapter specifications
- `config/security-config.json` - Security settings
- `config/monitoring-config.json` - Monitoring configuration
- `config/ood-config.json` - OOD detection parameters
- `config/router-config.json` - Router settings

### For Deployment

#### Docker Deployment

The Docker setup is simplified:

```bash
# Build and run
docker-compose up --build

# Or with specific profile
docker-compose --profile monitoring up
```

No more version switching in Docker - the image contains the definitive version.

#### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export APP_ENV=production
export PYTHONPATH=$(pwd)/src

# Run API
python api/main.py
```

### For CI/CD

Update your CI/CD pipelines:

1. **Remove version management steps**
2. **Use simplified test commands**:
   ```bash
   pytest tests/ -v --cov=src
   ```
3. **Update artifact paths** - no more version-specific directories
4. **Use Git tags** for version tracking instead of directory structure

## Model Configuration

### DINOv2 (Default)

The default configuration uses DINOv2:

```json
{
  "backbone_model": "facebook/dinov2-base",
  "model_variant": "giant",
  "feature_dim": 1536
}
```

### DINOv3 Integration

To use DINOv3, update `config/adapter-spec.json`:

```json
{
  "backbone_model": "facebook/dinov3-base",
  "model_variant": "giant",
  "feature_dim": 1536,
  "dynamic_block_count": true
}
```

Then update requirements:

```bash
# Add DINOv3 dependencies
pip install git+https://github.com/facebookresearch/dinov3.git
```

## API Changes

All API endpoints remain the same:

- `GET /health` - Health check
- `POST /diagnose` - Disease diagnosis
- `GET /crops` - List supported crops
- `GET /metrics` - System metrics

No breaking changes to the API interface.

## Testing

Run the complete test suite:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v
```

## Rollback Procedures

If you need to revert to a previous state:

### Option 1: Git-Based Rollback

```bash
# View history
git log --oneline

# Checkout previous version (detached HEAD)
git checkout <commit-hash>

# Or create a branch
git checkout -b previous-version <commit-hash>
```

### Option 2: Restore from Backup Branch

If you have backup branches:

```bash
git checkout backup/backup-branch-name
```

### Option 3: Fresh Clone

```bash
# Clone repository fresh
git clone https://github.com/EfeErim/bitirmeprojesi.git
cd bitirme-projesi
```

## Troubleshooting

### Import Errors

If you get import errors, ensure:

1. `PYTHONPATH` includes the `src/` directory:
   ```bash
   export PYTHONPATH=$(pwd)/src
   ```

2. You're in the project root directory

3. All dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration Issues

Validate your configuration:

```bash
python scripts/config_utils.py validate
```

### Missing Version Directories

If code references `versions/` or `current/` directories:

1. These directories no longer exist
2. Update imports and paths to use unified structure
3. Check documentation for current structure

### Docker Volume Mounts

Update Docker volume mounts:

**OLD:**
```yaml
volumes:
  - ./current/config:/app/config:ro
```

**NEW:**
```yaml
volumes:
  - ./config:/app/config:ro
  - ./src:/app/src:ro
```

## Benefits of Consolidation

1. **Simpler Deployment** - No version switching complexity
2. **Easier Maintenance** - Single codebase to manage
3. **Better CI/CD** - Straightforward pipeline configuration
4. **Reduced Confusion** - Clear, definitive structure
5. **Git-Powered Versioning** - Use Git's built-in version control
6. **Faster Onboarding** - Easier for new developers to understand

## Support and Resources

- **Documentation**: See `docs/` directory
- **API Reference**: `docs/api/api-reference.md`
- **Architecture**: `docs/architecture/overview.md`
- **Deployment**: `DEPLOYMENT.md`
- **Issues**: Use GitHub Issues for problems
- **Discussions**: GitHub Discussions for questions

## Checklist for Migration

- [ ] Update local repository to consolidated structure
- [ ] Review and update `.env` configuration
- [ ] Update any import paths in your code
- [ ] Test API endpoints
- [ ] Update deployment scripts
- [ ] Update CI/CD pipeline configuration
- [ ] Train team on new structure
- [ ] Update internal documentation
- [ ] Archive old version-specific documentation

## Need Help?

1. Check the [FAQ](docs/development/faq.md) if available
2. Search existing [GitHub Issues](https://github.com/EfeErim/bitirmeprojesi/issues)
3. Create a new issue with details about your problem
4. Consult the [Architecture Overview](docs/architecture/overview.md) for system understanding

---

**Migration Date**: February 2026  
**Target Version**: AADS-ULoRA v5.5 (Consolidated)  
**Previous Structure**: Multi-version with `versions/`, `backups/`, `current/`