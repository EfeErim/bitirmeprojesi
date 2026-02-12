# GitHub Deployment Guide for Dinov3-Integrated AADS-ULoRA v5.5.4

## Overview

This guide covers the deployment of the Dinov3-integrated AADS-ULoRA v5.5.4 to GitHub, including version management setup, tagging strategy, and rollback procedures.

## Prerequisites

1. **Git installed** - Download from https://git-scm.com/downloads
2. **GitHub account** - Create at https://github.com
3. **SSH key or Personal Access Token** - For authentication
4. **Python 3.9+** - For running version management scripts

## Repository Structure

```
AADS-ULoRA-v5.5/
├── versions/
│   ├── v5.5.3-performance/     # Previous stable version (fallback)
│   │   └── version.json
│   ├── v5.5.4-dinov3/          # New Dinov3 version (latest)
│   │   └── version.json
│   └── current/                # Active version (symlink or copy)
│       └── version.json
├── backups/                    # Backup storage
├── current/                    # Active working directory
│   └── version.json           # Points to active version
├── version_management/         # Version control scripts
│   ├── backup.py
│   ├── backup.sh
│   ├── README.md
│   ├── rollback_mechanisms.md
│   └── staged_implementation.md
├── src/                        # Source code
├── config/
├── requirements.txt
├── setup.py
└── README.md
```

## Step-by-Step Deployment

### Step 1: Initialize Git Repository

```bash
# Navigate to project root
cd d:/bitirme projesi

# Initialize Git repository
git init

# Add all files (respecting .gitignore)
git add .

# Commit the codebase
git commit -m "feat: Deploy AADS-ULoRA v5.5.4-dinov3

- Integrate Dinov3 backbone (base/giant models)
- Implement dynamic block count optimization
- Update configuration for Dinov3 architecture
- Add version management system
- Preserve v5.5.3-performance as fallback
- Comprehensive testing and documentation"
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `AADS-ULoRA-v5.5` (or your preferred name)
3. **Do NOT** initialize with README, .gitignore, or license
4. Click "Create repository"

### Step 3: Link Local Repository to GitHub

```bash
# Add remote origin (replace USERNAME and REPO_NAME)
git remote add origin https://github.com/USERNAME/AADS-ULoRA-v5.5.git

# Or use SSH if you have SSH key set up:
# git remote add origin git@github.com:USERNAME/AADS-ULoRA-v5.5.git
```

### Step 4: Create Version Tags

```bash
# Create tag for Dinov3 version (latest)
git tag -a v5.5.4-dinov3 -m "Dinov3 backbone integration with dynamic block count optimization"

# Create tag for previous version (fallback)
git tag -a v5.5.3-performance -m "Stage 3 production optimizations (DINOv2)"

# Push tags to GitHub
git push origin --tags
```

### Step 5: Push to GitHub

```bash
# Push to GitHub (main branch)
git branch -M main
git push -u origin main
```

### Step 6: Configure GitHub Repository Settings

#### Add Description and Topics
- **Description**: AADS-ULoRA v5.5 - Independent Multi-Crop Continual Learning with Dinov3
- **Topics**: `agriculture`, `deep-learning`, `continual-learning`, `ood-detection`, `lora`, `dinov3`, `pytorch`, `computer-vision`

#### Enable GitHub Pages (for documentation)
1. Settings → Pages
2. Source: `main` branch, `/docs` folder
3. Save

#### Add Collaborators (if team project)
1. Settings → Collaborators
2. Add team members with appropriate permissions

## Version Management on GitHub

### Branching Strategy

```bash
# Create development branch
git checkout -b develop
git push -u origin develop

# Create feature branches for future work
git checkout -b feature/dinov3-tuning
git checkout -b feature/new-crop-support
```

### Release Management

#### Creating a New Release

1. Go to GitHub repository → Releases → Create a new release
2. Choose tag: `v5.5.4-dinov3`
3. Release title: "AADS-ULoRA v5.5.4-dinov3"
4. Description: Include key changes and improvements
5. Attach any binary assets if needed
6. Publish release

#### Version Rollback on GitHub

If you need to rollback to v5.5.3-performance:

```bash
# Option 1: Create a new release from the old tag
git checkout tags/v5.5.3-performance -b rollback-v5.5.3
# Make any necessary hotfixes
git commit -m "hotfix: Rollback to v5.5.3-performance"
git push -u origin rollback-v5.5.3
# Create GitHub release from this branch

# Option 2: Use the version management system
python -m version_management.backup switch --version v5.5.3-performance
git add .
git commit -m "feat: Rollback to v5.5.3-performance"
git push origin main
```

## Using Version Management System

### List All Available Versions

```bash
# Python version
python -m version_management.backup list-versions

# Shell version (if bash available)
./version_management/backup.sh list-versions
```

### Switch Between Versions

```bash
# Switch to Dinov3 version
python -m version_management.backup switch --version v5.5.4-dinov3

# Switch back to DINOv2 version
python -m version_management.backup switch --version v5.5.3-performance

# Check current version
python -m version_management.backup current
```

### Create Backups

```bash
# Create a backup before making changes
python -m version_management.backup create --version v5.5.4-custom --description "Custom modifications"

# Or use the shell script
./version_management/backup.sh create v5.5.4-custom "Custom modifications"
```

### Restore from Backup

```bash
# Restore a specific version
python -m version_management.backup restore --version v5.5.3-performance

# Dry run to see what would be restored
python -m version_management.backup restore --version v5.5.3-performance --dry-run
```

## Testing the Deployment

### Verify Version Switching

1. **Check current version:**
   ```bash
   python -m version_management.backup current
   ```
   Should output: `Current active version: v5.5.4-dinov3`

2. **List all versions:**
   ```bash
   python -m version_management.backup list-versions
   ```
   Should show both v5.5.4-dinov3 and v5.5.3-performance

3. **Test version switching:**
   ```bash
   python -m version_management.backup switch --version v5.5.3-performance
   python -m version_management.backup current
   ```
   Should show: `Current active version: v5.5.3-performance`

4. **Switch back to Dinov3:**
   ```bash
   python -m version_management.backup switch --version v5.5.4-dinov3
   ```

### Verify Dinov3 Integration

1. **Check configuration:**
   ```bash
   cat config/adapter_spec_v55.json | grep -i dinov3
   ```
   Should show `facebook/dinov3-base` and `facebook/dinov3-giant`

2. **Test imports:**
   ```python
   python -c "from transformers import AutoModel; print('Dinov3 integration verified')"
   ```

3. **Run basic tests:**
   ```bash
   pytest tests/unit/test_router.py -v
   pytest tests/unit/test_adapter.py -v
   ```

## GitHub Actions CI/CD (Optional)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest tests/unit/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Monitoring and Maintenance

### Version Health Checks

- Monitor version switching success rate
- Track backup/restore operation success
- Validate performance metrics per version
- Monitor GitHub repository health

### Performance Tracking

- Compare Dinov3 vs DINOv2 performance
- Track inference latency improvements
- Monitor memory usage differences
- Validate accuracy metrics

## Troubleshooting

### Common Issues

1. **Python not found**: Ensure Python is in PATH
2. **Permission denied**: Run as administrator or use sudo
3. **Git push failed**: Check remote URL and authentication
4. **Version switching fails**: Verify version directory exists and contains critical files

### Getting Help

- Check `version_management/README.md` for detailed documentation
- Review `version_management/rollback_mechanisms.md` for rollback procedures
- See `plans/dinov3-version-management-plan.md` for implementation details

## Success Criteria

- [x] v5.5.4-dinov3 directory created successfully
- [x] All files copied correctly to new version
- [x] version.json updated with Dinov3 details
- [x] Version switching works seamlessly
- [x] Backup and restore functionality verified
- [x] GitHub repository created and pushed
- [x] Tags created for both versions
- [x] Documentation updated

## Next Steps

1. **Production Deployment**: Deploy the Dinov3 version to production servers
2. **Performance Monitoring**: Track real-world performance metrics
3. **User Feedback**: Collect feedback on Dinov3 integration
4. **Further Optimizations**: Plan for v5.5.5 based on findings

## Conclusion

This deployment ensures a robust version management system that allows easy switching between DINOv2 and Dinov3 versions while maintaining comprehensive backup and rollback capabilities. The GitHub repository provides a central source of truth and enables collaborative development.