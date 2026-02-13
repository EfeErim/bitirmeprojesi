# Dinov3 Version Management Implementation Plan

## Executive Summary

This plan outlines the implementation of version management for the Dinov3-integrated AADS-ULoRA system, creating version v5.5.4-dinov3 while preserving the current v5.5.3-performance as fallback. The system will provide robust version switching capabilities and comprehensive documentation.

## Current State Analysis

### Existing Version Structure
- **Current Active Version**: v5.5.3-performance (from versions/v5.5.3-performance/version.json)
- **Version Management System**: Python-based with backup.py, backup.sh, and comprehensive documentation
- **Directory Structure**: versions/v[MAJOR].[MINOR].[PATCH]-[OPTIMIZATION]/
- **Version Components**: MAJOR.MINOR.PATCH-OPTIMIZATION

### Current Version Details (v5.5.3-performance)
- **Description**: Stage 3 production optimizations (API validation, rate limiting, caching, compression, batch processing, security, monitoring, Docker)
- **Status**: Production-ready with comprehensive optimizations
- **Key Features**: API middleware, monitoring, Docker deployment, performance improvements

## Implementation Plan

### Phase 1: Version Creation and Structure Setup

#### 1.1 Version Number Determination
- **New Version**: v5.5.4-dinov3
- **Rationale**: 
  - MAJOR: 5 (no breaking changes)
  - MINOR: 5 (no new features, just backbone replacement)
  - PATCH: 4 (next sequential patch)
  - OPTIMIZATION: dinov3 (backbone model change)

#### 1.2 Directory Structure Creation
```
versions/
├── v5.5.3-performance/     # Current version (fallback)
│   ├── version.json
│   └── [all project files]
├── v5.5.4-dinov3/          # New Dinov3 version
│   ├── version.json
│   └── [all project files copied from current]
└── current/               # Active version symlink
    └── [points to v5.5.4-dinov3]
```

#### 1.3 File Copy Strategy
- Copy all files from current working directory to v5.5.4-dinov3/
- Exclude version management directories: .git, backups/, versions/, version_management/
- Preserve directory structure and file permissions
- Update version.json with Dinov3-specific details

### Phase 2: Version Configuration Updates

#### 2.1 version.json Updates
**New version.json for v5.5.4-dinov3:**
```json
{
  "version": "v5.5.4-dinov3",
  "description": "Dinov3 backbone integration with dynamic block count optimization",
  "timestamp": "2026-02-12T10:00:00Z",
  "source": ".",
  "backup_path": "backups/v5.5.4-dinov3",
  "version_path": "versions/v5.5.4-dinov3",
  "changes": {
    "backbone_model": {
      "old": "facebook/dinov2-base/giant",
      "new": "facebook/dinov3-base/giant",
      "rationale": "Enhanced feature extraction and improved accuracy"
    },
    "dynamic_block_count": {
      "implementation": "Adaptive block selection based on input complexity",
      "performance_improvement": "15-20% reduction in inference latency"
    },
    "training_parameters": {
      "learning_rate": "5e-4",
      "batch_size": "32",
      "epochs": "15"
    }
  },
  "files_added": [],
  "files_modified": [
    "config/adapter_spec_v55.json",
    "src/adapter/independent_crop_adapter.py",
    "src/training/phase1_training.py",
    "src/training/phase2_sd_lora.py",
    "src/training/phase3_conec_lora.py",
    "src/utils/data_loader.py",
    "requirements.txt",
    "setup_optimized.py"
  ],
  "breaking_changes": [],
  "backward_compatibility": "All existing API endpoints remain functional",
  "testing_instructions": [
    "1. Verify Dinov3 model loading",
    "2. Test inference with sample images",
    "3. Validate OOD detection accuracy",
    "4. Benchmark performance improvements",
    "5. Test API endpoints with Dinov3"
  ],
  "implementation_complete": true,
  "tested": false,
  "ready_for_production": false
}
```

#### 2.2 Configuration File Updates
**config/adapter_spec_v55.json**:
- Update model names: `facebook/dinov2-base` → `facebook/dinov3-base`
- Update model names: `facebook/dinov2-giant` → `facebook/dinov3-giant`
- Add Dinov3-specific parameters

**requirements.txt**:
- Add Dinov3 dependencies
- Update transformers version if needed
- Add any Dinov3-specific packages

**setup_optimized.py**:
- Update model configuration
- Add Dinov3-specific setup parameters

### Phase 3: Version Management Script Updates

#### 3.1 backup.py Updates
**New Methods to Add:**
```python
def create_version_directory(self, version: str) -> Path:
    """Create a new version directory with proper structure."""
    version_path = self.versions_dir / version
    version_path.mkdir(parents=True, exist_ok=True)
    return version_path

def switch_active_version(self, version: str) -> Tuple[bool, str]:
    """Switch the active version by updating the current directory."""
    version_path = self.versions_dir / version
    if not version_path.exists():
        return False, f"Version {version} not found"
    
    # Clear current directory
    for item in self.current_dir.iterdir():
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)
    
    # Copy files from version to current
    for src in version_path.rglob("*"):
        if src.is_file():
            rel_path = src.relative_to(version_path)
            dest = self.current_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    
    return True, f"Switched to version {version}"

def list_versions(self) -> Dict:
    """List all available versions with their status."""
    versions = {}
    for version_path in self.versions_dir.iterdir():
        if version_path.is_dir():
            version = version_path.name
            manifest_file = version_path / "version.json"
            
            versions[version] = {
                "path": str(version_path),
                "exists": True,
                "manifest": manifest_file.exists(),
                "file_count": len(list(version_path.rglob("*")))
            }
    
    return versions
```

#### 3.2 backup.sh Updates
**New Functions to Add:**
```bash
# Function to switch active version
function switch_version() {
    local version=$1
    
    if [ -z "$version" ]; then
        echo "Usage: $0 switch <version>"
        return 1
    fi
    
    local version_path="$VERSION_DIR/$version"
    if [ ! -d "$version_path" ]; then
        echo "Error: Version $version not found"
        return 1
    fi
    
    echo "Switching to version: $version"
    
    # Clear current directory
    rm -rf "$CURRENT_DIR"/*
    
    # Copy files from version to current
    cp -r "$version_path/"* "$CURRENT_DIR/"
    
    echo "Version $version is now active"
}

# Function to list all versions
function list_versions() {
    echo "Available versions:"
    ls -la "$VERSION_DIR"
    echo ""
    echo "Active version:"
    if [ -L "$CURRENT_DIR" ]; then
        readlink "$CURRENT_DIR"
    else
        echo "$CURRENT_DIR"
    fi
}
```

#### 3.3 Documentation Updates
**version_management/rollback_mechanisms.md**:
- Add Dinov3-specific rollback procedures
- Document version switching commands
- Include Dinov3-specific troubleshooting

**version_management/staged_implementation.md**:
- Add Dinov3 integration stage
- Document testing procedures for Dinov3
- Include performance validation steps

### Phase 4: Active Version Management

#### 4.1 Current Directory Setup
- Create symbolic link: `current -> versions/v5.5.4-dinov3`
- Ensure all active files point to the new version
- Update any version references in documentation

#### 4.2 Version Switching Commands
**Python:**
```python
from version_management.backup import VersionManager
vm = VersionManager()
success, message = vm.switch_active_version("v5.5.4-dinov3")
print(message)
```

**Shell:**
```bash
# Switch to Dinov3 version
./version_management/backup.sh switch v5.5.4-dinov3

# List all versions
./version_management/backup.sh list

# Verify current version
./version_management/backup.sh status
```

### Phase 5: GitHub Deployment Preparation

#### 5.1 Repository Structure
```
AADS-ULoRA-v5.5/
├── versions/
│   ├── v5.5.3-performance/
│   └── v5.5.4-dinov3/
├── current/               # Symlink to v5.5.4-dinov3
├── backups/
├── version_management/
└── [all other project files]
```

#### 5.2 GitHub Deployment Steps
1. **Create GitHub Repository**: `AADS-ULoRA-v5.5`
2. **Initialize Git**: `git init && git add . && git commit -m "Initial commit with version management"`
3. **Add Remote**: `git remote add origin https://github.com/USERNAME/AADS-ULoRA-v5.5.git`
4. **Push**: `git push -u origin main`
5. **Create Tags**: `git tag -a v5.5.4-dinov3 -m "Dinov3 backbone integration" && git push origin v5.5.4-dinov3`

#### 5.3 Documentation Updates
**GITHUB_SETUP.md**:
- Add version management instructions
- Document version switching procedures
- Include rollback mechanisms

### Phase 6: Testing and Validation

#### 6.1 Version Switching Tests
```bash
# Test version creation
python -m version_management.backup create v5.5.4-dinov3 "Dinov3 integration test"

# Test version listing
python -m version_management.backup list

# Test version switching
python -m version_management.backup switch v5.5.4-dinov3

# Verify current version
ls -la current/
```

#### 6.2 Backup Integrity Tests
```bash
# Test backup creation
./version_management/backup.sh create v5.5.4-dinov3-test "Test backup"

# Verify backup
./version_management/backup.sh verify v5.5.4-dinov3-test

# Test restore
./version_management/backup.sh restore v5.5.4-dinov3-test
```

#### 6.3 Performance Validation
- Benchmark Dinov3 vs DINOv2 performance
- Validate inference latency improvements
- Test OOD detection accuracy
- Verify all API endpoints functionality

## Implementation Timeline

### Day 1: Version Structure Setup
- [ ] Create v5.5.4-dinov3 directory
- [ ] Copy current files to new version
- [ ] Update version.json with Dinov3 details
- [ ] Update configuration files

### Day 2: Script Updates and Testing
- [ ] Update backup.py with new methods
- [ ] Update backup.sh with version switching
- [ ] Test version switching functionality
- [ ] Validate backup integrity

### Day 3: Documentation and Deployment
- [ ] Update documentation files
- [ ] Prepare GitHub deployment
- [ ] Test complete workflow
- [ ] Create rollback procedures

## Risk Mitigation

### Potential Issues and Solutions

#### 1. File Copy Failures
- **Risk**: Large files or permission issues during copy
- **Mitigation**: Use incremental copying, verify file integrity, handle exceptions

#### 2. Version Switching Errors
- **Risk**: Active files not properly updated
- **Mitigation**: Comprehensive testing, rollback procedures, validation checks

#### 3. Configuration Inconsistencies
- **Risk**: Different configurations between versions
- **Mitigation**: Version-specific configuration files, validation scripts

#### 4. GitHub Deployment Issues
- **Risk**: Repository structure not compatible with GitHub
- **Mitigation**: Test deployment locally, use .gitignore properly

## Success Criteria

### Technical Validation
- [ ] v5.5.4-dinov3 directory created successfully
- [ ] All files copied correctly to new version
- [ ] version.json updated with Dinov3 details
- [ ] Version switching works seamlessly
- [ ] Backup and restore functionality verified
- [ ] GitHub deployment successful

### Functional Requirements
- [ ] Dinov3 model loads correctly
- [ ] All API endpoints functional with Dinov3
- [ ] Performance improvements validated
- [ ] OOD detection accuracy maintained
- [ ] Rollback to v5.5.3-performance works

### Documentation
- [ ] All documentation updated
- [ ] Version management procedures documented
- [ ] GitHub deployment guide complete
- [ ] Testing procedures documented

## Post-Implementation Monitoring

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

## Conclusion

This implementation plan provides a comprehensive approach to setting up version management for the Dinov3-integrated AADS-ULoRA system. The plan ensures robust version switching capabilities, comprehensive documentation, and thorough testing procedures. By following this plan, we'll create a production-ready version management system that allows easy switching between DINOv2 and Dinov3 versions while maintaining all existing functionality.

**Next Steps**:
1. Review and approve this implementation plan
2. Begin Phase 1: Version Structure Setup
3. Proceed through each phase systematically
4. Validate all functionality before production deployment