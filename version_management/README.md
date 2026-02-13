# AADS-ULoRA Version Management System

## Version Naming Convention

**Format:** `v[MAJOR].[MINOR].[PATCH]-[OPTIMIZATION]`

### Version Components
- **MAJOR**: Breaking changes or major architecture updates
- **MINOR**: New features or significant optimizations
- **PATCH**: Bug fixes or minor improvements
- **OPTIMIZATION**: Specific optimization type (optional)

### Examples
- `v5.5.0` - Base version
- `v5.5.1-ood` - OOD detection optimization
- `v5.6.0-router` - Router architecture improvement
- `v6.0.0` - Major architecture change

## Version Control Strategy

### 1. Git-Based Version Control
```bash
# Create version tags
git tag -a v5.5.1-ood -m "OOD detection optimization"
git push origin v5.5.1-ood

# List all versions
git tag --list

# Checkout specific version
git checkout tags/v5.5.1-ood -b v5.5.1-ood-branch
```

### 2. Directory-Based Versioning
```
versions/
├── v5.5.0/
│   ├── src/
│   ├── config/
│   └── requirements.txt
├── v5.5.1-ood/
│   ├── src/
│   ├── config/
│   └── requirements.txt
└── current/
    ├── src/
    ├── config/
    └── requirements.txt
```

## Backup System Implementation

### Automated Backup Script
```bash
#!/bin/bash
# backup.sh - Automated backup system

# Configuration
BACKUP_DIR="backups"
VERSION_DIR="versions"
CURRENT_DIR="current"
LOG_FILE="backup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to create backup
create_backup() {
    local version=$1
    local description=$2
    
    log "Creating backup for version: $version"
    log "Description: $description"
    
    # Create backup directory
    local backup_path="$BACKUP_DIR/$version"
    mkdir -p "$backup_path"
    
    # Copy current files
    cp -r "$CURRENT_DIR/"* "$backup_path/"
    
    # Create version directory
    local version_path="$VERSION_DIR/$version"
    mkdir -p "$version_path"
    cp -r "$CURRENT_DIR/"* "$version_path/"
    
    log "Backup created at: $backup_path"
    log "Version saved at: $version_path"
    
    # Create version manifest
    cat > "$version_path/version.json" << EOF
{
    "version": "$version",
    "description": "$description",
    "timestamp": "$(date -Iseconds)",
    "files": [
        $(find "$CURRENT_DIR" -type f | sed 's|'$CURRENT_DIR'/||g' | awk '{print "\""$0"\""}' | paste -sd,)
    ]
}
EOF
    
    log "Version manifest created: $version_path/version.json"
}

# Function to list backups
list_backups() {
    log "Listing all backups..."
    ls -la "$BACKUP_DIR"
    echo ""
    log "Listing all versions..."
    ls -la "$VERSION_DIR"
}

# Function to restore backup
restore_backup() {
    local version=$1
    
    log "Restoring version: $version"
    
    local backup_path="$BACKUP_DIR/$version"
    if [ ! -d "$backup_path" ]; then
        log "${RED}Error: Backup not found for version $version${NC}"
        return 1
    fi
    
    # Clear current directory
    rm -rf "$CURRENT_DIR/*"
    
    # Restore files
    cp -r "$backup_path/"* "$CURRENT_DIR/"
    
    log "Version $version restored successfully"
}

# Function to create rollback script
create_rollback_script() {
    local version=$1
    local description=$2
    
    log "Creating rollback script for version: $version"
    
    local rollback_script="rollback_$version.sh"
    cat > "$rollback_script" << EOF
#!/bin/bash
# Rollback script for version: $version
# Description: $description

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Rolling back to version: $version"
echo "Description: $description"

# Restore backup
./backup.sh restore $version

if [ $? -eq 0 ]; then
    echo "${GREEN}Rollback to version $version completed successfully${NC}"
else
    echo "${RED}Rollback to version $version failed${NC}"
fi
EOF
    
    chmod +x "$rollback_script"
    log "Rollback script created: $rollback_script"
}

# Main execution
case "${1:-}" in
    "create")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 create <version> <description>"
            exit 1
        fi
        create_backup "$2" "$3"
        create_rollback_script "$2" "$3"
        ;;
    "list")
        list_backups
        ;;
    "restore")
        if [ -z "$2" ]; then
            echo "Usage: $0 restore <version>"
            exit 1
        fi
        restore_backup "$2"
        ;;
    *)
        echo "Usage: $0 {create|list|restore} [args]"
        echo ""
        echo "Commands:"
        echo "  create <version> <description>  Create backup for version"
        echo "  list                            List all backups"
        echo "  restore <version>               Restore specific version"
        ;;
esac