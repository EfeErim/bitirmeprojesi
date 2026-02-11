#!/bin/bash
# AADS-ULoRA Backup System
# Automated backup creation with fallback mechanisms

# Configuration
BACKUP_DIR="backups"
VERSION_DIR="versions"
CURRENT_DIR="current"
LOG_FILE="backup.log"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
BACKUP_BRANCH="backup-$TIMESTAMP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    
    # Create version manifest
    cat > "$version_path/version.json" << EOF
{
    "version": "$version",
    "description": "$description",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_path": "$backup_path",
    "files_count": $(find "$backup_path" -type f | wc -l),
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
}
EOF
    
    log "Backup created at: $backup_path"
    log "Version saved at: $version_path"
    
    # Create git backup branch
    git checkout -b "$BACKUP_BRANCH" 2>/dev/null || git checkout "$BACKUP_BRANCH"
    git add .
    git commit -m "feat: backup $version - $description" 2>/dev/null || true
    git push -u origin "$BACKUP_BRANCH" 2>/dev/null || true
    
    log "Git backup branch created: $BACKUP_BRANCH"
}

# Function to verify backup integrity
verify_backup() {
    local backup_path=$1
    
    log "Verifying backup integrity at: $backup_path"
    
    # Check if directory exists
    if [ ! -d "$backup_path" ]; then
        log "${RED}ERROR: Backup directory does not exist${NC}"
        return 1
    fi
    
    # Check file count
    local file_count=$(find "$backup_path" -type f | wc -l)
    log "Files in backup: $file_count"
    
    # Check for critical files
    local critical_files=0
    for file in "setup.py" "requirements.txt" "README.md" "src/"; do
        if [ -e "$backup_path/$file" ]; then
            ((critical_files++))
        fi
    done
    
    if [ $critical_files -lt 3 ]; then
        log "${YELLOW}WARNING: Some critical files missing in backup${NC}"
    else
        log "${GREEN}CRITICAL FILES VERIFIED${NC}"
    fi
    
    return 0
}

# Function to restore from backup
restore_backup() {
    local backup_path=$1
    
    log "Restoring from backup: $backup_path"
    
    # Check if backup exists
    if [ ! -d "$backup_path" ]; then
        log "${RED}ERROR: Backup directory does not exist${NC}"
        return 1
    fi
    
    # Clear current directory
    rm -rf "$CURRENT_DIR"/*
    
    # Restore files
    cp -r "$backup_path/"* "$CURRENT_DIR/"
    
    log "${GREEN}RESTORE COMPLETED${NC} - Files restored from $backup_path"
    
    # Commit restored state
    git add .
    git commit -m "feat: restore from backup $backup_path" 2>/dev/null || true
    
    return 0
}

# Function to handle push failures
safe_push() {
    local branch=$1
    local remote=$2
    
    log "Attempting safe push to $remote/$branch"
    
    # Try normal push first
    if git push "$remote" "$branch" 2>/dev/null; then
        log "${GREEN}Push successful${NC}"
        return 0
    fi
    
    # If push fails, create backup branch
    local backup_branch="backup-push-failed-$TIMESTAMP"
    git checkout -b "$backup_branch"
    git push -u "$remote" "$backup_branch"
    
    log "${YELLOW}Push failed - created fallback branch: $backup_branch${NC}"
    log "${BLUE}Manual intervention required${NC} - Push to $backup_branch"
    
    return 1
}

# Main execution
main() {
    log "${BLUE}AADS-ULoRA Backup System Starting${NC}"
    log "Timestamp: $TIMESTAMP"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log "${RED}ERROR: Not in a git repository${NC}"
        exit 1
    fi
    
    # Get current git status
    local git_status=$(git status --porcelain)
    
    if [ -n "$git_status" ]; then
        log "${YELLOW}Uncommitted changes detected${NC}"
        log "Stashing changes..."
        git stash push -m "backup-stash-$TIMESTAMP"
    fi
    
    # Create backup
    local version="v$(date +"%Y%m%d_%H%M%S")-auto-backup"
    local description="Automated backup before synchronization"
    
    create_backup "$version" "$description"
    
    # Verify backup
    if verify_backup "$BACKUP_DIR/$version"; then
        log "${GREEN}Backup verification successful${NC}"
    else
        log "${RED}Backup verification failed${NC}"
        exit 1
    fi
    
    # Restore stashed changes if any
    if [ -n "$git_status" ]; then
        log "Restoring stashed changes..."
        git stash pop 2>/dev/null || true
    fi
    
    log "${GREEN}Backup completed successfully${NC}"
    log "Backup branch: $BACKUP_BRANCH"
    log "Backup path: $BACKUP_DIR/$version"
    
    # Show summary
    echo ""
    echo "${BLUE}=== BACKUP SUMMARY ===${NC}"
    echo "Version: $version"
    echo "Description: $description"
    echo "Timestamp: $TIMESTAMP"
    echo "Backup Path: $BACKUP_DIR/$version"
    echo "Git Branch: $BACKUP_BRANCH"
    echo "Files Backed Up: $(find "$BACKUP_DIR/$version" -type f | wc -l)"
    echo "${BLUE}======================${NC}"
}

# Execute main function
main "$@"
