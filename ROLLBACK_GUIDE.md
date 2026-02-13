# Project Reorganization - Rollback Guide

## Overview
This guide provides step-by-step instructions to rollback the project reorganization changes if needed. The rollback process is designed to be safe and reversible.

## Prerequisites
- Git must be installed and configured
- You must have write access to the repository
- Ensure no uncommitted work exists before starting rollback

## Rollback Scenarios

### Scenario 1: Rollback Before Pushing (Local Changes Only)
If you haven't pushed the changes yet, rollback is straightforward:

#### Option A: Reset to Previous Commit (Discard All Changes)
```bash
# Check current status
git status

# Reset to the last committed state (WARNING: This discards all uncommitted changes)
git reset --hard HEAD

# Clean up untracked files and directories
git clean -fd

# Verify rollback
git status
```

#### Option B: Stash Changes (Preserve for Later)
```bash
# Stash all changes (including untracked files)
git stash push -u -m "Reorganization changes before push"

# Verify changes are stashed
git stash list

# To restore later if needed:
# git stash pop stash@{0}
```

### Scenario 2: Rollback After Pushing (Remote Changes)
If changes have been pushed to the remote repository:

#### Step 1: Create a Backup Branch (Safety First)
```bash
# Create a backup branch with current state before rollback
git branch backup/reorganization-$(date +%Y%m%d-%H%M%S)
git push origin backup/reorganization-$(date +%Y%m%d-%H%M%S)

# Verify backup branch exists
git branch -a
```

#### Step 2: Revert Commits on Main Branch
```bash
# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main

# Revert the reorganization commit (replace <commit-hash> with actual hash)
git revert <commit-hash> -m 1

# If there are multiple commits to revert:
# git revert <commit-hash-1> <commit-hash-2> ...

# Push the revert to remote
git push origin main
```

#### Step 3: Restore Deleted Files (If Needed)
If the revert doesn't fully restore deleted files from version directories:

```bash
# Check if any files are still missing
git status

# If files are missing, restore from backup branches:
# First, find the backup branch with the files
git checkout backup/reorganization-<timestamp> -- versions/v5.5.0-baseline/
git checkout backup/reorganization-<timestamp> -- versions/v5.5.1-ood/
git checkout backup/reorganization-<timestamp> -- versions/v5.5.4-dinov3/

# Commit the restored files
git add versions/
git commit -m "Restore version directories after rollback"
git push origin main
```

### Scenario 3: Partial Rollback (Selective Reversion)
If you only need to rollback specific parts of the reorganization:

#### Revert Specific Files
```bash
# Checkout specific files from the previous commit
git checkout HEAD^ -- path/to/file1 path/to/file2

# Or revert specific commits
git revert <commit-hash>

# Commit and push
git add .
git commit -m "Partial rollback: restore specific files"
git push origin main
```

## Detailed Rollback Steps for This Reorganization

### Step 1: Identify the Reorganization Commit
```bash
# View recent commit history
git log --oneline -10

# Look for the commit with message like:
# "feat: implement DINOv3 integration with enhanced router and ULoRA optimizations"
# or your reorganization commit message
```

### Step 2: Verify Current State
```bash
# Check what will be affected
git status
git diff HEAD~1 HEAD --stat
```

### Step 3: Execute Rollback Based on Your Situation

#### If You Haven't Pushed Yet:
```bash
# Simple reset to previous state
git reset --hard HEAD~1

# Remove all untracked files and directories
git clean -fd

# Verify
git status
```

#### If You Have Pushed:
```bash
# Create backup branch
git branch backup/before-rollback-$(date +%Y%m%d-%H%M%S)

# Revert the commit (preserving history)
git revert HEAD~1

# Resolve any merge conflicts if they occur
# After resolving conflicts:
git add .
git revert --continue

# Push the revert
git push origin main
```

## Restoring Version Directories

If you need to restore the `versions/` directory structure:

### Option A: From Backup Branch
```bash
# If you created a backup before rollback:
git checkout backup/reorganization-<timestamp> -- versions/

# Commit the restoration
git add versions/
git commit -m "Restore version directories from backup"
```

### Option B: From Git History
```bash
# Find the commit before reorganization
git log --oneline --all

# Checkout the versions directory from that commit
git checkout <commit-before-reorg> -- versions/

# Commit the restoration
git add versions/
git commit -m "Restore versions directory from commit <hash>"
```

## Verification After Rollback

### Check File Structure
```bash
# List directories to verify restoration
ls -la
ls -la versions/ 2>/dev/null || echo "versions directory not present"

# Check git status
git status

# Verify no unexpected changes
git diff --stat
```

### Test the Application
```bash
# Run tests to ensure functionality
pytest tests/unit/ -v

# Or run specific tests
python -m pytest tests/unit/test_adapter_comprehensive.py -v
```

## Emergency Recovery

If something goes wrong during rollback:

### Option 1: Use Git Reflog
```bash
# View all recent actions
git reflog

# Find the commit before the problematic operation
# Reset to that state
git reset --hard <reflog-entry-hash>
```

### Option 2: Restore from Remote Backup
```bash
# If you pushed a backup branch:
git checkout backup/reorganization-<timestamp>

# Create a new branch from the backup
git checkout -b recovery-branch

# Merge or cherry-pick as needed
```

## Rollback Checklist

- [ ] Identify the reorganization commit hash
- [ ] Create backup branch (if changes were pushed)
- [ ] Choose appropriate rollback method (reset vs revert)
- [ ] Execute rollback command
- [ ] Resolve any merge conflicts
- [ ] Verify file structure is restored
- [ ] Run tests to ensure functionality
- [ ] Push rollback changes (if applicable)
- [ ] Document the rollback in a new file or issue

## Important Notes

1. **Communication**: If working in a team, communicate rollback plans before executing
2. **Backup**: Always create a backup branch before rollback
3. **Testing**: Test thoroughly after rollback
4. **Documentation**: Update this guide or create a new one if rollback process differs
5. **History**: Git revert preserves history; git reset rewrites it. Use revert for shared branches.

## Contact
If you encounter issues during rollback, contact the project maintainer or consult the Git documentation.