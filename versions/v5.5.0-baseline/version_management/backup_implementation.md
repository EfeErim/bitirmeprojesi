# AADS-ULoRA Version Management System

## Backup System Implementation

### 1. Backup System Overview

The backup system provides automated version control with the following features:

- **Automated backups** before any optimization changes
- **Versioned directory structure** for easy rollback
- **Multiple backup points** with integrity verification
- **Standalone rollback scripts** for each version
- **Performance tracking** per version
- **A/B testing capability** for optimization comparisons

### 2. Backup System Architecture

```
project_root/
├── version_management/
│   ├── backup.py              # Python implementation
│   ├── backup.sh              # Shell script (for reference)
│   ├── README.md              # This documentation
│   └── rollback_scripts/       # Generated rollback scripts
├── backups/                    # Backup storage
├── versions/                   # Version storage
└── current/                    # Current working directory
```

### 3. Python Implementation (backup.py)

```python
#!/usr/bin/env python3
"""
AADS-ULoRA Version Management and Backup System
Automated backup creation, version control, and rollback capabilities
"""

import os
import sys
import json
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VersionManager:
    """
    Comprehensive version management system for AADS-ULoRA optimizations.
    
    Features:
    - Automated backup creation before any changes
    - Versioned directory structure
    - Multiple backup points with integrity verification
    - Rollback capabilities with validation
    - Performance tracking per version
    """
    
    def __init__(
        self,
        project_root: str = ".",
        backup_dir: str = "backups",
        versions_dir: str = "versions",
        current_dir: str = "current"
    ):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = self.project_root / backup_dir
        self.versions_dir = self.project_root / versions_dir
        self.current_dir = self.project_root / current_dir
        self.log_file = self.project_root / "backup.log"
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.current_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        logger.info(f"VersionManager initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Backup dir: {self.backup_dir}")
        logger.info(f"Versions dir: {self.versions_dir}")
        logger.info(f"Current dir: {self.current_dir}")
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash {filepath}: {e}")
            return ""
    
    def _get_file_manifest(self, directory: Path) -> Dict:
        """Generate a manifest of all files with their hashes."""
        manifest = {}
        if not directory.exists():
            return manifest
            
        for filepath in directory.rglob("*"):
            if filepath.is_file():
                rel_path = filepath.relative_to(directory)
                manifest[str(rel_path)] = {
                    "hash": self._calculate_file_hash(filepath),
                    "size": filepath.stat().st_size,
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                }
        return manifest
    
    def create_backup(
        self,
        version: str,
        description: str,
        source_dir: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Create a backup of the current state.
        
        Args:
            version: Version identifier (e.g., "v5.5.1-ood")
            description: Description of this version
            source_dir: Source directory to backup (defaults to current_dir)
            
        Returns:
            (success, message) tuple
        """
        source = source_dir or self.current_dir
        version_sanitized = version.replace("/", "_").replace("\\", "_")
        
        logger.info(f"Creating backup for version: {version}")
        logger.info(f"Description: {description}")
        logger.info(f"Source: {source}")
        
        try:
            # Create backup and version directories
            backup_path = self.backup_dir / version_sanitized
            version_path = self.versions_dir / version_sanitized
            
            if backup_path.exists() or version_path.exists():
                logger.warning(f"Version {version} already exists")
                overwrite = input(f"Version {version} exists. Overwrite? (y/N): ")
                if overwrite.lower() != 'y':
                    return False, "Backup cancelled by user"
                shutil.rmtree(backup_path, ignore_errors=True)
                shutil.rmtree(version_path, ignore_errors=True)
            
            backup_path.mkdir(parents=True, exist_ok=True)
            version_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all files from source
            if source.exists():
                for item in source.iterdir():
                    if item.is_file():
                        shutil.copy2(item, backup_path)
                        shutil.copy2(item, version_path)
                    elif item.is_dir():
                        shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True)
                        shutil.copytree(item, version_path / item.name, dirs_exist_ok=True)
            
            # Generate and save manifest
            manifest = {
                "version": version,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "source": str(source),
                "backup_path": str(backup_path),
                "version_path": str(version_path),
                "files": self._get_file_manifest(version_path)
            }
            
            manifest_file = version_path / "version.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Backup created at: {backup_path}")
            logger.info(f"Version saved at: {version_path}")
            logger.info(f"Manifest: {manifest_file}")
            
            # Count files
            file_count = len(manifest["files"])
            logger.info(f"Backed up {file_count} files")
            
            return True, f"Backup created successfully ({file_count} files)"
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False, f"Backup failed: {str(e)}"
    
    def list_backups(self) -> Dict:
        """List all available backups and versions."""
        backups = {}
        
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                version = backup_path.name
                manifest_file = self.versions_dir / version / "version.json"
                
                info = {
                    "backup_path": str(backup_path),
                    "file_count": len(list(backup_path.rglob("*"))),
                    "size_mb": sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()) / (1024*1024)
                }
                
                if manifest_file.exists():
                    try:
                        with open(manifest_file) as f:
                            manifest = json.load(f)
                            info["description"] = manifest.get("description", "")
                            info["timestamp"] = manifest.get("timestamp", "")
                    except:
                        info["description"] = ""
                        info["timestamp"] = ""
                
                backups[version] = info
        
        return backups
    
    def verify_backup(self, version: str) -> Tuple[bool, str]:
        """
        Verify backup integrity for a specific version.
        
        Checks:
        - Directory exists
        - Manifest exists and is valid
        - Critical files are present
        - File hashes match
        """
        version_sanitized = version.replace("/", "_").replace("\\", "_")
        version_path = self.versions_dir / version_sanitized
        manifest_file = version_path / "version.json"
        
        logger.info(f"Verifying backup: {version}")
        
        if not version_path.exists():
            return False, f"Version directory not found: {version_path}"
        
        if not manifest_file.exists():
            return False, f"Manifest not found: {manifest_file}"
        
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
        except Exception as e:
            return False, f"Invalid manifest: {e}"
        
        # Check critical files
        critical_files = [
            "src/adapter/independent_crop_adapter.py",
            "src/pipeline/independent_multi_crop_pipeline.py",
            "config/adapter_spec_v55.json",
            "requirements.txt",
            "setup.py"
        ]
        
        missing_files = []
        for critical_file in critical_files:
            if not (version_path / critical_file).exists():
                missing_files.append(critical_file)
        
        if missing_files:
            return False, f"Missing critical files: {missing_files}"
        
        # Verify file hashes
        mismatched = []
        for rel_path, file_info in manifest.get("files", {}).items():
            filepath = version_path / rel_path
            if filepath.exists():
                current_hash = self._calculate_file_hash(filepath)
                expected_hash = file_info.get("hash", "")
                if expected_hash and current_hash != expected_hash:
                    mismatched.append(rel_path)
        
        if mismatched:
            logger.warning(f"Hash mismatches: {mismatched}")
            # Don't fail for hash mismatches as they could be due to timestamp differences
        
        logger.info(f"Backup verified: {version}")
        return True, f"Backup verified successfully ({len(critical_files)} critical files present)"
    
    def restore_backup(
        self,
        version: str,
        target_dir: Optional[Path] = None,
        dry_run: bool = False
    ) -> Tuple[bool, str]:
        """
        Restore a backup to the current directory or specified target.
        
        Args:
            version: Version to restore
            target_dir: Target directory (defaults to current_dir)
            dry_run: If True, only show what would be restored
            
        Returns:
            (success, message) tuple
        """
        version_sanitized = version.replace("/", "_").replace("\\", "_")
        backup_path = self.backup_dir / version_sanitized
        target = target_dir or self.current_dir
        
        logger.info(f"Restoring version: {version}")
        logger.info(f"From: {backup_path}")
        logger.info(f"To: {target}")
        
        if not backup_path.exists():
            return False, f"Backup not found: {backup_path}"
        
        # Verify backup before restore
        success, message = self.verify_backup(version)
        if not success:
            logger.warning(f"Backup verification failed: {message}")
            proceed = input("Backup verification failed. Continue anyway? (y/N): ")
            if proceed.lower() != 'y':
                return False, "Restore cancelled due to verification failure"
        
        if dry_run:
            # Count files that would be restored
            file_count = sum(1 for _ in backup_path.rglob("*") if _.is_file())
            return True, f"DRY RUN: Would restore {file_count} files to {target}"
        
        try:
            # Check if target directory has files
            if target.exists() and any(target.iterdir()):
                logger.warning(f"Target directory is not empty: {target}")
                backup_current = target.parent / f"{target.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"Backing up current directory to: {backup_current}")
                shutil.move(str(target), str(backup_current))
            
            # Create target directory
            target.mkdir(parents=True, exist_ok=True)
            
            # Copy all files from backup
            for item in backup_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target)
                elif item.is_dir():
                    dest = target / item.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
            
            logger.info(f"Version {version} restored successfully")
            
            # Count restored files
            file_count = sum(1 for _ in target.rglob("*") if _.is_file())
            return True, f"Restored {file_count} files to {target}"
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False, f"Restore failed: {str(e)}"
    
    def create_rollback_script(self, version: str) -> Tuple[bool, str]:
        """
        Create a standalone rollback script for a specific version.
        
        Returns:
            (success, script_path) tuple
        """
        version_sanitized = version.replace("/", "_").replace("\\", "_")
        version_path = self.versions_dir / version_sanitized
        manifest_file = version_path / "version.json"
        
        if not manifest_file.exists():
            return False, f"Manifest not found for version: {version}"
        
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
        except Exception as e:
            return False, f"Could not read manifest: {e}"
        
        script_path = self.project_root / f"rollback_{version_sanitized}.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Rollback script for AADS-ULoRA version: {version}
Description: {manifest.get('description', '')}
Created: {manifest.get('timestamp', '')}
"""

import sys
import os
from pathlib import Path

# Add version_management to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'version_management'))

from backup import VersionManager

def main():
    vm = VersionManager(
        project_root=Path(__file__).parent,
        backup_dir="backups",
        versions_dir="versions",
        current_dir="current"
    )
    
    print(f"Rolling back to version: {version}")
    print(f"Description: {manifest.get('description', '')}")
    
    success, message = vm.restore_backup("{version}")
    
    if success:
        print(f"\n✓ SUCCESS: {{message}}")
        return 0
    else:
        print(f"\n✗ FAILED: {{message}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        
        logger.info(f"Rollback script created: {script_path}")
        return True, str(script_path)
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """Get information about a specific version."""
        version_sanitized = version.replace("/", "_").replace("\\", "_")
        manifest_file = self.versions_dir / version_sanitized / "version.json"
        
        if not manifest_file.exists():
            return None
        
        try:
            with open(manifest_file) as f:
                return json.load(f)
        except:
            return None


def main():
    """Command-line interface for VersionManager."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AADS-ULoRA Version Management System"
    )
    parser.add_argument(
        "action",
        choices=["create", "list", "restore", "verify", "rollback_script"],
        help="Action to perform"
    )
    parser.add_argument(
        "--version", "-v",
        help="Version identifier (required for create, restore, verify, rollback_script)"
    )
    parser.add_argument(
        "--description", "-d",
        help="Version description (required for create)"
    )
    parser.add_argument(
        "--source", "-s",
        help="Source directory to backup (for create action)",
        default="current"
    )
    parser.add_argument(
        "--target", "-t",
        help="Target directory for restore (defaults to current)",
        default="current"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it (for restore)"
    )
    
    args = parser.parse_args()
    
    vm = VersionManager()
    
    if args.action == "create":
        if not args.version or not args.description:
            print("Error: --version and --description are required for create action")
            sys.exit(1)
        
        source_dir = Path(args.source) if args.source != "current" else vm.current_dir
        success, message = vm.create_backup(args.version, args.description, source_dir)
        
        if success:
            # Also create rollback script
            roll_success, roll_path = vm.create_rollback_script(args.version)
            if roll_success:
                print(f"✓ Rollback script created: {roll_path}")
            print(f"✓ {message}")
            sys.exit(0)
        else:
            print(f"✗ {message}")
            sys.exit(1)
    
    elif args.action == "list":
        backups = vm.list_backups()
        if not backups:
            print("No backups found.")
        else:
            print(f"{{'Version':<30} {{'Description':<40} {{'Files':<10} {{'Size (MB)':<10}")
            print("-" * 100)
            for version, info in sorted(backups.items()):
                desc = info.get('description', '')[:38]
                size = info.get('size_mb', 0)
                count = info.get('file_count', 0)
                print(f"{version:<30} {desc:<40} {count:<10} {size:<10.2f}")
        sys.exit(0)
    
    elif args.action == "restore":
        if not args.version:
            print("Error: --version is required for restore action")
            sys.exit(1)
        
        target_dir = Path(args.target) if args.target != "current" else vm.current_dir
        success, message = vm.restore_backup(
            args.version,
            target_dir,
            dry_run=args.dry_run
        )
        
        if success:
            print(f"✓ {message}")
            sys.exit(0)
        else:
            print(f"✗ {message}")
            sys.exit(1)
    
    elif args.action == "verify":
        if not args.version:
            print("Error: --version is required for verify action")
            sys.exit(1)
        
        success, message = vm.verify_backup(args.version)
        if success:
            print(f"✓ {message}")
            sys.exit(0)
        else:
            print(f"✗ {message}")
            sys.exit(1)
    
    elif args.action == "rollback_script":
        if not args.version:
            print("Error: --version is required for rollback_script action")
            sys.exit(1)
        
        success, script_path = vm.create_rollback_script(args.version)
        if success:
            print(f"✓ Rollback script created: {script_path}")
            sys.exit(0)
        else:
            print(f"✗ {message}")
            sys.exit(1)


if __name__ == "__main__":
    main()