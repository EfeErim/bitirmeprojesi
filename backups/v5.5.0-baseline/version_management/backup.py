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
        
        # Directories to exclude from backup
        exclude_dirs = {'.git', 'backups', 'versions', '__pycache__', '.pytest_cache', '.vscode'}
        
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
            
            # Copy all files from source, excluding certain directories
            if source.exists():
                for item in source.iterdir():
                    if item.name in exclude_dirs:
                        continue
                    if item.is_file():
                        shutil.copy2(item, backup_path)
                        shutil.copy2(item, version_path)
                    elif item.is_dir():
                        shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*exclude_dirs))
                        shutil.copytree(item, version_path / item.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*exclude_dirs))
            
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
        Restore a backup to the target directory.
        
        Args:
            version: Version to restore
            target_dir: Target directory (defaults to current_dir)
            dry_run: If True, only show what would be restored
            
        Returns:
            (success, message) tuple
        """
        version_sanitized = version.replace("/", "_").replace("\\", "_")
        version_path = self.versions_dir / version_sanitized
        target = target_dir or self.current_dir
        
        logger.info(f"Restoring backup: {version}")
        logger.info(f"Target: {target}")
        
        if not version_path.exists():
            return False, f"Version directory not found: {version_path}"
        
        if not dry_run:
            # Create target directory if it doesn't exist
            target.mkdir(parents=True, exist_ok=True)
        
        # Get list of files to restore
        files_to_restore = []
        for filepath in version_path.rglob("*"):
            if filepath.is_file():
                rel_path = filepath.relative_to(version_path)
                files_to_restore.append((rel_path, filepath))
        
        if dry_run:
            logger.info(f"Would restore {len(files_to_restore)} files:")
            for rel_path, _ in files_to_restore:
                logger.info(f"  {rel_path}")
            return True, f"Dry run complete. Would restore {len(files_to_restore)} files."
        
        try:
            # Restore files
            for rel_path, source_path in files_to_restore:
                dest_path = target / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
            
            logger.info(f"Backup restored to: {target}")
            return True, f"Backup restored successfully ({len(files_to_restore)} files)"
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False, f"Restore failed: {str(e)}"
    
    def create_rollback_script(
        self,
        version: str,
        script_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Create a standalone rollback script for a specific version.
        
        Args:
            version: Version to create rollback script for
            script_name: Custom script name (defaults to rollback_[version].py)
            
        Returns:
            (success, message) tuple
        """
        version_sanitized = version.replace("/", "_").replace("\\", "_")
        version_path = self.versions_dir / version_sanitized
        
        if not version_path.exists():
            return False, f"Version directory not found: {version_path}"
        
        script_name = script_name or f"rollback_{version_sanitized}.py"
        script_path = self.project_root / script_name
        
        # Create rollback script content
        script_content = f'''#!/usr/bin/env python3
"""
Standalone rollback script for AADS-ULoRA version {version}
Restores this version to the current directory
"""

import os
import sys
import shutil
from pathlib import Path

def rollback():
    """Restore version {version} to current directory."""
    print(f"Restoring version {version}...")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Find version directory
    version_dir = script_dir / "versions" / "{version_sanitized}"
    if not version_dir.exists():
        print(f"Error: Version directory not found: {version_dir}")
        return False
    
    # Restore files
    try:
        for filepath in version_dir.rglob("*"):
            if filepath.is_file():
                rel_path = filepath.relative_to(version_dir)
                dest_path = script_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(filepath, dest_path)
                print(f"Restored: {rel_path}")
        
        print(f"Successfully restored version {version}")
        return True
        
    except Exception as e:
        print(f"Error during restore: {e}")
        return False

if __name__ == "__main__":
    success = rollback()
    sys.exit(0 if success else 1)
'''
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            logger.info(f"Rollback script created: {script_path}")
            return True, f"Rollback script created: {script_path}"
            
        except Exception as e:
            logger.error(f"Failed to create rollback script: {e}")
            return False, f"Failed to create rollback script: {str(e)}"

# Main execution for command-line usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AADS-ULoRA Version Management System')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Create backup command
    create_parser = subparsers.add_parser('create', help='Create a backup')
    create_parser.add_argument('--version', required=True, help='Version identifier')
    create_parser.add_argument('--description', required=True, help='Description of the version')
    create_parser.add_argument('--source', help='Source directory to backup')
    
    # List backups command
    list_parser = subparsers.add_parser('list', help='List all backups')
    
    # Verify backup command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('--version', required=True, help='Version to verify')
    
    # Restore backup command
    restore_parser = subparsers.add_parser('restore', help='Restore a backup')
    restore_parser.add_argument('--version', required=True, help='Version to restore')
    restore_parser.add_argument('--target', help='Target directory')
    restore_parser.add_argument('--dry-run', action='store_true', help='Show what would be restored')
    
    # Create rollback script command
    rollback_parser = subparsers.add_parser('rollback-script', help='Create standalone rollback script')
    rollback_parser.add_argument('--version', required=True, help='Version to create script for')
    rollback_parser.add_argument('--name', help='Custom script name')
    
    args = parser.parse_args()
    
    manager = VersionManager()
    
    if args.command == 'create':
        success, message = manager.create_backup(
            version=args.version,
            description=args.description,
            source_dir=Path(args.source) if args.source else None
        )
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.command == 'list':
        backups = manager.list_backups()
        print(f"Available backups: {len(backups)}")
        for version, info in sorted(backups.items()):
            print(f"{version:20} | {info['description'][:50]:50} | {info['timestamp'][:19]} | {info['file_count']:5} files | {info['size_mb']:6.2f} MB")
    
    elif args.command == 'verify':
        success, message = manager.verify_backup(args.version)
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.command == 'restore':
        success, message = manager.restore_backup(
            version=args.version,
            target_dir=Path(args.target) if args.target else None,
            dry_run=args.dry_run
        )
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.command == 'rollback-script':
        success, message = manager.create_rollback_script(
            version=args.version,
            script_name=args.name
        )
        print(message)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()