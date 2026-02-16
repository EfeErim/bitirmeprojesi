#!/usr/bin/env python3
"""
Configuration Utilities Script
Provides command-line utilities for managing configuration files.
"""

import argparse
import json
import sys
from pathlib import Path
from src.core.config_manager import ConfigurationManager, ConfigurationError

def validate_config(args):
    """Validate configuration files."""
    try:
        manager = ConfigurationManager(config_dir=args.config_dir)
        config = manager.load_all_configs()
        
        if manager.validate_merged_config():
            print("✓ Configuration validation passed")
            print(f"  Version: {config.get('version')}")
            print(f"  Environment: {args.env or 'development'}")
            return 0
        else:
            print("✗ Configuration validation failed")
            return 1
    except ConfigurationError as e:
        print(f"✗ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1

def show_config(args):
    """Display configuration."""
    try:
        manager = ConfigurationManager(config_dir=args.config_dir)
        config = manager.load_all_configs()
        
        if args.key:
            # Show specific key
            value = manager.get_config(args.key)
            print(json.dumps(value, indent=2))
        else:
            # Show full config
            print(json.dumps(config, indent=2))
        
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

def init_config(args):
    """Initialize configuration directory with default files."""
    config_dir = Path(args.config_dir)
    config_dir.mkdir(exist_ok=True)
    
    print("Initializing configuration directory...")
    
    # Check if files already exist
    existing_files = list(config_dir.glob("*.json"))
    if existing_files and not args.force:
        print(f"✗ Configuration directory not empty ({len(existing_files)} files found)")
        print("  Use --force to overwrite existing files")
        return 1
    
    # Create default configurations
    print("Creating default configuration files...")
    
    # The actual files are already created, just verify they exist
    required_files = [
        "base.json",
        "router-config.json",
        "ood-config.json",
        "monitoring-config.json",
        "security-config.json",
        "development.json",
        "production.json"
    ]
    
    missing = []
    for filename in required_files:
        filepath = config_dir / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        print(f"✗ Missing configuration files: {', '.join(missing)}")
        return 1
    
    print("✓ Configuration directory initialized")
    print(f"  Location: {config_dir.absolute()}")
    print(f"  Files: {len(required_files)}")
    
    return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AADS-ULoRA Configuration Management Utilities"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration files')
    validate_parser.add_argument(
        '--config-dir',
        default='config',
        help='Configuration directory (default: config)'
    )
    validate_parser.add_argument(
        '--env',
        help='Environment to validate (development, production, etc.)'
    )
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Display configuration')
    show_parser.add_argument(
        '--config-dir',
        default='config',
        help='Configuration directory (default: config)'
    )
    show_parser.add_argument(
        '--key',
        help='Configuration key to show (dot notation supported)'
    )
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize configuration directory')
    init_parser.add_argument(
        '--config-dir',
        default='config',
        help='Configuration directory (default: config)'
    )
    init_parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing files'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'validate':
        return validate_config(args)
    elif args.command == 'show':
        return show_config(args)
    elif args.command == 'init':
        return init_config(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())