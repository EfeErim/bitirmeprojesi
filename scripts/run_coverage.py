#!/usr/bin/env python3
"""
AADS-ULoRA v5.5 Coverage Report Generator

This script runs the test suite with coverage reporting and generates
comprehensive coverage reports in multiple formats.

Usage:
    python scripts/run_coverage.py [options]

Options:
    --html       Generate HTML coverage report (default: True)
    --xml        Generate XML coverage report for CI/CD
    --json       Generate JSON coverage report
    --report     Show coverage report in terminal
    --min-coverage PERCENT  Minimum coverage threshold (default: 75)
    --fail-under PERCENT    Fail if coverage below threshold (default: 75)
    --parallel   Run tests in parallel (requires pytest-xdist)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_coverage(args):
    """Run pytest with coverage."""
    
    # Build command
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-config=.coveragerc"
    ]
    
    # Add coverage report formats
    if args.html:
        cmd.append("--cov-report=html")
    if args.xml:
        cmd.append("--cov-report=xml")
    if args.json:
        cmd.append("--cov-report=json")
    if args.report:
        cmd.append("--cov-report=term")
    
    # Add parallel execution if requested
    if args.parallel:
        try:
            import pytest_xdist
            cmd.append("-n auto")
        except ImportError:
            print("Warning: pytest-xdist not installed, running sequentially")
    
    # Add fail under threshold
    if args.fail_under:
        cmd.append(f"--cov-fail-under={args.fail_under}")
    
    # Add verbose if requested
    if args.verbose:
        cmd.append("-v")
    
    # Add show outputs
    if args.show_outputs:
        cmd.append("-s")
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the command
    result = subprocess.run(cmd, cwd=project_root)
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(
        description="AADS-ULoRA v5.5 Coverage Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all reports with default settings
    python scripts/run_coverage.py

    # Generate only HTML report
    python scripts/run_coverage.py --no-xml --no-json

    # Set minimum coverage to 80%
    python scripts/run_coverage.py --fail-under 80

    # Run tests in parallel
    python scripts/run_coverage.py --parallel

    # Show detailed output
    python scripts/run_coverage.py --verbose --show-outputs
        """
    )
    
    parser.add_argument("--html", action="store_true", default=True,
                       help="Generate HTML coverage report (default: True)")
    parser.add_argument("--no-html", dest="html", action="store_false",
                       help="Disable HTML coverage report")
    parser.add_argument("--xml", action="store_true", default=True,
                       help="Generate XML coverage report (default: True)")
    parser.add_argument("--no-xml", dest="xml", action="store_false",
                       help="Disable XML coverage report")
    parser.add_argument("--json", action="store_true", default=False,
                       help="Generate JSON coverage report")
    parser.add_argument("--report", action="store_true", default=True,
                       help="Show coverage report in terminal")
    parser.add_argument("--no-report", dest="report", action="store_false",
                       help="Hide terminal coverage report")
    parser.add_argument("--min-coverage", type=int, default=75,
                       help="Minimum coverage threshold (default: 75)")
    parser.add_argument("--fail-under", type=int, default=75,
                       help="Fail if coverage below threshold (default: 75)")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel using pytest-xdist")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--show-outputs", action="store_true",
                       help="Show test outputs")
    
    args = parser.parse_args()
    
    # Ensure reports directory exists
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Run coverage
    exit_code = run_coverage(args)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COVERAGE REPORT SUMMARY")
    print("=" * 80)
    
    if args.html:
        html_report = project_root / "htmlcov" / "index.html"
        if html_report.exists():
            print(f"✅ HTML report generated: file://{html_report}")
        else:
            print("❌ HTML report not found")
    
    if args.xml:
        xml_report = project_root / "coverage.xml"
        if xml_report.exists():
            print(f"✅ XML report generated: {xml_report}")
        else:
            print("❌ XML report not found")
    
    if args.json:
        json_report = project_root / "coverage.json"
        if json_report.exists():
            print(f"✅ JSON report generated: {json_report}")
        else:
            print("❌ JSON report not found")
    
    print(f"\nMinimum coverage threshold: {args.fail_under}%")
    print(f"Exit code: {exit_code}")
    
    if exit_code == 0:
        print("✅ All tests passed and coverage requirements met")
    else:
        print("❌ Tests failed or coverage below threshold")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())