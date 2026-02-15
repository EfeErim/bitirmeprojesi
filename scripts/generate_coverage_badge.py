#!/usr/bin/env python3
"""
AADS-ULoRA v5.5 Coverage Badge Generator

This script reads the coverage report and generates a shields.io badge
for the README and documentation.

Usage:
    python scripts/generate_coverage_badge.py [--output BADGE_FILE]

Options:
    --output FILE    Output file for badge (default: coverage-badge.svg)
    --threshold N    Coverage threshold for badge color (default: 80)
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_coverage_percentage():
    """Get coverage percentage from coverage.json or run tests."""
    coverage_json = project_root / "coverage.json"
    
    if not coverage_json.exists():
        print("Coverage data not found. Running tests...")
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--cov=src", "--cov-report=json"],
            cwd=project_root,
            check=False
        )
    
    if coverage_json.exists():
        with open(coverage_json, 'r') as f:
            coverage_data = json.load(f)
        return coverage_data['totals']['percent_covered']
    else:
        print("Error: Could not generate coverage data")
        return 0

def generate_badge_svg(coverage_percent, threshold=80):
    """Generate a shields.io badge SVG."""
    # Determine color based on coverage
    if coverage_percent >= threshold:
        color = "brightgreen"
    elif coverage_percent >= threshold * 0.8:
        color = "yellow"
    elif coverage_percent >= threshold * 0.6:
        color = "orange"
    else:
        color = "red"
    
    # Format percentage
    coverage_str = f"{coverage_percent:.1f}%"
    
    # Generate badge URL
    badge_url = (
        f"https://img.shields.io/badge/coverage-{coverage_str}-{color}"
        f"?style=flat-square&logo=python&logoColor=white"
    )
    
    # Download or create SVG
    try:
        import requests
        response = requests.get(badge_url)
        if response.status_code == 200:
            return response.text
        else:
            # Fallback to simple SVG
            return create_simple_badge(coverage_str, color)
    except ImportError:
        # Create simple badge without requests
        return create_simple_badge(coverage_str, color)

def create_simple_badge(text, color):
    """Create a simple SVG badge."""
    colors = {
        "brightgreen": "#4c1",
        "green": "#28a745",
        "yellow": "#ffd700",
        "orange": "#ff8c00",
        "red": "#dc3545"
    }
    
    bg_color = colors.get(color, "#4c1")
    text_color = "#fff" if color in ["brightgreen", "green"] else "#000"
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20">
  <linearGradient id="b" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="{bg_color}"/>
    <stop offset="100%" stop-color="{bg_color}" stop-opacity="0.9"/>
  </linearGradient>
  <mask id="a">
    <rect width="120" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path d="M0 0h60v20H0z" fill="{bg_color}"/>
    <path d="M60 0h60v20H60z" fill="#555"/>
  </g>
  <text x="30" y="14" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="{text_color}" text-anchor="middle">{text}</text>
  <text x="90" y="14" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">coverage</text>
</svg>'''
    
    return svg

def main():
    parser = argparse.ArgumentParser(
        description="Generate coverage badge for README"
    )
    parser.add_argument("--output", default="coverage-badge.svg",
                       help="Output file for badge (default: coverage-badge.svg)")
    parser.add_argument("--threshold", type=int, default=80,
                       help="Coverage threshold for badge color (default: 80)")
    
    args = parser.parse_args()
    
    # Get coverage percentage
    coverage_percent = get_coverage_percentage()
    
    if coverage_percent == 0:
        print("Warning: Could not determine coverage")
        return 1
    
    print(f"Coverage: {coverage_percent:.1f}%")
    
    # Generate badge
    badge_svg = generate_badge_svg(coverage_percent, args.threshold)
    
    # Save badge
    output_path = project_root / args.output
    with open(output_path, 'w') as f:
        f.write(badge_svg)
    
    print(f"Badge saved to: {output_path}")
    print(f"Badge URL: file://{output_path}")
    
    # Print markdown for README
    print("\nAdd this to your README.md:")
    print(f"![Coverage]({args.output})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())