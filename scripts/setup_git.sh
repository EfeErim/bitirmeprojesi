#!/bin/bash

# AADS-ULoRA v5.5 - Git Repository Initialization Script
# Run this script to initialize Git and make the first commit

set -e  # Exit on error

echo "=========================================="
echo "AADS-ULoRA v5.5 Git Setup"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed. Please install Git from https://git-scm.com/downloads"
    exit 1
fi

echo "✓ Git is installed: $(git --version)"
echo ""

# Initialize git repository
if [ -d .git ]; then
    echo "Git repository already initialized."
else
    echo "Initializing Git repository..."
    git init
    echo "✓ Git repository initialized."
fi

# Add all files
echo "Adding files to Git..."
git add .
echo "✓ Files added."
echo ""npm install -g @google/gemini-cli

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: AADS-ULoRA v5.5 implementation

- Complete independent multi-crop architecture
- Dynamic OOD detection with per-class thresholds
- DoRA, SD-LoRA, CONEC-LoRA implementations
- FastAPI backend with mobile integration
- Comprehensive tests and documentation"
echo "✓ Initial commit created."
echo ""

echo "=========================================="
echo "Git repository is ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub: https://github.com/new"
echo "2. Do NOT initialize with README, .gitignore, or license"
echo "3. After creating, link your local repo:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "4. Push to GitHub:"
echo "   git branch -M main"
