# AADS-ULoRA v5.5 - GitHub Repository Setup Guide

This guide will help you push this codebase to GitHub.

## Prerequisites

1. **Git installed** - Download from https://git-scm.com/downloads
2. **GitHub account** - Create at https://github.com
3. **SSH key or Personal Access Token** - For authentication

---

## Step 1: Initialize Git Repository

Open terminal in the project root (`d:/bitirme projesi`):

```bash
# Initialize Git repository
git init

# Add all files (respecting .gitignore)
git add .

# Commit the codebase
git commit -m "Initial commit: AADS-ULoRA v5.5 implementation

- Complete independent multi-crop architecture
- Dynamic OOD detection with per-class thresholds
- DoRA, SD-LoRA, CONEC-LoRA implementations
- FastAPI backend with mobile integration
- Comprehensive tests and documentation"
```

---

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `AADS-ULoRA-v5.5` (or your preferred name)
3. **Do NOT** initialize with README, .gitignore, or license
4. Click "Create repository"

---

## Step 3: Link Local Repository to GitHub

After creating the repo, GitHub will show you instructions. Use:

```bash
# Add remote origin (replace USERNAME and REPO_NAME)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# Or use SSH if you have SSH key set up:
# git remote add origin git@github.com:USERNAME/REPO_NAME.git
```

---

## Step 4: Push to GitHub

```bash
# Push to GitHub (main branch)
git branch -M main
git push -u origin main
```

---

## Step 5: Verify

Refresh your GitHub repository page. All code should be visible.

---

## Optional: GitHub Repository Settings

### Add Description and Topics
- **Description**: AADS-ULoRA v5.5 - Independent Multi-Crop Continual Learning with Dynamic OOD Detection
- **Topics**: `agriculture`, `deep-learning`, `continual-learning`, `ood-detection`, `lora`, `dino`, `pytorch`, `computer-vision`

### Enable GitHub Pages (for documentation)
1. Settings → Pages
2. Source: `main` branch, `/docs` folder
3. Save

### Add Collaborators (if team project)
1. Settings → Collaborators
2. Add team members with appropriate permissions

---

## Recommended .gitignore (Already Included)

The project already includes a `.gitignore` file that excludes:
- Python cache files
- Virtual environments
- Large model files (*.pth, *.pt, *.ckpt)
- Training data and outputs
- Mobile build artifacts
- IDE configuration files
- Sensitive configuration (API keys, etc.)

---

## Branching Strategy (Optional)

For production development, consider:

```bash
# Create development branch
git checkout -b develop

# Push develop branch
git push -u origin develop

# Feature branches
git checkout -b feature/new-crop-support
```

---

## GitHub Actions CI/CD (Optional)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
```

---

## Troubleshooting

### Error: "fatal: not a git repository"
Make sure you ran `git init` in the correct directory.

### Error: "remote origin already exists"
Remove and re-add:
```bash
git remote remove origin
git remote add origin https://github.com/USERNAME/REPO_NAME.git
```

### Authentication failed
- Use SSH instead of HTTPS, or
- Create a Personal Access Token: GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic) with `repo` scope

### Large files rejected
GitHub has a 100MB file limit. The `.gitignore` should exclude model checkpoints. If you have large files already staged:
```bash
git rm --cached large_file.pth
```

---

## Post-Setup Checklist

- [ ] Repository created on GitHub
- [ ] Local repo initialized and committed
- [ ] Remote origin added
- [ ] Code pushed successfully
- [ ] README visible on GitHub
- [ ] Team members added as collaborators (if applicable)
- [ ] Branch protection rules configured (if needed)
- [ ] GitHub Pages enabled for docs (optional)

---

## Need Help?

- GitHub Docs: https://docs.github.com/en
- Git Cheatsheet: https://education.github.com/git-cheat-sheet-education.pdf
- AADS-ULoRA Documentation: See `docs/` folder
