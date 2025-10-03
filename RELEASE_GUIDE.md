# PropFlow Release Guide
## Step-by-Step Instructions for Publishing to PyPI

---

## Prerequisites

### 1. PyPI Account Setup
You need accounts on both TestPyPI (for testing) and PyPI (production).

#### Create Accounts
- **TestPyPI** (testing): https://test.pypi.org/account/register/
- **PyPI** (production): https://pypi.org/account/register/

#### Enable 2FA (Required)
1. Go to Account Settings
2. Enable Two-Factor Authentication
3. Save recovery codes in a safe place

#### Create API Tokens

**For TestPyPI**:
1. Go to https://test.pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `propflow-test`
4. Scope: "Entire account" (or specific to project later)
5. Copy the token (starts with `pypi-`)
6. **Save it immediately** - you can't view it again!

**For PyPI**:
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `propflow-release`
4. Scope: "Entire account"
5. Copy and save the token

---

## Step 1: Pre-Release Checks

### Verify Package Quality
```bash
# 1. Check version is correct
cat src/propflow/_version.py
# Should show: __version__ = "0.1.0"

# 2. Run a quick test of core functionality
uv run python -c "from propflow import BPEngine, FGBuilder; print('✓ Imports work')"

# 3. Test an example
uv run python examples/quick_start.py

# 4. Verify no sensitive data in repo
git status
git diff
```

### Clean Build Environment
```bash
# Remove old build artifacts
rm -rf dist/ build/ src/*.egg-info/

# Verify clean state
ls dist/  # Should show: No such file or directory
```

---

## Step 2: Build the Package

```bash
# Build wheel and source distribution
uv build

# Verify build succeeded
ls -lh dist/
# Should show:
# propflow-0.1.0-py3-none-any.whl
# propflow-0.1.0.tar.gz
```

### Validate the Build
```bash
# Check package metadata and integrity
uv run twine check dist/*

# Should output:
# Checking dist/propflow-0.1.0-py3-none-any.whl: PASSED
# Checking dist/propflow-0.1.0.tar.gz: PASSED
```

---

## Step 3: Test on TestPyPI (IMPORTANT!)

### Upload to TestPyPI
```bash
# Upload using your TestPyPI token
uv run twine upload --repository testpypi dist/*

# You'll be prompted for:
# Username: __token__
# Password: pypi-xxxxx (your TestPyPI token)
```

**Alternative**: Create `~/.pypirc` to avoid typing credentials:
```ini
[testpypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxx  # Your TestPyPI token

[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxx  # Your PyPI token
```

Then you can just run:
```bash
uv run twine upload --repository testpypi dist/*
```

### Verify Upload
1. Go to https://test.pypi.org/project/propflow/
2. Check that version 0.1.0 appears
3. Review the package description (README)
4. Check metadata (dependencies, classifiers, etc.)

### Test Installation from TestPyPI
```bash
# Create a fresh virtual environment for testing
python3 -m venv test_env
source test_env/bin/activate

# Install from TestPyPI
# Note: We need --extra-index-url because dependencies come from real PyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            propflow

# Test the installation
python -c "from propflow import BPEngine, FGBuilder; print('✓ TestPyPI install works!')"

# Try running an example
python -c "
from propflow import FactorGraph, VariableAgent, FactorAgent, BPEngine
from propflow.configs import CTFactory
import numpy as np

v1 = VariableAgent('v1', domain=2)
v2 = VariableAgent('v2', domain=2)
f = FactorAgent('f', domain=2, ct_creation_func=CTFactory.random_int.fn, param={'low': 0, 'high': 10})
fg = FactorGraph([v1, v2], [f], edges={f: [v1, v2]})
engine = BPEngine(factor_graph=fg)
engine.run(max_iter=5)
print(f'✓ Engine ran successfully! Cost: {engine.calculate_global_cost()}')
"

# Clean up test environment
deactivate
rm -rf test_env
```

### If TestPyPI Test Fails
If you find issues:
```bash
# 1. Fix the issues in your code
# 2. Update version (required for new upload)
# Edit src/propflow/_version.py: __version__ = "0.1.0.post1"
# 3. Rebuild
rm -rf dist/ build/
uv build
# 4. Re-upload to TestPyPI
uv run twine upload --repository testpypi dist/*
```

---

## Step 4: Publish to Production PyPI

**⚠️ WARNING**: Once uploaded to PyPI, you **CANNOT** delete or re-upload the same version!

### Final Checks
```bash
# 1. Verify version is exactly what you want
cat src/propflow/_version.py

# 2. Double-check the built packages
ls -lh dist/
uv run twine check dist/*

# 3. Make sure you're uploading the right files
# dist/propflow-0.1.0-py3-none-any.whl
# dist/propflow-0.1.0.tar.gz
```

### Upload to PyPI
```bash
# Upload to production PyPI
uv run twine upload dist/*

# You'll be prompted for:
# Username: __token__
# Password: pypi-xxxxx (your PyPI token)
```

### Verify Publication
1. Go to https://pypi.org/project/propflow/
2. Verify version 0.1.0 is live
3. Check package page looks correct

### Test Installation from PyPI
```bash
# In a fresh environment
python3 -m venv verify_env
source verify_env/bin/activate

# Install from PyPI (real this time!)
pip install propflow

# Test it works
python -c "from propflow import BPEngine; print('✓ PyPI install successful!')"

# Clean up
deactivate
rm -rf verify_env
```

---

## Step 5: Post-Release Tasks

### Tag the Release in Git
```bash
# Create a git tag for this version
git tag -a v0.1.0 -m "Release version 0.1.0"

# Push the tag to GitHub
git push origin v0.1.0
```

### Create GitHub Release
1. Go to https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/releases
2. Click "Create a new release"
3. Choose tag: `v0.1.0`
4. Release title: `PropFlow v0.1.0 - Initial Release`
5. Description: Copy from CHANGELOG.md
6. Click "Publish release"

### Update README Badges
Add PyPI badge to README.md:
```markdown
[![PyPI version](https://badge.fury.io/py/propflow.svg)](https://badge.fury.io/py/propflow)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
```

### Update Installation Instructions
Update README.md:
```markdown
## Installation

```bash
pip install propflow
```

For development installation:
```bash
git clone https://github.com/OrMullerHahitti/Belief-Propagation-Simulator.git
cd Belief-Propagation-Simulator
pip install -e ".[dev]"
```

### Announce the Release
Consider announcing on:
- GitHub Discussions
- Twitter/X
- Reddit (r/Python, r/MachineLearning if appropriate)
- Your institution/research group
- Relevant mailing lists

---

## Troubleshooting

### "File already exists" Error
This means you've already uploaded this version. You must:
1. Increment version number
2. Rebuild
3. Upload new version

### "Invalid credentials" Error
- Make sure you're using `__token__` as username
- Check your API token is correct
- Verify token hasn't been revoked

### Package Description Not Rendering
- Check README.md is valid Markdown
- Verify `readme = {file = "README.md", content-type = "text/markdown"}` in pyproject.toml

### Dependencies Not Installing
- Check dependency specifications in pyproject.toml
- Verify all dependencies exist on PyPI
- Test with `pip install propflow` in fresh environment

### Import Errors After Install
- Verify package structure is correct
- Check `__init__.py` exports
- Test with: `python -c "import propflow; print(propflow.__version__)"`

---

## Quick Reference Commands

### Complete Release Workflow
```bash
# 1. Clean and build
rm -rf dist/ build/ src/*.egg-info/
uv build

# 2. Validate
uv run twine check dist/*

# 3. Upload to TestPyPI
uv run twine upload --repository testpypi dist/*

# 4. Test from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ propflow

# 5. Upload to PyPI (if test passed)
uv run twine upload dist/*

# 6. Tag release
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

---

## For Future Releases (v0.1.1, v0.2.0, etc.)

### Version Update Checklist
1. Update `src/propflow/_version.py`
2. Update `pyproject.toml` version
3. Update `CHANGELOG.md` with changes
4. Run full test suite: `pytest`
5. Follow this guide from Step 1

### Semantic Versioning Guide
- **Patch** (0.1.0 → 0.1.1): Bug fixes, no API changes
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Major** (0.1.0 → 1.0.0): Breaking API changes

---

## Security Notes

### API Token Security
- **NEVER** commit tokens to git
- Store in `~/.pypirc` with permissions `chmod 600 ~/.pypirc`
- Revoke and regenerate if compromised
- Use project-scoped tokens after first upload

### .pypirc Template
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxx

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxx
```

---

## Getting Help

- **PyPI Support**: https://pypi.org/help/
- **Packaging Guide**: https://packaging.python.org/
- **Twine Docs**: https://twine.readthedocs.io/
- **GitHub Issues**: For PropFlow-specific questions

---

**Good luck with your release! 🚀**
