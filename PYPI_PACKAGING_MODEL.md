# PropFlow Packaging & PyPI Delivery Model

This document models every step needed to turn the PropFlow codebase into a distributable Python package and deliver it to both TestPyPI and PyPI. It captures the exact tooling, metadata, and validation workflow used in this project so future releases remain reproducible.

---

## 1. Source Layout & Export Surface

PropFlow follows the modern `src/` layout. All package modules live beneath `src/propflow`, and every subpackage exported on PyPI must include an `__init__.py` so `setuptools` discovers it.

```
src/
  propflow/
    __init__.py
    _version.py
    bp/
    configs/
    simulator.py
    ...
```

### 1.1 Public API (`__init__.py`)
`src/propflow/__init__.py` re-exports the public surface, including the version string. Keep this file ASCII and annotate with brief comments only when behavior is non-obvious.

```python
from ._version import __version__
from .bp.engine_base import BPEngine
from .bp.factor_graph import FactorGraph
from .core import VariableAgent, FactorAgent
from .utils import FGBuilder
from .configs import (
    CTFactory,
    create_random_int_table,
    create_uniform_float_table,
    create_poisson_table,
)
from .snapshots import SnapshotsConfig, SnapshotManager

__all__ = [
    "__version__",
    "BPEngine",
    "FactorGraph",
    "VariableAgent",
    "FactorAgent",
    "FGBuilder",
    "CTFactory",
    "create_random_int_table",
    "create_uniform_float_table",
    "create_poisson_table",
    "SnapshotsConfig",
    "SnapshotManager",
]
```

### 1.2 Single Source of Version Truth
`src/propflow/_version.py` stores the release identifier. Update this file and `pyproject.toml` together for every release increment.

```python
__version__ = "0.1.1"
```

---

## 2. pyproject.toml Configuration

`pyproject.toml` centralizes metadata, build backend configuration, dependencies, CLI entry points, and optional extras.

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[project]
name = "propflow"
version = "0.1.1"
description = "Highly configurable belief propagation simulator for factor graphs"
readme = {file = "README.md", content-type = "text/markdown"}
authors = [{name = "Or Muller"}]
license = "MIT"
requires-python = ">=3.10"
keywords = ["belief-propagation", "factor-graphs", "graphical-models", "inference", "optimization"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Framework :: Matplotlib",
]
dependencies = [
    "networkx~=3.4.2",
    "numpy~=2.2.1",
    "matplotlib~=3.10.0",
    "scipy~=1.15.2",
    "dash~=2.18.2",
    "dash_cytoscape~=1.0.2",
    "colorlog~=6.9.0",
    "pandas~=2.2.3",
    "seaborn~=0.13.2",
    "psutil~=7.0.0",
    "twine>=6.2.0",
]

[project.urls]
Homepage = "https://github.com/OrMullerHahitti/Belief-Propagation-Simulator"
Repository = "https://github.com/OrMullerHahitti/Belief-Propagation-Simulator"
Issues = "https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/issues"
Documentation = "https://github.com/OrMullerHahitti/Belief-Propagation-Simulator#readme"

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "mypy",
    "black",
    "pylint",
    "jupyter",
    "build",
    "twine",
    "pre-commit"
]

[project.scripts]
bp-sim = "propflow.cli:main"
```

Key practices:
- Keep runtime dependencies minimal and exact (PropFlow pins via compatible release `~=`).
- Provide a `dev` extra that mirrors pre-commit hooks and developer tooling.
- Set `bp-sim` CLI entry point for packaging the console script.

---

## 3. MANIFEST & Ancillary Assets

`MANIFEST.in` ensures non-code assets ship with the sdist:

```text
include README.md
include LICENSE
include CHANGELOG.md
include CONTRIBUTING.md
include requirements.txt
recursive-include examples *.py *.ipynb
recursive-include configs *.py *.md
prune configs/logs
prune .vscode
prune .venv
```

Confirm referenced files exist at the repository root. Update the manifest whenever new documentation or configuration directories must reach the source distribution.

---

## 4. Environment Preparation & Quality Gates

1. **Bootstrap environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. **Static checks**
   ```bash
   flake8
   mypy src
   black . --check
   ```
3. **Test suite & coverage**
   ```bash
   pytest -q
   pytest --cov=src --cov-report=term-missing
   ```
4. **Smoke CLI & examples**
   ```bash
   bp-sim --help
   python examples/minsum_basic.py
   ```
5. **Ensure clean tree**
   ```bash
   git status
   git diff
   ```

---

## 5. Build Artifacts

Always start from a clean build directory:

```bash
rm -rf dist/ build/ src/*.egg-info/
```

PropFlow uses `uv` internally, but the standard library approach works as well. Choose one consistent method per release:

```bash
# Preferred project flow
uv build

# or vanilla build module
python -m build
```

Inspect the output:

```bash
ls -lh dist/
# Expect:
# propflow-0.1.0-py3-none-any.whl
# propflow-0.1.0.tar.gz
```

Validate metadata before uploading:

```bash
uv run twine check dist/*
# or python -m twine check dist/*
```

---

## 6. Credentials & `~/.pypirc`

Create API tokens on both TestPyPI and PyPI after enabling 2FA. Store them in `~/.pypirc` with restrictive permissions (`chmod 600 ~/.pypirc`).

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

Never commit tokens or `.pypirc` to the repository.

---

## 7. Upload & Verification Workflow

### 7.1 Publish to TestPyPI

```bash
uv run twine upload --repository testpypi dist/*
```

- Username: `__token__`
- Password: TestPyPI API token (`pypi-...`)

Verify the staging release:
1. Visit <https://test.pypi.org/project/propflow/> and confirm version, description, and metadata.
2. Test installation from a clean environment:
   ```bash
   python3 -m venv test_env
   source test_env/bin/activate
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ \
               propflow
   python -c "from propflow import BPEngine; print('✓ TestPyPI install works')"
   deactivate
   rm -rf test_env
   ```

If issues arise, address them, bump the version (`0.1.0` → `0.1.0.post1`), rebuild, and re-upload to TestPyPI.

### 7.2 Publish to PyPI

After TestPyPI validation:

```bash
uv run twine upload dist/*
```

Repeat the clean-environment install check:

```bash
python3 -m venv verify_env
source verify_env/bin/activate
pip install propflow
python -c "from propflow import BPEngine; print('✓ PyPI install successful')"
deactivate
rm -rf verify_env
```

Remember: PyPI rejects duplicate version uploads. Increment both `_version.py` and `pyproject.toml` before rebuilding if anything fails after publication.

---

## 8. Post-Release Activities

1. **Git tagging**
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```
2. **GitHub release**
   - Title: `PropFlow v0.1.0`
   - Body: summarize changes from `CHANGELOG.md`
   - Attach release artifacts if desired (already available on PyPI).
3. **README updates**
   ```markdown
   [![PyPI version](https://badge.fury.io/py/propflow.svg)](https://badge.fury.io/py/propflow)
   [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

   ## Installation
   ```bash
   pip install propflow
   ```
   ```
4. **Announcement**
   Share the release on GitHub Discussions, social channels, mailing lists, or research groups.

---

## 9. Change Control & Future Versions

Follow semantic versioning:
- **Patch**: bug fixes (`0.1.0 → 0.1.1`)
- **Minor**: backwards-compatible features (`0.1.0 → 0.2.0`)
- **Major**: breaking changes (`0.1.0 → 1.0.0`)

Before starting a new release cycle:
1. Update `src/propflow/_version.py`
2. Update `pyproject.toml` version
3. Amend `CHANGELOG.md`
4. Run full CI-quality checks (`pytest`, `coverage`, lint, type checks)
5. Repeat the build and upload workflow above

---

## 10. Quick Reference Commands

```bash
# Clean workspace
rm -rf dist/ build/ src/*.egg-info/

# Build artifacts
uv build

# Validate distributions
uv run twine check dist/*

# Stage upload
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*

# Tag release
git tag -a vX.Y.Z -m "Release version vX.Y.Z"
git push origin vX.Y.Z
```

---

## 11. Troubleshooting Cheatsheet

- **"File already exists"**: The version is already on PyPI; bump `_version.py` and `pyproject.toml`, rebuild, and re-upload.
- **Invalid credentials**: Confirm `__token__` username and the correct API token; ensure it has not been revoked.
- **Long description not rendering**: Validate README, confirm `readme = {file = "README.md", content-type = "text/markdown"}` in `pyproject.toml`.
- **Dependencies missing**: Verify they are published on PyPI and pinned correctly in `pyproject.toml`.
- **Import errors after install**: Confirm package structure, `__init__.py` exports, and re-run `python -c "import propflow; print(propflow.__version__)"` in a fresh environment.

---

## 12. References

- Packaging User Guide: <https://packaging.python.org/>
- Twine Documentation: <https://twine.readthedocs.io/>
- PropFlow Release Guide (full detail): `RELEASE_GUIDE.md`
- PyPI Help: <https://pypi.org/help/>

---

**Good luck shipping PropFlow releases! 🚀**
