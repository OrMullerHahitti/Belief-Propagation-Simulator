# PropFlow PyPI Readiness Report

## Executive Summary

**Status**: ⚠️ **Nearly Ready** - Minor issues to address before publication
**Build Status**: ✅ Builds successfully (wheel + sdist)
**Package Validation**: ✅ Passes `twine check`
**Version**: 0.1.0

---

## ✅ What's Working Well

### Package Structure
- ✅ Proper `src/` layout following modern best practices
- ✅ Clear separation: `propflow` (core) + `analyzer` (utilities)
- ✅ Comprehensive test suite in `tests/`
- ✅ Examples directory with working code and Jupyter notebook

### Metadata
- ✅ Appropriate classifiers for scientific/research software
- ✅ Keywords align with target audience
- ✅ Python version requirement (>=3.10) is appropriate
- ✅ MIT License included
- ✅ README.md is comprehensive and well-formatted
- ✅ Entry point defined (`bp-sim` CLI command)

### Build System
- ✅ Uses modern pyproject.toml configuration
- ✅ setuptools backend configured correctly
- ✅ Package builds cleanly (wheel + source distribution)
- ✅ Passes twine validation

### Documentation
- ✅ Excellent handbook in `docs/handbook/` (7 comprehensive guides)
- ✅ README with examples, architecture, and quick start
- ✅ CLAUDE.MD for development guidance
- ✅ Analyzer README explaining snapshot recording

---

## ⚠️ Issues to Address Before PyPI Release

### 1. **CRITICAL: License Configuration (Deprecation Warning)**
**Severity**: HIGH (will break builds by Feb 2026)

**Current Issue**:
```toml
license = {text = "MIT"}
```

**Required Fix**:
```toml
license = "MIT"
# OR use SPDX identifier:
license = "MIT"
license-files = ["LICENSE"]
```

**Action**: Update `pyproject.toml` line 17

---

### 2. **CRITICAL: Remove License Classifier**
**Severity**: MEDIUM

Remove this classifier (deprecated with SPDX):
```toml
"License :: OSI Approved :: MIT License",  # REMOVE THIS
```

---

### 3. **CRITICAL: pytest in Runtime Dependencies**
**Severity**: HIGH

**Issue**: `pytest>=8.4.2` is in `dependencies` (line 50) - should be dev-only

**Fix**: Move to `[project.optional-dependencies].dev` section (already there)

---

### 4. **MANIFEST.in Warnings**
**Severity**: LOW

Missing files referenced in MANIFEST.in:
- `CHANGELOG.md` - Should be created
- `CONTRIBUTING.md` - Should be created or removed from MANIFEST
- `requirements.txt` - Can be removed (using pyproject.toml)
- `configs/*.py` and `configs/*.md` - Paths don't match project structure

**Fix**: Update MANIFEST.in to match actual project structure

---

### 5. **Jupyter Checkpoints in Distribution**
**Severity**: LOW

`.ipynb_checkpoints` included in build artifacts

**Fix**: Add to .gitignore and MANIFEST.in exclusions:
```
global-exclude .ipynb_checkpoints
global-exclude *-checkpoint.ipynb
```

---

### 6. **Version Management**
**Severity**: LOW

Version hardcoded in two places:
- `pyproject.toml` (line 13)
- `src/propflow/_version.py` (line 1)

**Recommendation**: Use dynamic versioning OR single-source-of-truth

---

### 7. **Dependencies Pinning Too Strict**
**Severity**: MEDIUM

Using `~=` pins may be too restrictive for library distribution:

```toml
# Current (very strict):
"numpy~=2.2.1"  # Only allows 2.2.x

# Recommended for libraries:
"numpy>=2.0.0,<3.0.0"  # More flexible
```

**Rationale**: Libraries should be flexible; applications should pin strictly

---

### 8. **Missing Package Classifiers**
**Severity**: LOW

Consider adding:
```toml
"Programming Language :: Python :: 3.13",
"Operating System :: OS Independent",
"Topic :: Scientific/Engineering :: Mathematics",
"Framework :: Matplotlib",
```

---

### 9. **Missing Long Description Content Type**
**Severity**: LOW

Add to `pyproject.toml`:
```toml
readme = {file = "README.md", content-type = "text/markdown"}
```

---

### 10. **Comprehensive User Guide**
**Severity**: MEDIUM

**Current State**: Good deployment handbook but missing user-focused quick start guide

**Recommendation**: Create `GETTING_STARTED.md` or `USER_GUIDE.md` that bridges:
- Installation → First factor graph → First simulation → Understanding results
- Top-down conceptual approach
- Less operations-focused than current handbook

---

## 📊 Dependency Analysis

### Runtime Dependencies (9 packages)
| Package | Version Constraint | Concern |
|---------|-------------------|---------|
| networkx | ~=3.4.2 | Too strict |
| numpy | ~=2.2.1 | Too strict |
| matplotlib | ~=3.10.0 | Too strict |
| scipy | ~=1.15.2 | Too strict |
| dash | ~=2.18.2 | OK for specialized dep |
| dash_cytoscape | ~=1.0.2 | OK |
| colorlog | ~=6.9.0 | OK |
| pandas | ~=2.2.3 | Too strict |
| seaborn | ~=0.13.2 | OK |
| psutil | ~=7.0.0 | Too strict |
| **pytest** | >=8.4.2 | **MOVE TO DEV** |

**Recommendation**: Loosen core scientific packages to allow broader compatibility

---

## 📝 Action Plan for PyPI Release

### Phase 1: Critical Fixes (MUST DO)
1. ✅ Fix license configuration in pyproject.toml
2. ✅ Remove license classifier
3. ✅ Move pytest from runtime to dev dependencies
4. ✅ Create CHANGELOG.md
5. ✅ Update .gitignore for Jupyter checkpoints

### Phase 2: Recommended Improvements (SHOULD DO)
6. ✅ Loosen dependency version constraints
7. ✅ Update MANIFEST.in
8. ✅ Add missing classifiers
9. ✅ Create comprehensive user guide (top-down, conceptual)

### Phase 3: Nice to Have (COULD DO)
10. ⚪ Setup automatic versioning
11. ⚪ Add GitHub Actions for CI/CD
12. ⚪ Create readthedocs documentation
13. ⚪ Add badges to README

---

## 🎯 Recommendation

**Proceed with PyPI release after addressing Phase 1 issues.**

The package is well-structured and professionally maintained. The critical issues are all straightforward to fix and primarily involve modernizing the package metadata to comply with current Python packaging standards.

### Pre-Release Checklist
```bash
# 1. Fix pyproject.toml issues
# 2. Create missing files
# 3. Rebuild and test
uv build
uv run twine check dist/*

# 4. Test install from wheel
pip install dist/propflow-0.1.0-py3-none-any.whl

# 5. Run test suite
pytest

# 6. Upload to TestPyPI first
twine upload --repository testpypi dist/*

# 7. Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ propflow

# 8. If all tests pass, upload to PyPI
twine upload dist/*
```

---

## 📚 Documentation Quality

**Existing Documentation**: Excellent

- ✅ Comprehensive deployment handbook
- ✅ Well-structured README with examples
- ✅ Code-level documentation (CLAUDE.md)
- ✅ Analyzer-specific README

**Gap**: User-focused getting started guide with conceptual foundation

---

**Report Generated**: 2025-10-02
**Reviewer**: Automated PyPI Readiness Assessment
