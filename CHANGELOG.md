# Changelog

All notable changes to PropFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive Jupyter notebook tutorial for analyzer module
- PyPI readiness improvements
- Enhanced README with badges and better structure
- User guide with top-down conceptual approach

### Changed
- Updated dependency constraints to be more flexible for library distribution
- Modernized license configuration in pyproject.toml
- Improved package metadata with additional classifiers

### Fixed
- Removed pytest from runtime dependencies (moved to dev-only)
- Fixed MANIFEST.in to exclude build artifacts
- Updated gitignore for Jupyter checkpoints

## [0.1.0] - 2025-10-02

### Added
- Core belief propagation engine with synchronous message passing
- Multiple BP variants: Min-Sum, Max-Sum, Sum-Product, Max-Product
- Policy system: damping, splitting, cost reduction, message pruning
- Factor graph construction utilities (FGBuilder)
- Parallel simulator for comparative experiments
- Snapshot recording and visualization (analyzer module)
- Local search algorithms: DSA, MGM, K-Opt MGM
- Comprehensive test suite
- Deployment handbook with 7 focused guides
- Examples directory with working demonstrations
- CLI entry point (bp-sim)

### Infrastructure
- Modern src/ layout
- pyproject.toml configuration
- MIT License
- Python 3.10+ support
- Continuous testing with pytest

[Unreleased]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/releases/tag/v0.1.0
