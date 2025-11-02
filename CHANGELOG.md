# Changelog

All notable changes to PropFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Optional **SoftMinTorchComputator** (PyTorch-based soft-min for factor→variable messages).
- Docs & example for Torch integration.
- Comprehensive Jupyter notebook tutorial for snapshot analysis module
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

## [1.1.1] - 2025-10-29

### Added
- Enhanced `SnapshotVisualizer` with BCT visualisation, improved factor-cost plotting, and an engine snapshot saver hook.
- Shipped `.pyi` stubs so IDEs surface signatures for snapshot utilities and corrected hover information.
- Expanded automated tests covering factor graph algorithms and the refreshed snapshot visualisations.

### Changed
- Overhauled `CTFactories`, simplifying registration to direct function references and cleaning metadata/docstrings.
- Removed redundant `bp_policies` scaffolding and streamlined global configuration descriptions.
- Migrated examples and documentation to the unified `propflow.snapshots` pipeline, including new snapshot guides.
- Updated CI workflows to target Python 3.12.

### Fixed
- Corrected static method annotations and `TypeAlias` usage for broader Python compatibility.
- Addressed notebook lint issues and tightened global configuration handling.

## [1.0.1] - 2025-10-26

### Added
- Introduced DSA, MGM, and MGM2 local search engines.
- Added winner serialization and consolidated snapshot analysis directly under `propflow.snapshots`.
- Supplied automation helpers for coverage reporting and version bumping.

### Changed
- Refactored snapshot management with non-serializable fallbacks and improved `BPEngine` default handling.
- Updated notebooks, documentation, and README content to match the new snapshot API.
- Tweaked GitHub workflows and aligned search tests with the refreshed agent setup.

### Fixed
- Ensured generated random factor graphs remain connected.
- Added safe fallbacks for snapshot builder/manager serialization edge cases.
- Hardened version detection and refreshed type hints/defaults in `BPEngine` and `SnapshotsConfig`.

## [0.12] - 2025-10-08

### Added
- Delivered the first `SnapshotAnalyzer` release with math utilities, JSON parser, advanced metrics, and a reporting CLI.
- Added a gradient-based trainable BP module for cost-table optimisation.
- Authored an analyzer reporting notebook, handbook chapter, and Makefile helpers.

### Changed
- Rewrote top-level documentation to follow a top-down flow and linked to the new analyzer tooling.
- Refreshed dependencies and README pointers after the analyzer landing.

## [0.1.1] - 2025-10-07

### Changed
- Prepared the first follow-up release with documentation updates to the changelog (no code changes).

## [0.1.0] - 2025-10-02

### Added
- Core belief propagation engine with synchronous message passing
- Multiple BP variants: Min-Sum, Max-Sum, Sum-Product, Max-Product
- Policy system: damping, splitting, cost reduction, message pruning
- Factor graph construction utilities (FGBuilder)
- Parallel simulator for comparative experiments
- Snapshot recording and visualization (snapshot module)
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

[Unreleased]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/compare/v1.0.1...v1.1.1
[1.0.1]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/compare/v0.12...v1.0.1
[0.12]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/compare/v0.1.1...v0.12
[0.1.1]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/tree/v0.1.1
[0.1.0]: https://github.com/OrMullerHahitti/Belief-Propagation-Simulator/releases/tag/v0.1.0
