# PropFlow developer tasks
SHELL := /bin/bash
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
UV ?= uv

.PHONY: help venv install sync sync-dev precommit ci fmt fmt-check lint type test cov build check-dist publish-test publish release clean distclean docs-example start-python notebook repl bump bump-patch bump-minor bump-major

help: ; @grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*## "} {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' ## Show this help

venv: ; python -m venv $(VENV) ## Create virtual environment

install: venv ; $(PIP) install -U pip && $(PIP) install -e '.[dev]' ## Install project in dev mode

# Added helper to align the uv-managed environment with declared runtime deps.
sync: ; $(UV) sync ## Sync runtime dependencies into the uv environment

# Added helper to sync development extras (pytest, linters, notebooks, etc.).
sync-dev: ; $(UV) sync --extra dev ## Sync runtime + dev dependencies via uv

# Added shortcut to run the full pre-commit suite locally before pushing.
precommit: ; $(UV) run pre-commit run -a ## Execute all pre-commit hooks across the repo

# Added composite check mirroring common CI (format check, lint, type-check, tests).
ci: fmt-check lint type test ## Run the main quality gates locally

fmt: ; $(PYTHON) -m black src tests ## Format code with black

fmt-check: ; $(PYTHON) -m black --check src tests ## Check formatting

lint: ; $(PYTHON) -m flake8 ## Lint with flake8

type: ; $(PYTHON) -m mypy src ## Type-check with mypy

test: ; $(PYTHON) -m pytest -q ## Run pytest suite

cov: ; $(PYTHON) -m pytest --cov=src --cov-report=term-missing ## Run coverage-enabled tests

build: clean ; $(UV) build ## Build wheel and sdist via uv

check-dist: build ; $(UV) run twine check dist/* ## Validate built artifacts

publish-test: check-dist ; $(UV) run twine upload --repository testpypi dist/* ## Upload to TestPyPI

publish: check-dist ; $(UV) run twine upload dist/* ## Upload to PyPI

release: check-dist publish ## Convenience alias for PyPI release

# Version bumping targets
bump: bump-patch ## Bump patch version (default)

bump-patch: ## Bump patch version (e.g., 0.1.2 -> 0.1.3)
	$(PYTHON) tools/bump_version.py . patch

bump-minor: ## Bump minor version (e.g., 0.1.2 -> 0.2.0)
	$(PYTHON) tools/bump_version.py . minor

bump-major: ## Bump major version (e.g., 0.1.2 -> 1.0.0)
	$(PYTHON) tools/bump_version.py . major

docs-example: ; $(PYTHON) examples/minsum_basic.py ## Run example to smoke-test docs flow

clean: ; rm -rf dist build src/*.egg-info __pycache__ .pytest_cache .mypy_cache htmlcov .coverage ## Remove build/test artifacts

distclean: clean ; rm -rf $(VENV) ## Remove artifacts and virtualenv



start-python: ## Create (if needed) .venv via uv and drop into an activated shell
	@test -d .venv || $(UV) venv --seed
	@echo "Launching shell with .venv activated (exit to return)."
	@$(SHELL) -lc "source .venv/bin/activate && exec $$SHELL"

notebook: ## Launch Jupyter Lab inside the uv-managed environment
	$(UV) run jupyter lab

repl: ## Open a Python REPL with project dependencies via uv
	$(UV) run python
