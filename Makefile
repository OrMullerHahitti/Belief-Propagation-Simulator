# PropFlow developer tasks
SHELL := /bin/bash
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
UV ?= uv

.PHONY: help venv install fmt fmt-check lint type test cov build check-dist publish-test publish release clean distclean docs-example

help: ; @grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*## "} {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' ## Show this help

venv: ; python -m venv $(VENV) ## Create virtual environment

install: venv ; $(PIP) install -U pip && $(PIP) install -e '.[dev]' ## Install project in dev mode

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

docs-example: ; $(PYTHON) examples/minsum_basic.py ## Run example to smoke-test docs flow

clean: ; rm -rf dist build src/*.egg-info __pycache__ .pytest_cache .mypy_cache htmlcov .coverage ## Remove build/test artifacts

distclean: clean ; rm -rf $(VENV) ## Remove artifacts and virtualenv
