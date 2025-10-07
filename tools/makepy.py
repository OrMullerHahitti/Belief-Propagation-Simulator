#!/usr/bin/env python3
"""Project scaffolding helper for `make makePy`.

This script mirrors the original Makefile templates without relying on shell
heredocs so the main Makefile stays portable across BSD/GNU make variants.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent


def write_file(path: Path, contents: str, *, strip: bool = True) -> None:
    """Write *contents* to *path*, ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dedent(contents)
    if strip:
        text = text.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: makepy.py <root> <pkg> <name> <year>")

    root = Path(sys.argv[1])
    pkg = sys.argv[2] or "package"
    project_name = sys.argv[3] or pkg
    year = sys.argv[4]

    (root / f"src/{pkg}").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / ".github/workflows").mkdir(parents=True, exist_ok=True)

    write_file(
        root / "pyproject.toml",
        f"""
        [build-system]
        requires = ["hatchling>=1.25"]
        build-backend = "hatchling.build"

        [project]
        name = "{pkg}"
        version = "0.1.0"
        description = "{project_name}"
        readme = "README.md"
        requires-python = ">=3.8"
        license = {{ text = "MIT" }}
        authors = [{{ name = "Your Name", email = "you@example.com" }}]
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
        ]
        dependencies = []

        [project.urls]
        Homepage = "https://example.com/{pkg}"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{pkg}"]

        [tool.pytest.ini_options]
        addopts = "-q"
        testpaths = ["tests"]
        """,
    )

    write_file(
        root / "README.md",
        f"""
        # {project_name}

        Basic Python package scaffold generated via `make makePy`.

        ## Quickstart

        ```bash
        make venv install
        make test
        make build
        ```
        """,
    )

    write_file(
        root / "LICENSE",
        f"""
        MIT License

        Copyright (c) {year} Your Name

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
        """,
    )

    write_file(
        root / ".gitignore",
        """
        .venv/
        __pycache__/
        *.pyc
        dist/
        *.egg-info/
        .pytest_cache/
        .coverage
        htmlcov/
        """,
    )

    write_file(
        root / ".editorconfig",
        """
        root = true

        [*]
        end_of_line = lf
        insert_final_newline = true
        charset = utf-8
        indent_style = space
        indent_size = 4
        trim_trailing_whitespace = true
        """,
    )

    write_file(
        root / "ruff.toml",
        """
        line-length = 100
        target-version = "py38"

        [lint]
        select = ["E", "F", "I", "UP", "B"]
        ignore = []
        """,
    )

    write_file(
        root / ".pre-commit-config.yaml",
        """
        repos:
          - repo: https://github.com/psf/black
            rev: 24.8.0
            hooks:
              - id: black
          - repo: https://github.com/astral-sh/ruff-pre-commit
            rev: v0.6.9
            hooks:
              - id: ruff
                args: ["--fix"]
        """,
    )

    write_file(
        root / f"src/{pkg}/__init__.py",
        "__all__ = [\"greet\"]\n",
        strip=False,
    )

    write_file(
        root / f"src/{pkg}/core.py",
        """
        def greet(name: str) -> str:
            """Return a friendly greeting."""
            return f"Hello, {name}!"
        """,
    )

    write_file(
        root / "tests/test_core.py",
        f"""
        from {pkg}.core import greet


        def test_greet() -> None:
            assert greet("Alice") == "Hello, Alice!"
        """,
    )

    write_file(
        root / ".github/workflows/ci.yml",
        """
        name: ci

        on: [push, pull_request]

        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with:
                  python-version: "3.11"
              - run: python -m pip install --upgrade pip
              - run: pip install build pytest ruff black
              - run: ruff check .
              - run: black --check .
              - run: pytest -q
              - run: python -m build
        """,
    )

    write_file(
        root / "Makefile",
        """
        .PHONY: help venv install uv-install fmt fmt-check lint fix type test cov cov-html build sdist wheel twine-check publish testpublish clean distclean freeze

        help: ## Show help
        	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS := ":.*## "}; {printf "\\033[36m%-15s\\033[0m %s\\n", $$1, $$2}'

        venv: ## Create virtual env
        	python -m venv .venv

        install: ## Install dev tools
        	. .venv/bin/activate; pip install -U pip build twine pytest ruff black mypy coverage pre-commit

        uv-install: ## Install with uv (optional)
        	uv pip install -U build twine pytest ruff black mypy coverage pre-commit

        fmt: ## Format code with black
        	. .venv/bin/activate; black src tests

        fmt-check: ## Check formatting
        	. .venv/bin/activate; black --check src tests

        lint: ## Lint with ruff
        	. .venv/bin/activate; ruff check .

        fix: ## Ruff auto-fix
        	. .venv/bin/activate; ruff check . --fix

        type: ## Type-check with mypy
        	. .venv/bin/activate; mypy src

        test: ## Run pytest
        	. .venv/bin/activate; pytest -q

        cov: ## Coverage text report
        	. .venv/bin/activate; coverage run -m pytest -q && coverage report -m

        cov-html: ## Coverage HTML report
        	. .venv/bin/activate; coverage run -m pytest -q && coverage html && python -c "import webbrowser; webbrowser.open('htmlcov/index.html')"

        build: ## Build wheel+sdist
        	. .venv/bin/activate; python -m build

        sdist: ## Build sdist only
        	. .venv/bin/activate; python -m build --sdist

        wheel: ## Build wheel only
        	. .venv/bin/activate; python -m build --wheel

        twine-check: ## Validate dist metadata
        	. .venv/bin/activate; twine check dist/*

        publish: ## Upload to PyPI (set PYPI_TOKEN env)
        	. .venv/bin/activate; TWINE_USERNAME=__token__ TWINE_PASSWORD=$$PYPI_TOKEN twine upload dist/*

        testpublish: ## Upload to TestPyPI (set TEST_PYPI_TOKEN env)
        	. .venv/bin/activate; TWINE_USERNAME=__token__ TWINE_PASSWORD=$$TEST_PYPI_TOKEN twine upload -r testpypi dist/*

        freeze: ## Export exact deps
        	. .venv/bin/activate; pip freeze > requirements.txt

        clean: ## Remove build and cache artifacts
        	rm -rf dist *.egg-info .pytest_cache .coverage htmlcov

        distclean: clean ## Also remove venv
        	rm -rf .venv
        """,
    )

    print(f"Scaffold written to {root}")


if __name__ == "__main__":
    main()
