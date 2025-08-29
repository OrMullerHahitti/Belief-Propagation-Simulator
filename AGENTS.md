# Repository Guidelines

## Project Structure & Module Organization
- `src/propflow`: core library (BP engines, factor graph, policies, configs, CLI).
- `src/analyzer`: analysis and validation helpers for simulations.
- `tests`: pytest suite covering engines, graph, configs, and utilities.
- `examples`: runnable samples and reference configs.
- `configs` and `src/propflow/configs`: predefined config files and factories.
- `docs`: additional notes and visuals; `README.md` for quick start.

## Build, Test, and Development Commands
```bash
# Setup (editable install + dev tools)
python -m venv .venv && source .venv/bin/activate  # or Scripts\\activate on Windows
pip install -e .[dev] && pre-commit install

# Run tests and coverage
pytest -q
pytest --cov=src --cov-report=term-missing

# Lint, type-check, format
flake8
mypy src
black .

# CLI / examples
bp-sim --help
python examples/minsum_basic.py  # if example uses a main
```

## Coding Style & Naming Conventions
- Python 3.10+ with type hints; format with Black; `flake8` max line length 120.
- Naming: `snake_case` for modules/functions/vars, `PascalCase` for classes.
- Keep modules within `src/propflow` or `src/analyzer`; avoid circular imports.
- Prefer pure, testable functions; document public APIs with docstrings.

## Testing Guidelines
- Framework: `pytest`. Place tests in `tests/` named `test_*.py`; functions `test_*`.
- Add tests for new behavior and edge cases; update fixtures in `tests/conftest.py` when needed.
- Run `pytest -q` locally; ensure coverage does not regress (`pytest --cov=src`).

## Commit & Pull Request Guidelines
- Commit style: Conventional Commits seen in history (e.g., `feat: ...`, `fix: ...`, `refactor: ...`, `test: ...`, `chore: ...`).
- PRs: clear description, rationale, and scope; link issues; include reproduction steps (config/seed), logs or small screenshots for plots if applicable.
- PR checklist: tests added/updated, `pre-commit run -a` clean, no files in `__pycache__`, large artifacts excluded.

## Security & Configuration Tips
- Do not commit secrets; `.env` is ignored. Use deterministic seeds in configs for reproducibility.
- Keep long runs and generated outputs out of Git; prefer saving to paths under `results/` or temp directories noted in configs.
