# Repository Guidelines

## Project Structure & Module Organization
PropFlow’s core runtime sits in `src/propflow`, covering belief-propagation engines, factor graph primitives, policy logic, configs, and the CLI entrypoint. Validation helpers live in `src/analyzer` for metrics and sanity checks. Pytest suites are grouped under `tests/`, while runnable demonstrations and reference setups live in `examples/`. Prebuilt configs reside in `configs/` and `src/propflow/configs/`. Documentation, diagrams, and quick-start notes live in `docs/` and `README.md`. Use `notebooks/` for exploratory analyses kept outside packaged modules.

## Build, Test, and Development Commands
Run commands from the repository root after activating the virtual environment.
- `python -m venv .venv && source .venv/bin/activate`: Bootstrap a Unix venv (`Scripts\activate` on Windows).
- `pip install -e .[dev] && pre-commit install`: Install runtime deps plus Git hooks.
- `pytest -q`: Run the fast regression suite.
- `pytest --cov=src --cov-report=term-missing`: Check coverage gaps.
- `flake8`, `mypy src`, `black .`: Lint, type-check, and auto-format; hooks mirror this trio.
- `bp-sim --help` / `python examples/minsum_basic.py`: Smoke-test the CLI and sample config.

## Coding Style & Naming Conventions
Target Python 3.10+, keep modules ASCII, and let Black format code (complements `flake8`’s 120-character enforcement). Use `snake_case` for functions, modules, and variables; reserve `PascalCase` for classes and `UPPER_CASE` for constants. Annotate public APIs with type hints and docstrings, and isolate reusable helpers inside `src/propflow` or `src/analyzer` to avoid import cycles.

## Testing Guidelines
All automated tests use `pytest`. Add files as `tests/test_feature.py` and functions `def test_behavior()`. Cover happy paths, edge cases, and failure signals; reuse deterministic seeds from configs when randomness appears. Run `pytest -q` and `pytest --cov=src --cov-report=term-missing` before opening a PR to prevent coverage regressions.

## Commit & Pull Request Guidelines
Commits follow Conventional Commit prefixes (`feat`, `fix`, `refactor`, `test`, `chore`) and focus on one logical change. Reference issues or key configs in the subject when helpful. PRs need a clear problem statement, reproduction steps or config/seed, and refreshed logs or screenshots for analyzer outputs. Confirm `pre-commit run -a` and the full test suite pass, and keep artifacts and `__pycache__` out of Git.

## Security & Configuration Tips
Never commit secrets or raw datasets; `.env` stays untracked. Store long-running experiment outputs under `results/` or temp paths referenced in configs. Stick to deterministic seeds so simulations remain reproducible across contributors.
