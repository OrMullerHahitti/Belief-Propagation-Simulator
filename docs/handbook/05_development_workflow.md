# Development Workflow

## 1. Repository Hygiene
- Work from feature branches (`feat/`, `fix/`, etc.) and keep `main` clean.
- Follow Conventional Commit messages (`feat:`, `fix:`, `chore:`, `docs:`, etc.).
- Avoid committing generated artefacts (plots, large logs, `__pycache__`); ensure `.gitignore` is up to date.

## 2. Tooling
| Tool | Command | Purpose |
| --- | --- | --- |
| `pre-commit` | `uv run pre-commit run -a` | Apply linting/formatting hooks across the repo. |
| `black` | `uv run black .` | Enforce consistent formatting (line length 120). |
| `flake8` | `uv run flake8` | Static lint checks. |
| `mypy` | `uv run mypy src` | Type checking. |
| `pytest` | `uv run pytest -q` | Run unit tests. |
| `pytest --cov` | `uv run pytest --cov=src --cov-report=term-missing` | Code coverage analysis. |

## 3. Coding Guidelines
- Python 3.10+ features are encouraged (pattern matching, dataclasses).
- Use type hints throughout; prefer dataclasses for structured data.
- Keep modules within `src/propflow` or `src/analyzer` to avoid circular imports.
- Document public interfaces with docstrings and include inline comments only when necessary for clarity.
- Maintain ASCII encoding unless the file already contains Unicode and there is a compelling reason.

## 4. Testing Strategy
- Place tests under `tests/` with names `test_*.py`.
- Mock external dependencies carefully; most BP logic is deterministic and can be tested directly.
- For stochastic components (random graph generation), seed `numpy` and `random` to guarantee reproducibility.
- Extend fixtures in `tests/conftest.py` when you need reusable graph setups.

## 5. Adding New Features
1. Discuss or document the approach (issues, TODO comments).
2. Implement feature in `src/propflow/...` or `src/analyzer/...`.
3. Add unit tests or update existing ones.
4. Update documentation (this handbook, README, or module docstrings) as needed.
5. Run the quality gates (formatters, linters, tests).
6. Open a pull request summarising the change, rationale, and testing evidence.

## 6. Release Process
1. Update version identifiers (`src/propflow/_version.py`, `pyproject.toml`).
2. Update CHANGELOG or release notes (if maintained separately).
3. Run the full test suite and static checks.
4. Build distribution artefacts: `uv build`.
5. Upload to distribution channel (e.g., PyPI Test, internal index) using `twine`.
6. Tag the release in git (`git tag vX.Y.Z && git push --tags`).

## 7. Documentation Maintenance
- This handbook should live alongside code changes; add or edit relevant sections when capacities evolve (e.g., new CLI commands, policies).
- For API-specific notes, embed docstrings or dedicated markdown under `docs/`.
- Keep example notebooks (under `notebooks/`) in sync by re-running them prior to publication.

## 8. Collaboration
- Review peers’ code focusing on correctness, maintainability, and performance.
- Highlight potential regressions or missing tests during reviews.
- Use draft PRs for work-in-progress features; convert to ready-for-review when tests pass and scope is locked.

By following this workflow the project remains stable, testable, and ready for deployment across environments.
