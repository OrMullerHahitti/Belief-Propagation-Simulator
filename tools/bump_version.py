#!/usr/bin/env python3
"""Version bumper invoked from the Makefile."""

from __future__ import annotations

import re
import sys
from pathlib import Path

VALID_PARTS = {"patch", "minor", "major"}


def bump_version_in_pyproject(pyproject: Path, part: str) -> str:
    """Bump version in pyproject.toml and return the new version."""
    text = pyproject.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(?m)^(?P<prefix>\s*version\s*=\s*")(?P<version>[^"]+)(?P<suffix>")'
    )
    match = pattern.search(text)
    if not match:
        raise SystemExit("version not found in pyproject.toml")

    major, minor, patch = map(int, match["version"].split("."))
    if part == "patch":
        patch += 1
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "major":
        major, minor, patch = major + 1, 0, 0
    else:
        raise SystemExit(f"unknown part '{part}' (expected patch/minor/major)")

    new_version = f"{major}.{minor}.{patch}"
    updated = pattern.sub(f"\\g<prefix>{new_version}\\g<suffix>", text, count=1)
    pyproject.write_text(updated, encoding="utf-8")
    return new_version


def update_version_py(version_file: Path, new_version: str) -> None:
    """Update version in _version.py file."""
    if not version_file.exists():
        print(f"Warning: {version_file} does not exist, skipping")
        return

    text = version_file.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(?m)^(?P<prefix>__version__\s*=\s*")(?P<version>[^"]+)(?P<suffix>")'
    )
    match = pattern.search(text)
    if not match:
        print(f"Warning: __version__ not found in {version_file}, skipping")
        return

    updated = pattern.sub(f"\\g<prefix>{new_version}\\g<suffix>", text, count=1)
    version_file.write_text(updated, encoding="utf-8")
    print(f"Updated {version_file}")


def update_docs_conf(conf_file: Path, new_version: str) -> None:
    """Update version in docs/conf.py file."""
    if not conf_file.exists():
        print(f"Warning: {conf_file} does not exist, skipping")
        return

    text = conf_file.read_text(encoding="utf-8")

    # Update version
    version_pattern = re.compile(
        r"(?m)^(?P<prefix>version\s*=\s*')(?P<version>[^']+)(?P<suffix>')"
    )
    version_match = version_pattern.search(text)
    if version_match:
        text = version_pattern.sub(
            f"\\g<prefix>{new_version}\\g<suffix>", text, count=1
        )
    else:
        print(f"Warning: version not found in {conf_file}")

    # Update release
    release_pattern = re.compile(
        r"(?m)^(?P<prefix>release\s*=\s*')(?P<version>[^']+)(?P<suffix>')"
    )
    release_match = release_pattern.search(text)
    if release_match:
        text = release_pattern.sub(
            f"\\g<prefix>{new_version}\\g<suffix>", text, count=1
        )
    else:
        print(f"Warning: release not found in {conf_file}")

    if version_match or release_match:
        conf_file.write_text(text, encoding="utf-8")
        print(f"Updated {conf_file}")


def bump_version(root: Path, part: str) -> str:
    """Bump version in all relevant files."""
    # First, bump version in pyproject.toml
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        raise SystemExit(f"{pyproject} does not exist")

    new_version = bump_version_in_pyproject(pyproject, part)
    print(f"Updated {pyproject}")

    # Update other version files
    version_py = root / "src" / "propflow" / "_version.py"
    update_version_py(version_py, new_version)

    docs_conf = root / "docs" / "conf.py"
    update_docs_conf(docs_conf, new_version)

    return new_version


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: bump_version.py <root> <patch|minor|major>")

    root = Path(sys.argv[1])
    part = sys.argv[2]

    if part not in VALID_PARTS:
        raise SystemExit(
            f"unknown part '{part}' (expected one of {sorted(VALID_PARTS)})"
        )

    new_version = bump_version(root, part)
    print(f"Version bumped to {new_version}")


if __name__ == "__main__":
    main()
