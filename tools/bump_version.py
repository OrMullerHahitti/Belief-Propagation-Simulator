#!/usr/bin/env python3
"""Version bumper invoked from the Makefile."""

from __future__ import annotations

import re
import sys
from pathlib import Path

VALID_PARTS = {"patch", "minor", "major"}


def bump_version(pyproject: Path, part: str) -> str:
    text = pyproject.read_text(encoding="utf-8")
    pattern = re.compile(r'(?m)^(?P<prefix>\s*version\s*=\s*")(?P<version>[^"]+)(?P<suffix>")')
    match = pattern.search(text)
    if not match:
        raise SystemExit("version not found in pyproject.toml")

    major, minor, patch = map(int, match.group("version").split("."))
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


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: bump_version.py <root> <patch|minor|major>")

    root = Path(sys.argv[1])
    part = sys.argv[2]

    if part not in VALID_PARTS:
        raise SystemExit(f"unknown part '{part}' (expected one of {sorted(VALID_PARTS)})")

    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        raise SystemExit(f"{pyproject} does not exist")

    new_version = bump_version(pyproject, part)
    print(f"Version -> {new_version}")


if __name__ == "__main__":
    main()
