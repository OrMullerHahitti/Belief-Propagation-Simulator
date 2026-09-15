"""Build the self-contained explorer fragment from saved analysis data."""

from __future__ import annotations

import argparse
from pathlib import Path

from .run import OUTPUT


def build(inline_path: Path | None = None) -> Path:
    """Render literal markup with inline data; no network requests are needed."""
    template = Path(__file__).with_name("explorer.template.html").read_text()
    data = (OUTPUT / "inline_data.json").read_text()
    fragment = template.replace("__DATA__", data)
    if "__DATA__" in fragment or len(fragment.encode()) >= 1_000_000:
        raise ValueError("unresolved data or oversized explorer fragment")
    path = OUTPUT / "explorer.fragment.html"
    path.write_text(fragment)
    if inline_path is not None:
        inline_path.parent.mkdir(parents=True, exist_ok=True)
        inline_path.write_text(fragment)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inline-path", type=Path)
    print(build(parser.parse_args().inline_path))
