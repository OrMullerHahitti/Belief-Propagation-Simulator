"""Shared plotting helpers for experiment scripts."""

from __future__ import annotations


def remove_frame(ax) -> None:
    """Remove top and right spines from an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
