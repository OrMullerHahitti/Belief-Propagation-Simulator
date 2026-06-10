"""Shared plotting helpers (same conventions as experiments/aij)."""


def remove_frame(ax) -> None:
    """Remove top and right spines from an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
