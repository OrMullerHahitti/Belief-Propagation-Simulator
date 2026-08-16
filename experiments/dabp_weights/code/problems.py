"""10-agent random problems for the DABP edge-weight analysis.

Scaled-down version of the AAAI ``random_sparse``/``random_dense`` recipe
(experiments/aaai/code/problems.py): Erdos-Renyi topology (force-connected by
FGBuilder), domain 10, integer costs U[100, 200), plus tiny random unary value
preferences (uniform [0, 1e-2)) for tie breaking. Density 0.3 keeps the
10-agent graph cyclic (~13 binary factors); at the AAAI sparse density 0.1 a
10-node graph would be stitched into a near-tree, where BP is exact and the
learned weights are uninteresting.
"""

from __future__ import annotations

import numpy as np

from propflow import FGBuilder
from propflow.bp.factor_graph import FactorGraph

NUM_AGENTS = 10
DOMAIN_SIZE = 10
DENSITY = 0.3
PREF_SCALE = 1e-2


def _with_tiebreak_prefs(fg: FactorGraph, rng: np.random.Generator) -> FactorGraph:
    """add tiny random unary value preferences for tie breaking."""
    unary = {v.name: rng.uniform(0.0, PREF_SCALE, size=v.domain) for v in fg.variables}
    return FGBuilder.build_with_unary_costs(fg, unary)


def build_random_10(seed: int) -> FactorGraph:
    """AAAI-style random problem at 10 agents; fully determined by ``seed``."""
    # cost tables are drawn from the legacy global rng inside FGBuilder
    np.random.seed(seed)
    fg = FGBuilder.build_random_graph(
        num_vars=NUM_AGENTS,
        domain_size=DOMAIN_SIZE,
        ct_factory="random_int",
        ct_params={"low": 100, "high": 200},
        density=DENSITY,
        seed=seed,
    )
    return _with_tiebreak_prefs(fg, np.random.default_rng(seed))
