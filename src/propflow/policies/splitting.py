"""A Policy for Splitting Factors in a Factor Graph.

This module provides a function that implements the factor splitting policy.
This technique modifies the structure of the factor graph by replacing each
factor with two new "cloned" factors. The original cost table is distributed
between these two clones. This can be useful for altering the message-passing
dynamics and can sometimes help with convergence or finding better solutions.
"""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Dict, List, Sequence

import networkx as nx

from ..bp.factor_graph import FactorGraph
from ..configs.global_config_mapping import PolicyDefaults
from ..core.agents import FactorAgent


def _split_factors(
    fg: FactorGraph, factors: List[FactorAgent], p: float
) -> Dict[str, List[FactorAgent]]:
    """Replaces each factor with two clones having cost tables p*C and (1-p)*C."""
    G: nx.Graph = fg.G
    split_mapping: Dict[str, List[FactorAgent]] = {}

    for f in list(factors):
        cost1 = p * f.cost_table  # type: ignore
        cost2 = (1.0 - p) * f.cost_table  # type: ignore

        f1 = f.create_from_cost_table(cost_table=cost1, name=f"{f.name}'")
        f2 = f.create_from_cost_table(cost_table=cost2, name=f"{f.name}''")

        f1.connection_number = deepcopy(f.connection_number)
        f2.connection_number = deepcopy(f.connection_number)

        for v, edge_data in G[f].items():
            G.add_edge(f1, v, **edge_data)
            G.add_edge(f2, v, **edge_data)

        fg.factors.append(f1)
        fg.factors.append(f2)

        G.remove_node(f)
        fg.factors.remove(f)
        split_mapping[f.name] = [f1, f2]

    return split_mapping


def _select_factors_for_split(
    fg: FactorGraph,
    *,
    factor_names: Sequence[str] | None = None,
    split_fraction: float | None = None,
    seed: int | None = None,
) -> List[FactorAgent]:
    candidates = list(fg.factors)
    if factor_names is not None:
        by_name = {factor.name: factor for factor in candidates}
        missing = [name for name in factor_names if name not in by_name]
        if missing:
            raise ValueError(f"Unknown factor(s) requested for splitting: {missing}")
        candidates = [by_name[name] for name in factor_names]

    candidates = sorted(candidates, key=lambda factor: factor.name)
    if split_fraction is None:
        return candidates

    if not 0.0 < split_fraction <= 1.0:
        raise ValueError("split_fraction must be in the range (0, 1].")

    if not candidates:
        return []

    count = max(1, math.ceil(len(candidates) * split_fraction))
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, count), key=lambda factor: factor.name)


def split_factors(
    fg: FactorGraph,
    p: float | None = None,
    *,
    factor_names: Sequence[str] | None = None,
    split_fraction: float | None = None,
    seed: int | None = None,
) -> Dict[str, List[FactorAgent]]:
    """Split selected factors in-place and return original-to-clone mapping.

    Args:
        fg: The factor graph to modify.
        p: Split proportion allocated to the first clone.
        factor_names: Optional factor names to restrict the split target set.
            When omitted, all factors are candidates.
        split_fraction: Optional fraction of the candidate factors to split.
            Selection is deterministic for a given seed.
        seed: Seed used when ``split_fraction`` is provided.

    Returns:
        A mapping from original factor name to the two replacement clone factors.
    """
    if p is None:
        p = PolicyDefaults().split_factor
    assert 0.0 < p < 1.0, "p must be in (0,1)"  # type: ignore

    selected = _select_factors_for_split(
        fg,
        factor_names=factor_names,
        split_fraction=split_fraction,
        seed=seed,
    )
    return _split_factors(fg, selected, p)


def split_all_factors(
    fg: FactorGraph,
    p: float = None,  # type: ignore
) -> Dict[str, List[FactorAgent]]:
    """Performs an in-place replacement of every factor with two cloned factors.

    Each factor `f` with cost table `C` is replaced by two new factors, `f'`
    and `f''`, with cost tables `p*C` and `(1-p)*C`, respectively. The new
    factors inherit all the connections of the original factor.

    This function directly modifies the provided `FactorGraph` object, including
    its underlying `networkx.Graph` and its list of factors.

    Args:
        fg: The `FactorGraph` to modify.
        p: The splitting proportion, which must be between 0 and 1. This
           determines how the original cost is distributed between the two
           new factors. If None, the default from `POLICY_DEFAULTS` is used.

    Raises:
        AssertionError: If `p` is not in the range (0, 1).
    """
    if p is None:
        p = PolicyDefaults().split_factor

    assert 0.0 < p < 1.0, "p must be in (0,1)"
    return _split_factors(fg, fg.factors, p)


def split_specific_factors(
    fg: FactorGraph, factors: List[FactorAgent], p: float | None = None
) -> Dict[str, List[FactorAgent]]:
    """Split specific factors in the factor graph.

    Args:
        fg (FactorGraph): The factor graph to modify.
        factors (List[FactorAgent]): The factors to split.
        p (float | None, optional): The splitting proportion. Defaults to None.
    """
    if p is None:
        p = PolicyDefaults().split_factor

    assert 0.0 < p < 1.0, "p must be in (0,1)"  # type: ignore
    return _split_factors(fg, factors, p)
