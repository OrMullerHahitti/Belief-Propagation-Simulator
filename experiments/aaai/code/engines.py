"""Engine variants and helpers for the AAAI experiments.

Everything here composes existing propflow engines/policies; nothing in
src/propflow is modified.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from propflow.bp.engine_base import BPEngine
from propflow.bp.engines import DampingEngine, MidRunSplitEngine
from propflow.bp.factor_graph import FactorGraph
from propflow.policies.cost_reduction import discount_attentive
from propflow.policies.damping import damp


class CostOnlySnapshot:
    """minimal snapshot holding only the per-step global cost."""

    def __init__(self, step: int) -> None:
        self.step = int(step)
        self.global_cost: float | None = None
        self.metadata: dict[str, Any] = {}


class CostOnlySnapshotManager:
    """capture only per-step global cost to keep long runs small and fast."""

    def capture_step(self, step_index: int, step: Any, engine: Any) -> CostOnlySnapshot:
        del step, engine
        return CostOnlySnapshot(step_index)


def split_all_factors_random_range(
    fg: FactorGraph, low: float, high: float, rng: np.random.Generator
) -> None:
    """split every factor f with table C into clones U*C and (1-U)*C, where U is
    drawn elementwise from uniform[low, high).

    this is the "random SCFG" version of Cohen, Galiki & Zivan (2020), Section
    6.2: for each entry containing cost c, the first clone's entry is selected
    randomly between low*c and high*c (their best-performing 0.4-0.6 variant).
    mechanics mirror propflow.policies.splitting._split_factors.
    """
    G = fg.G
    for f in list(fg.factors):
        u = rng.uniform(low, high, size=f.cost_table.shape)
        ct1 = u * f.cost_table
        ct2 = f.cost_table - ct1

        f1 = f.create_from_cost_table(cost_table=ct1, name=f"{f.name}'")
        f2 = f.create_from_cost_table(cost_table=ct2, name=f"{f.name}''")
        f1.connection_number = deepcopy(f.connection_number)
        f2.connection_number = deepcopy(f.connection_number)

        for v, edge_data in G[f].items():
            G.add_edge(f1, v, **edge_data)
            G.add_edge(f2, v, **edge_data)

        fg.factors.append(f1)
        fg.factors.append(f2)
        G.remove_node(f)
        fg.factors.remove(f)


class DampingRandomSplitEngine(DampingEngine):
    """DMS on a random SCFG: per-entry split ratio drawn from [split_low, split_high)."""

    def __init__(
        self,
        *args,
        split_low: float = 0.4,
        split_high: float = 0.6,
        split_seed: int = 0,
        **kwargs,
    ) -> None:
        self.split_low = float(split_low)
        self.split_high = float(split_high)
        self.split_seed = int(split_seed)
        super().__init__(*args, **kwargs)
        self._set_name({"split": f"{self.split_low}-{self.split_high}"})

    def post_init(self) -> None:
        split_all_factors_random_range(
            self.graph,
            self.split_low,
            self.split_high,
            np.random.default_rng(self.split_seed),
        )


class DampedMidRunSplitEngine(MidRunSplitEngine):
    """mid-run splitting with the same Q-message damping hook as DampingEngine."""

    def __init__(self, *args, damping_factor: float = 0.9, **kwargs) -> None:
        self.damping_factor = float(damping_factor)
        super().__init__(*args, **kwargs)
        self._set_name({"damping": str(self.damping_factor)})

    def post_var_compute(self, var: Any) -> None:
        damp(var, self.damping_factor)
        var.append_last_iteration()


class AttentiveEngine(BPEngine):
    """min-sum where, each iteration, every variable discounts its incoming
    messages by the inverse of its degree before computing (the repo's
    `discount_attentive` policy)."""

    def step(self, i: int = 0):
        discount_attentive(self.graph)
        return super().step(i)


def run_full_horizon(engine: BPEngine, max_iter: int) -> list[float]:
    """step the engine for exactly max_iter iterations, preserving the cycle
    events (message normalization) but ignoring convergence stops so every run
    has the full horizon. returns the per-iteration global cost read from the
    snapshots API.
    """
    engine.convergence_monitor.reset()
    for i in range(max_iter):
        engine.step(i)
        try:
            engine._handle_cycle_events(i)
        except StopIteration:
            # converged: record nothing special, keep stepping for a full curve
            continue
    return [float(engine._snapshots[i].global_cost) for i in range(max_iter)]
