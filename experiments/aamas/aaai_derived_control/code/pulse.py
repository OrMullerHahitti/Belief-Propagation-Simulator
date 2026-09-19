"""Native experimental engine for a temporary change of pairwise split weights."""

from __future__ import annotations

from typing import Any

import numpy as np

from propflow.bp.engines import DampingSCFGEngine
from propflow.policies.splitting import split_all_factors


class SplitPulseEngine(DampingSCFGEngine):
    """Temporarily change complementary pairwise splits without resetting Q/R.

    Defaults match the frozen experiment: .5/.5 initially, .95/.05 before
    zero-based step 64, and .5/.5 again before step 256; old-Q damping stays .9.
    Unary factors retain their initial split. Run with the paper's
    ``run_full_horizon`` helper so an early convergence stop does not omit the
    intervention or the restoration. This is a fixed schedule, not a learner.
    """

    def __init__(
        self,
        *args: Any,
        pulse_start: int = 64,
        pulse_stop: int = 256,
        pulse_weight: float = 0.95,
        **kwargs: Any,
    ) -> None:
        if not 0 <= pulse_start < pulse_stop:
            raise ValueError("require 0 <= pulse_start < pulse_stop")
        if not np.isfinite(pulse_weight) or not 0 < pulse_weight < 1:
            raise ValueError("pulse_weight must be finite and between zero and one")
        self.pulse_start = pulse_start
        self.pulse_stop = pulse_stop
        self.pulse_weight = float(pulse_weight)
        self.split_events: list[dict[str, float | int]] = []
        kwargs.setdefault("split_factor", 0.5)
        kwargs.setdefault("damping_factor", 0.9)
        kwargs.setdefault("normalize_messages", True)
        kwargs.setdefault("anytime", False)
        self._base_weight = float(kwargs["split_factor"])
        super().__init__(*args, **kwargs)
        self._name = "SplitPulseEngine"

    def post_init(self) -> None:
        """Save original tables and use the native original-to-clone mapping."""
        original = list(self.graph.factors)
        if any(len(f.connection_number) not in (1, 2) for f in original):
            raise ValueError("the pulse experiment supports unary/pairwise factors")
        tables = {
            f.name: np.array(f.cost_table, dtype=float, copy=True) for f in original
        }
        mapping = split_all_factors(self.graph, self.split_factor)
        self._pulse_pairs = [
            (tables[f.name], mapping[f.name])
            for f in original
            if len(f.connection_number) == 2
        ]

    def step(self, i: int = 0):
        """Apply the two table changes before the corresponding native update."""
        if i in (self.pulse_start, self.pulse_stop):
            weight = self.pulse_weight if i == self.pulse_start else self._base_weight
            for original, clones in self._pulse_pairs:
                clones[0].cost_table = weight * original
                clones[1].cost_table = (1 - weight) * original
            self.split_events.append({"step": i, "weight": weight})
        return super().step(i)
