"""``DABPEngine`` — runs Deep Attentive Belief Propagation as a PropFlow engine.

DABP is a *learned* solver (a Graph-Attention network with GRU memory, trained
per-instance with AdamW) rather than a fixed-point message-passing rule. To make
it comparable with the other engines, this wrapper drives **one DABP BP-iteration
per ``engine.step(i)``**:

- a "restart" (``restart_period`` steps) resets DABP's message state but keeps the
  learned weights, mirroring ``DABP-main/alg/run.py``;
- a "phase" (``update_interval`` steps) is one training window: losses accumulate
  across the phase, then the top-``eff_iterations`` cheapest are backpropagated and
  the optimizer steps;
- every step, DABP's current assignment is read off the beliefs and its cost is
  evaluated on the **original** PropFlow cost tables, so the recorded
  ``global_cost`` is directly comparable to every other engine.

Importing this module requires the optional ``[dabp]`` extra (torch +
torch-geometric); the core package never imports it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.optim import AdamW

from ...bp.engine_base import BPEngine
from ...bp.engine_components import Step
from .build import build_dabp_inputs
from .constant import SCALE, dtype_for_device, select_device
from .model import AttentiveBP


class DABPEngine(BPEngine):
    """Deep Attentive Belief Propagation, exposed as a PropFlow engine.

    Args:
        num_head: number of attention heads in the GNN.
        update_interval: BP iterations per training/optimizer window.
        restart_period: BP iterations between message-state restarts.
        eff_iterations: top-k cheapest iterations whose loss is backpropagated.
        lr: AdamW learning rate.
        scale: cost magnitude scaling used internally during training (does not
            affect the reported cost).
        device: optional torch device override ("cpu"/"mps"/"cuda"); auto-selects
            cuda > mps > cpu when None.
    """

    engine_name = "DABPEngine"
    factor_splitting_enabled = True

    def __init__(
        self,
        *args,
        num_head: int = 4,
        update_interval: int = 20,
        restart_period: int = 2000,
        eff_iterations: int = 2,
        lr: float = 1e-4,
        scale: float = SCALE,
        device: str | None = None,
        **kwargs,
    ) -> None:
        self.num_head = int(num_head)
        self.update_interval = max(1, int(update_interval))
        self.restart_period = max(1, int(restart_period))
        self.eff_iterations = max(1, int(eff_iterations))
        self.lr = float(lr)
        self.scale = float(scale)
        self._device_pref = device
        self._abp: AttentiveBP | None = None
        super().__init__(*args, **kwargs)
        self._name = self.engine_name
        self._set_name({"heads": str(self.num_head), "ui": str(self.update_interval)})

    # ------------------------------------------------------------------ setup
    def post_init(self) -> None:
        """build the DABP tensors, model and optimizer from the factor graph."""
        self.device = select_device(self._device_pref)
        self.dtype = dtype_for_device(self.device)
        self._data, self._ordered_names, self._domain = build_dabp_inputs(
            self.graph,
            scale=self.scale,
            factor_splitting_enabled=self.factor_splitting_enabled,
        )
        self._abp = AttentiveBP(
            in_channels=12,
            out_channels=16,
            num_heads=self.num_head,
            msg_dim=self._domain,
        )
        self._abp.configure(self.device, self.dtype)
        self._abp.to(device=self.device, dtype=self.dtype)
        self._optimizer = AdamW(self._abp.parameters(), lr=self.lr, weight_decay=5e-5)
        self._phase_losses: list = []
        self._phase_costs: list = []
        self._dabp_assignment: dict = {name: 0 for name in self._ordered_names}
        self._dabp_beliefs: dict = {}

    # -------------------------------------------------------------- main loop
    def step(self, i: int = 0) -> Step:
        """advance DABP by one BP iteration and record the cost of its assignment."""
        self._dabp_iterate(i)
        step = Step(i)
        cost = self._cost_of(self._dabp_assignment)
        snapshot = self.snapshot_manager.capture_step(i, step, self)
        # DABP does not populate the NumPy agent mailboxes, so a snapshot manager
        # that derives cost from the agents would be wrong; DABP's own assignment
        # cost (scored on the original tables) is authoritative.
        snapshot.global_cost = cost
        self._snapshots[i] = snapshot
        self._last_cost = cost
        return step

    def _dabp_iterate(self, i: int) -> None:
        abp = self._require_abp()
        local = i % self.restart_period
        phase_pos = local % self.update_interval

        if local == 0:
            # restart: reset message/hidden state, keep learned weights
            abp.preprocess_single(self._data)
        if phase_pos == 0:
            # new training window: cut the autograd graph, clear accumulators
            abp.detach_state()
            self._optimizer.zero_grad()
            self._phase_losses = []
            self._phase_costs = []

        first_iter = phase_pos == 0
        loss, cost_internal, beliefs = abp.step_once(first_iter)
        if not first_iter:
            self._phase_losses.append(loss)
            self._phase_costs.append(cost_internal)
        self._update_assignment(beliefs)

        phase_end = phase_pos == self.update_interval - 1
        restart_end = local == self.restart_period - 1
        if (phase_end or restart_end) and self._phase_losses:
            self._train_phase()

    def _train_phase(self) -> None:
        """backprop the top-eff_iterations cheapest losses, then step the optimizer."""
        abp = self._require_abp()
        losses = self._phase_losses
        costs = self._phase_costs
        k = min(self.eff_iterations, len(losses))
        order = sorted(range(len(costs)), key=costs.__getitem__)
        top_k_loss = sum(losses[order[j]] for j in range(k)) / k
        top_k_loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        abp.detach_state()
        self._phase_losses = []
        self._phase_costs = []

    def _require_abp(self) -> AttentiveBP:
        if self._abp is None:
            raise RuntimeError("DABP model is not initialized.")
        return self._abp

    # ------------------------------------------------------------- readouts
    def _update_assignment(self, beliefs: np.ndarray) -> None:
        assignment: dict = {}
        bel: dict = {}
        for k, name in enumerate(self._ordered_names):
            row = np.asarray(beliefs[k][: self._domain], dtype=float)
            assignment[name] = int(np.argmin(row))
            bel[name] = row
        self._dabp_assignment = assignment
        self._dabp_beliefs = bel

    def _cost_of(self, assignment: dict) -> float:
        """score an assignment on the original (unsplit/unscaled) cost tables."""
        total = 0.0
        for factor in self.graph.original_factors:
            ct = factor.cost_table
            cn = getattr(factor, "connection_number", {}) or {}
            if ct is None or not cn:
                continue
            idx: list = [None] * ct.ndim
            ok = True
            for vname, dim in cn.items():
                if vname in assignment and dim < ct.ndim:
                    idx[dim] = assignment[vname]
                else:
                    ok = False
                    break
            if ok and None not in idx:
                total += float(ct[tuple(idx)])
        return total

    # ----------------------------------------------------- BPEngine overrides
    @property
    def assignments(self) -> dict:
        return dict(self._dabp_assignment)

    def get_beliefs(self) -> dict:
        return dict(self._dabp_beliefs)

    def _handle_cycle_events(self, i: int) -> None:
        # DABP does not use the NumPy mailers; skip normalization/convergence so
        # every run covers the full horizon, one DABP iteration per step.
        return None


class DABPEngineNoSplit(DABPEngine):
    """DABP variant that builds one DABP factor per original binary factor."""

    engine_name = "DABPEngineNoSplit"
    factor_splitting_enabled = False


DABPEngine_No_Split = DABPEngineNoSplit
