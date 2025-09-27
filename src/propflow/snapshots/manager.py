"""Snapshot Manager for Simulation Analysis.

This module defines the `SnapshotManager`, a class responsible for capturing,
storing, and analyzing the state of a simulation at each step. It can be
attached to a simulation engine to record compact snapshots and optionally
compute Jacobian matrices and cycle metrics for detailed analysis of the
algorithm's dynamics.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import json
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix

from .types import (
    SnapshotsConfig,
    SnapshotRecord,
    SnapshotData,
    Jacobians,
    CycleMetrics,
)
from .builder import build_snapshot_from_engine


class SnapshotManager:
    """Manages the lifecycle of in-memory snapshots and orchestrates analysis.

    This class maintains a ring buffer of `SnapshotRecord` objects, one for each
    step of the simulation. Based on its configuration, it can trigger the
    computation of Jacobian matrices and cycle metrics for each snapshot. It also
    handles the persistence of snapshots to disk.

    Attributes:
        config (SnapshotsConfig): The configuration for snapshot management.
    """

    def __init__(self, config: SnapshotsConfig):
        """Initializes the SnapshotManager.

        Args:
            config: A `SnapshotsConfig` object that specifies how snapshots
                should be handled (e.g., how many to retain, whether to run analysis).
        """
        self.config = config
        self._buffer: Dict[int, SnapshotRecord] = {}
        self._order: list[int] = []

    def capture_step(self, step_index: int, step: Any, engine: Any) -> SnapshotRecord:
        """Captures the state of the engine at a given step.

        This is the main entry point for the manager. It builds a `SnapshotData`
        object and then, if configured, runs various analyses on it.

        Args:
            step_index: The index of the current simulation step.
            step: The `Step` object containing message data for the step.
            engine: The simulation engine instance.

        Returns:
            The newly created `SnapshotRecord` containing the data and analysis results.
        """
        data = build_snapshot_from_engine(step_index, step, engine)
        jac, cycles, winners, min_idx = None, None, None, None

        if self.config.compute_jacobians or self.config.compute_cycles or self.config.compute_block_norms:
            jac = self._compute_jacobians(data)
        if self.config.compute_cycles and jac:
            cycles = self._compute_cycles(data, jac)

        if jac:
            min_idx = self._compute_min_idx(data)
            winners = self._compute_winners(data)

        rec = SnapshotRecord(data=data, jacobians=jac, cycles=cycles, winners=winners, min_idx=min_idx)
        self._store(step_index, rec)

        if self.config.save_each_step and self.config.save_dir:
            try:
                self.save_step(step_index, self.config.save_dir)
            except Exception:
                pass
        return rec

    def get(self, step_index: int) -> Optional[SnapshotRecord]:
        """Retrieves a snapshot record for a specific step index.

        Args:
            step_index: The step index to retrieve.

        Returns:
            The `SnapshotRecord` if it exists in the buffer, otherwise `None`.
        """
        return self._buffer.get(step_index)

    def latest(self) -> Optional[SnapshotRecord]:
        """Retrieves the most recent snapshot record from the buffer.

        Returns:
            The latest `SnapshotRecord`, or `None` if the buffer is empty.
        """
        if not self._order:
            return None
        return self._buffer.get(self._order[-1])

    def save_step(self, step_index: int, out_dir: str | Path) -> Path:
        """Saves a snapshot record for a specific step to disk.

        The snapshot is saved in a directory named `step_<n>`, containing the
        Jacobian matrices (if computed) and a `meta.json` file with all other
        serializable data.

        Args:
            step_index: The step index of the snapshot to save.
            out_dir: The base directory where the snapshot directory will be created.

        Returns:
            The `Path` object for the directory where the snapshot was saved.

        Raises:
            ValueError: If no snapshot exists for the given `step_index`.
        """
        rec = self.get(step_index)
        if rec is None:
            raise ValueError(f"No snapshot for step {step_index}")

        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        step_dir = base / f"step_{int(step_index):04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if rec.jacobians:
            from scipy.sparse import save_npz
            save_npz(step_dir / "A.npz", rec.jacobians.A)
            save_npz(step_dir / "P.npz", rec.jacobians.P)
            save_npz(step_dir / "B.npz", rec.jacobians.B)

        def to_list(arr):
            return np.asarray(arr).tolist() if arr is not None else None

        data = rec.data
        meta = {
            "step": data.step, "lambda": data.lambda_, "dom": data.dom,
            "N_var": data.N_var, "N_fac": data.N_fac,
            "Q": {f"{u}->{f}": to_list(vec) for (u, f), vec in data.Q.items()},
            "R": {f"{f}->{v}": to_list(vec) for (f, v), vec in data.R.items()},
            "block_norms": rec.jacobians.block_norms if rec.jacobians else None,
            "cycles": rec.cycles.to_dict() if rec.cycles else None,
            "winners": rec.winners,
            "min_idx": {f"{u}->{f}": int(i) for (u, f), i in (rec.min_idx or {}).items()},
        }
        with (step_dir / "meta.json").open("w") as fh:
            json.dump(meta, fh, indent=2)
        return step_dir

    def _store(self, idx: int, rec: SnapshotRecord) -> None:
        """Stores a record in the buffer and handles eviction if full."""
        self._buffer[idx] = rec
        self._order.append(idx)
        if self.config.retain_last is not None and len(self._order) > self.config.retain_last:
            oldest = self._order.pop(0)
            self._buffer.pop(oldest, None)

    def _build_slot_indices(
        self, data: SnapshotData
    ) -> Tuple[Dict[Tuple[str, str, int], int], Dict[Tuple[str, str, int], int]]:
        """Creates mappings from message slots to unique integer indices."""
        idxQ, q_id = {}, 0
        for u, labels in data.dom.items():
            for f in data.N_var.get(u, []):
                for a in range(len(labels)):
                    idxQ[(u, f, a)] = q_id
                    q_id += 1
        idxR, r_id = {}, 0
        for f, vars_ in data.N_fac.items():
            for v in vars_:
                for a in range(len(data.dom[v])):
                    idxR[(f, v, a)] = r_id
                    r_id += 1
        return idxQ, idxR

    def _compute_min_idx(self, data: SnapshotData) -> Dict[Tuple[str, str], int]:
        """Computes the index of the minimum value for each Q-message."""
        return {
            (u, f): int(np.argmin(q)) if q.size else 0
            for (u, f), q in data.Q.items()
        }

    def _compute_winners(
        self, data: SnapshotData
    ) -> Dict[Tuple[str, str, str], Dict[str, str]]:
        """Computes winning assignments for factor-to-variable messages."""
        winners = {}
        for f, vars_ in data.N_fac.items():
            cost_fn = data.cost.get(f, lambda assignment: 0.0)
            for v in vars_:
                others = [u for u in vars_ if u != v]
                for a_label in data.dom[v]:
                    best_val, best_assign = float("inf"), {}
                    label_lists = [data.dom[u] for u in others]
                    combo_iter = [()] if not label_lists else [c + (lbl,) for c in [()] for lbl in label_lists[0]]
                    for i in range(1, len(label_lists)):
                        combo_iter = [c + (lbl,) for c in combo_iter for lbl in label_lists[i]]

                    for combo in combo_iter:
                        assignment = {u: lbl for u, lbl in zip(others, combo)}
                        assignment[v] = a_label
                        factor_c = float(cost_fn(assignment))
                        msg_c = sum(
                            float(data.Q.get((u, f))[data.dom[u].index(assignment[u])])
                            for u in others if data.Q.get((u, f)) is not None
                        )
                        if factor_c + msg_c < best_val:
                            best_val, best_assign = factor_c + msg_c, {u: assignment[u] for u in others}
                    winners[(f, v, a_label)] = best_assign
        return winners

    def _build_A(self, data: SnapshotData, idxQ: dict, idxR: dict) -> csr_matrix:
        """Builds the A matrix (R -> Q dependencies) for the Jacobian."""
        rows, cols, vals = [], [], []
        for (u, f, a), row_idx in idxQ.items():
            for g in data.N_var.get(u, []):
                if g != f and (g, u, a) in idxR:
                    rows.append(row_idx)
                    cols.append(idxR[(g, u, a)])
                    vals.append(1.0)
        return csr_matrix((vals, (rows, cols)), shape=(len(idxQ), len(idxR)))

    def _build_P(self, data: SnapshotData, min_idx: dict, idxQ: dict) -> csr_matrix:
        """Builds the P matrix (projection for min-sum) for the Jacobian."""
        P = lil_matrix((len(idxQ), len(idxQ)))
        groups = {}
        for (u, f, a), i in idxQ.items():
            groups.setdefault((u, f), []).append((a, i))
        for edge, items in groups.items():
            items.sort()
            m_idx = next((i for a, i in items if a == int(min_idx.get(edge, 0))), items[0][1])
            for _, i in items:
                P[i, i] = 1.0
                P[i, m_idx] -= 1.0
        return P.tocsr()

    def _build_B(self, data: SnapshotData, winners: dict, idxQ: dict, idxR: dict) -> csr_matrix:
        """Builds the B matrix (Q -> R dependencies) for the Jacobian."""
        rows, cols, vals = [], [], []
        for (f, v, a), row_idx in idxR.items():
            others = [u for u in data.N_fac.get(f, []) if u != v]
            w = winners.get((f, v, data.dom[v][a]), {}) if a < len(data.dom[v]) else {}
            for u in others:
                lbl = w.get(u)
                if lbl is not None:
                    try:
                        ai = data.dom[u].index(lbl)
                        if (u, f, ai) in idxQ:
                            rows.append(row_idx)
                            cols.append(idxQ[(u, f, ai)])
                            vals.append(1.0)
                    except ValueError:
                        pass
        return csr_matrix((vals, (rows, cols)), shape=(len(idxR), len(idxQ)))

    def _compute_block_norms(self, A: csr_matrix, P: csr_matrix, B: csr_matrix, lam: float) -> Dict[str, float]:
        """Computes the infinity norms of the Jacobian blocks."""
        def inf_norm(M: csr_matrix) -> float:
            if M.nnz == 0: return 0.0
            row_sums = np.asarray(abs(M).sum(axis=1)).ravel()
            return float(row_sums.max()) if row_sums.size else 0.0
        PA, BPA = P @ A, B @ (P @ A)
        tl = inf_norm(BPA) if lam < 1 else 0.0
        tr = inf_norm(B) if lam > 0 else 0.0
        bl = inf_norm(PA) if lam < 1 else 0.0
        return {
            "||BPA||_inf": tl / (1 - lam) if lam < 1 else 0.0,
            "||B||_inf": tr / lam if lam > 0 else 0.0,
            "||PA||_inf": bl / (1 - lam) if lam < 1 else 0.0,
            "||M||_inf_upper": max(tl + tr, bl + lam),
        }

    def _compute_jacobians(self, data: SnapshotData) -> Jacobians:
        """Orchestrates the computation of all Jacobian-related data."""
        idxQ, idxR = self._build_slot_indices(data)
        winners = self._compute_winners(data)
        min_idx = self._compute_min_idx(data)
        A = self._build_A(data, idxQ, idxR)
        P = self._build_P(data, min_idx, idxQ)
        B = self._build_B(data, winners, idxQ, idxR)
        norms = self._compute_block_norms(A, P, B, data.lambda_) if self.config.compute_block_norms else None
        return Jacobians(idxQ=idxQ, idxR=idxR, A=A, P=P, B=B, block_norms=norms)

    def _compute_cycles(self, data: SnapshotData, jac: Jacobians) -> CycleMetrics:
        """Computes cycle metrics from the graph structure derived from the Jacobians."""
        G = nx.DiGraph()
        revQ = {i: key for key, i in jac.idxQ.items()}
        revR = {i: key for key, i in jac.idxR.items()}
        for i in revQ: G.add_node(("Q", revQ[i]))
        for i in revR: G.add_node(("R", revR[i]))
        Bcoo = jac.B.tocoo()
        for r, q in zip(Bcoo.row, Bcoo.col):
            if r in revR and q in revQ: G.add_edge(("R", revR[r]), ("Q", revQ[q]))
        Acoo = jac.A.tocoo()
        for q, r in zip(Acoo.row, Acoo.col):
            if q in revQ and r in revR: G.add_edge(("Q", revQ[q]), ("R", revR[r]))

        cycles_list = list(nx.simple_cycles(G)) if self.config.max_cycle_len is None else \
                      [cyc for cyc in nx.simple_cycles(G) if len(cyc) <= self.config.max_cycle_len]

        has_cert = jac.block_norms and jac.block_norms.get("||M||_inf_upper", 1.0) < 1.0
        aligned_total, details = 0, None
        if self.config.include_detailed_cycles:
            details = []
            winners, min_idx = self._compute_winners(data), self._compute_min_idx(data)
            for cyc in cycles_list:
                aligned = self._cycle_has_aligned_hop(cyc, winners, min_idx, jac.idxQ, jac.idxR, data)
                if aligned: aligned_total += 1
                item = {"length": len(cyc), "aligned": aligned, "bound": (data.lambda_**len(cyc)) if aligned else None}
                if self.config.compute_numeric_cycle_gain:
                    item["numeric_gain_inf"] = self._estimate_cycle_gain_inf(cyc, jac.A, jac.P, jac.B, data.lambda_, jac.idxQ, jac.idxR)
                details.append(item)
        return CycleMetrics(num_cycles=len(cycles_list), aligned_hops_total=aligned_total, has_certified_contraction=has_cert, details=details)

    def _cycle_has_aligned_hop(self, cycle: list, winners: dict, min_idx: dict, idxQ: dict, idxR: dict, data: SnapshotData) -> bool:
        """Checks if a cycle contains an 'aligned hop'."""
        for i in range(len(cycle)):
            current, nxt = cycle[i], cycle[(i + 1) % len(cycle)]
            if current[0] == "R" and nxt[0] == "Q":
                f, v, a_idx = current[1]
                u, f2, _ = nxt[1]
                if f != f2: continue
                labels_v = data.dom.get(v, [])
                if not (0 <= a_idx < len(labels_v)): continue
                target_label = labels_v[a_idx]
                w = winners.get((f, v, target_label), {})
                winning_label = w.get(u)
                if winning_label is None: continue
                try:
                    winning_idx = data.dom[u].index(winning_label)
                    if min_idx.get((u, f)) == winning_idx: return True
                except ValueError:
                    continue
        return False

    def _estimate_cycle_gain_inf(self, cycle: list, A: csr_matrix, P: csr_matrix, B: csr_matrix, lam: float, idxQ: dict, idxR: dict) -> float | None:
        """Estimates the infinity norm gain of a message cycle."""
        try:
            PA, val = P @ A, 1.0
            for i in range(len(cycle)):
                cur, nxt = cycle[i], cycle[(i + 1) % len(cycle)]
                if cur[0] == "R" and nxt[0] == "Q":
                    r, q = idxR.get(cur[1]), idxQ.get(nxt[1])
                    if r is None or q is None: continue
                    val *= 1.0
                elif cur[0] == "Q" and nxt[0] == "R":
                    q, r = idxQ.get(cur[1]), idxR.get(nxt[1])
                    if q is None or r is None: continue
                    if q < PA.shape[0] and r < PA.shape[1]: val *= float(PA[q, r])
            return abs(val) * (lam ** len(cycle))
        except Exception:
            return None
