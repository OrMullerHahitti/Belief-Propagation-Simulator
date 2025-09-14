from __future__ import annotations

"""
SnapshotManager: lifecycle and analysis for per-step snapshots.

Attach this to the engine when initializing with snapshots=True. It records
compact per-step snapshots and, optionally, computes Jacobian blocks and a
light cycle summary for focused iteration analysis.
"""

from typing import Dict, Optional, Tuple, List
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
    """
    Owns the in-memory snapshot ring buffer and orchestrates optional analysis.
    """

    def __init__(self, config: SnapshotsConfig):
        self.config = config
        self._buffer: Dict[int, SnapshotRecord] = {}
        self._order: list[int] = []

    # --- Public API ---------------------------------------------------------
    def capture_step(self, step_index: int, step, engine) -> SnapshotRecord:
        data = build_snapshot_from_engine(step_index, step, engine)

        jac: Optional[Jacobians] = None
        cycles: Optional[CycleMetrics] = None

        if (
            self.config.compute_jacobians
            or self.config.compute_cycles
            or self.config.compute_block_norms
        ):
            jac = self._compute_jacobians(data)

        if self.config.compute_cycles and jac is not None:
            cycles = self._compute_cycles(data, jac)

        # winners/min_idx requested to be retained for downstream use
        winners = None
        min_idx = None
        if jac is not None:
            min_idx = self._compute_min_idx(data)
            winners = self._compute_winners(data)

        rec = SnapshotRecord(
            data=data, jacobians=jac, cycles=cycles, winners=winners, min_idx=min_idx
        )
        self._store(step_index, rec)

        # Optional auto-save if configured
        if self.config.save_each_step and self.config.save_dir:
            try:
                self.save_step(step_index, self.config.save_dir)
            except Exception:
                pass
        return rec

    def get(self, step_index: int) -> Optional[SnapshotRecord]:
        return self._buffer.get(step_index)

    def latest(self) -> Optional[SnapshotRecord]:
        if not self._order:
            return None
        return self._buffer.get(self._order[-1])

    # --- Persistence --------------------------------------------------------
    def save_step(self, step_index: int, out_dir: str | Path) -> Path:
        """
        Save a snapshot record for a specific step to disk.

        Layout:
            <out_dir>/step_<n>/
                meta.json          # domains, neighborhoods, winners, min_idx, Q/R
                A.npz, P.npz, B.npz

        Returns the directory path for the saved snapshot.
        """
        rec = self.get(step_index)
        if rec is None:
            raise ValueError(f"No snapshot for step {step_index}")

        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        step_dir = base / f"step_{int(step_index):04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save matrices if present
        if rec.jacobians is not None:
            from scipy.sparse import save_npz

            save_npz(step_dir / "A.npz", rec.jacobians.A)
            save_npz(step_dir / "P.npz", rec.jacobians.P)
            save_npz(step_dir / "B.npz", rec.jacobians.B)

        # Build JSON-serializable meta
        def to_list(arr):
            import numpy as _np

            if arr is None:
                return None
            a = _np.asarray(arr)
            return a.tolist()

        data = rec.data
        meta = {
            "step": data.step,
            "lambda": data.lambda_,
            "dom": data.dom,
            "N_var": data.N_var,
            "N_fac": data.N_fac,
            "Q": {f"{u}->{f}": to_list(vec) for (u, f), vec in data.Q.items()},
            "R": {f"{f}->{v}": to_list(vec) for (f, v), vec in data.R.items()},
            "block_norms": rec.jacobians.block_norms if rec.jacobians else None,
            "cycles": {
                "num_cycles": rec.cycles.num_cycles if rec.cycles else None,
                "aligned_hops_total": rec.cycles.aligned_hops_total
                if rec.cycles
                else None,
                "has_certified_contraction": rec.cycles.has_certified_contraction
                if rec.cycles
                else None,
                "details": rec.cycles.details if rec.cycles else None,
            },
            "winners": rec.winners,
            "min_idx": {
                f"{u}->{f}": int(i) for (u, f), i in (rec.min_idx or {}).items()
            },
        }

        with (step_dir / "meta.json").open("w") as fh:
            json.dump(meta, fh, indent=2)

        return step_dir

    # --- Internals ----------------------------------------------------------
    def _store(self, idx: int, rec: SnapshotRecord) -> None:
        self._buffer[idx] = rec
        self._order.append(idx)
        if (
            self.config.retain_last is not None
            and len(self._order) > self.config.retain_last
        ):
            # Drop oldest
            oldest = self._order.pop(0)
            self._buffer.pop(oldest, None)

    # ----- Core analysis helpers (self-contained) --------------------------
    def _build_slot_indices(
        self, data: SnapshotData
    ) -> Tuple[Dict[Tuple[str, str, int], int], Dict[Tuple[str, str, int], int]]:
        idxQ: Dict[Tuple[str, str, int], int] = {}
        idxR: Dict[Tuple[str, str, int], int] = {}

        q_id = 0
        for u, labels in data.dom.items():
            d = len(labels)
            for f in data.N_var.get(u, []):
                for a in range(d):
                    idxQ[(u, f, a)] = q_id
                    q_id += 1

        r_id = 0
        for f, vars_ in data.N_fac.items():
            for v in vars_:
                d = len(data.dom[v])
                for a in range(d):
                    idxR[(f, v, a)] = r_id
                    r_id += 1

        return idxQ, idxR

    def _compute_min_idx(self, data: SnapshotData) -> Dict[Tuple[str, str], int]:
        out: Dict[Tuple[str, str], int] = {}
        for (u, f), q in data.Q.items():
            if q.size:
                out[(u, f)] = int(np.argmin(q))
            else:
                out[(u, f)] = 0
        return out

    def _compute_winners(
        self, data: SnapshotData
    ) -> Dict[Tuple[str, str, str], Dict[str, str]]:
        winners: Dict[Tuple[str, str, str], Dict[str, str]] = {}

        for f, vars_ in data.N_fac.items():
            cost_fn = data.cost.get(f, lambda assignment: 0.0)
            for v in vars_:
                others = [u for u in vars_ if u != v]
                for a_label in data.dom[v]:
                    best_val = float("inf")
                    best_assign: Dict[str, str] = {}

                    # Brute-force over small product of labels for 'others'
                    # Build lists of label options
                    label_lists = [data.dom[u] for u in others]
                    # Simple nested loops using indices
                    if not label_lists:
                        combo_iter = [()]
                    else:
                        # iterative cartesian product
                        combo_iter = [()]
                        for labels in label_lists:
                            combo_iter = [
                                c + (lbl,) for c in combo_iter for lbl in labels
                            ]

                    for combo in combo_iter:
                        assignment = {u: lbl for u, lbl in zip(others, combo)}
                        assignment[v] = a_label

                        # factor cost + message costs from others
                        try:
                            factor_c = float(cost_fn(assignment))
                        except Exception:
                            factor_c = 0.0

                        msg_c = 0.0
                        for u in others:
                            q = data.Q.get((u, f))
                            if q is not None:
                                try:
                                    idx = data.dom[u].index(assignment[u])
                                    msg_c += float(q[idx])
                                except Exception:
                                    pass

                        total = factor_c + msg_c
                        if total < best_val:
                            best_val = total
                            best_assign = {u: assignment[u] for u in others}

                    winners[(f, v, a_label)] = best_assign

        return winners

    def _build_A(self, data: SnapshotData, idxQ, idxR) -> csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        for (u, f, a), row_idx in idxQ.items():
            for g in data.N_var.get(u, []):
                if g == f:
                    continue
                col_key = (g, u, a)
                if col_key in idxR:
                    rows.append(row_idx)
                    cols.append(idxR[col_key])
                    vals.append(1.0)

        return csr_matrix((vals, (rows, cols)), shape=(len(idxQ), len(idxR)))

    def _build_P(
        self, data: SnapshotData, min_idx: Dict[Tuple[str, str], int], idxQ
    ) -> csr_matrix:
        nQ = len(idxQ)
        P = lil_matrix((nQ, nQ))

        # group slot indices per edge (u,f)
        groups: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for (u, f, a), i in idxQ.items():
            groups.setdefault((u, f), []).append((a, i))

        for edge, items in groups.items():
            items.sort()
            u, f = edge
            m = int(min_idx.get((u, f), 0))
            # find matrix index for min label
            m_idx = next((i for a, i in items if a == m), items[0][1])
            for a, i in items:
                P[i, i] = 1.0
                P[i, m_idx] -= 1.0

        return P.tocsr()

    def _build_B(self, data: SnapshotData, winners, idxQ, idxR) -> csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        for (f, v, a), row_idx in idxR.items():
            others = [u for u in data.N_fac.get(f, []) if u != v]
            w = winners.get((f, v, data.dom[v][a]), {}) if a < len(data.dom[v]) else {}
            for u in others:
                lbl = w.get(u)
                if lbl is None:
                    continue
                try:
                    ai = data.dom[u].index(lbl)
                except ValueError:
                    ai = 0
                col_key = (u, f, ai)
                if col_key in idxQ:
                    rows.append(row_idx)
                    cols.append(idxQ[col_key])
                    vals.append(1.0)

        return csr_matrix((vals, (rows, cols)), shape=(len(idxR), len(idxQ)))

    def _compute_block_norms(
        self, A: csr_matrix, P: csr_matrix, B: csr_matrix, lam: float
    ) -> Dict[str, float]:
        def inf_norm(M: csr_matrix) -> float:
            if M.nnz == 0:
                return 0.0
            row_sums = np.asarray(abs(M).sum(axis=1)).ravel()
            return float(row_sums.max() if row_sums.size else 0.0)

        PA = P @ A
        BPA = B @ PA

        tl = inf_norm(BPA) if lam < 1 else 0.0
        tr = inf_norm(B) if lam > 0 else 0.0
        bl = inf_norm(PA) if lam < 1 else 0.0

        upper = max(tl + tr, bl + lam)

        out: Dict[str, float] = {
            "||BPA||_inf": tl / (1 - lam) if lam < 1 else 0.0,
            "||B||_inf": tr / lam if lam > 0 else 0.0,
            "||PA||_inf": bl / (1 - lam) if lam < 1 else 0.0,
            "||M||_inf_upper": upper,
        }
        return out

    def _compute_jacobians(self, data: SnapshotData) -> Jacobians:
        idxQ, idxR = self._build_slot_indices(data)
        winners = self._compute_winners(data)
        min_idx = self._compute_min_idx(data)

        A = self._build_A(data, idxQ, idxR)
        P = self._build_P(data, min_idx, idxQ)
        B = self._build_B(data, winners, idxQ, idxR)

        norms = None
        if self.config.compute_block_norms:
            norms = self._compute_block_norms(A, P, B, data.lambda_)

        return Jacobians(idxQ=idxQ, idxR=idxR, A=A, P=P, B=B, block_norms=norms)

    def _compute_cycles(self, data: SnapshotData, jac: Jacobians) -> CycleMetrics:
        # Build a slot graph: nodes are Q and R slots; edges follow non-zeros in A and B
        G = nx.DiGraph()

        # Reverse maps for readability
        revQ = {i: key for key, i in jac.idxQ.items()}
        revR = {i: key for key, i in jac.idxR.items()}

        # Add nodes
        for i in revQ:
            G.add_node(("Q", revQ[i]))
        for i in revR:
            G.add_node(("R", revR[i]))

        # Edges from B: R->Q
        Bcoo = jac.B.tocoo()
        for r, q in zip(Bcoo.row, Bcoo.col):
            if r in revR and q in revQ:
                G.add_edge(("R", revR[r]), ("Q", revQ[q]))

        # Edges from A: Q->R (through P @ A is not necessary just for structure)
        Acoo = jac.A.tocoo()
        for q, r in zip(Acoo.row, Acoo.col):
            if q in revQ and r in revR:
                G.add_edge(("Q", revQ[q]), ("R", revR[r]))

        # Enumerate simple cycles up to max length
        max_len = max(3, int(self.config.max_cycle_len))
        cycles_list = []
        try:
            for cyc in nx.simple_cycles(G):
                if len(cyc) <= max_len:
                    cycles_list.append(cyc)
        except Exception:
            cycles_list = []

        # A coarse indicator of contraction via block norms upper bound
        has_cert = False
        if jac.block_norms and jac.block_norms.get("||M||_inf_upper", 1.0) < 1.0:
            has_cert = True

        # Optionally compute per-cycle aligned hop info and numeric gain
        details = None
        aligned_total = 0
        if self.config.include_detailed_cycles:
            details = []
            winners = self._compute_winners(data)
            min_idx = self._compute_min_idx(data)
            for cyc in cycles_list:
                length = len(cyc)
                aligned = self._cycle_has_aligned_hop(
                    cyc, winners, min_idx, jac.idxQ, jac.idxR, data
                )
                if aligned:
                    aligned_total += 1
                item = {
                    "length": length,
                    "aligned": aligned,
                    "bound": (data.lambda_**length) if aligned else None,
                }
                if self.config.compute_numeric_cycle_gain:
                    item["numeric_gain_inf"] = self._estimate_cycle_gain_inf(
                        cyc, jac.A, jac.P, jac.B, data.lambda_, jac.idxQ, jac.idxR
                    )
                details.append(item)

        return CycleMetrics(
            num_cycles=int(len(cycles_list)),
            aligned_hops_total=int(aligned_total),
            has_certified_contraction=has_cert,
            details=details,
        )

    # --- Cycle helpers ------------------------------------------------------
    def _cycle_has_aligned_hop(
        self, cycle, winners, min_idx, idxQ, idxR, data: SnapshotData
    ) -> bool:
        for i in range(len(cycle)):
            current = cycle[i]
            nxt = cycle[(i + 1) % len(cycle)]
            if current[0] == "R" and nxt[0] == "Q":
                f, v, a_idx = current[1]
                u, f2, _ = nxt[1]
                if f != f2:
                    continue
                # Map label index to label string
                labels_v = data.dom.get(v, [])
                if a_idx < 0 or a_idx >= len(labels_v):
                    continue
                target_label = labels_v[a_idx]
                w = winners.get((f, v, target_label), {})
                winning_label = w.get(u)
                if winning_label is None:
                    continue
                try:
                    winning_idx = data.dom[u].index(winning_label)
                except Exception:
                    continue
                if min_idx.get((u, f)) == winning_idx:
                    return True
        return False

    def _estimate_cycle_gain_inf(
        self, cycle, A: csr_matrix, P: csr_matrix, B: csr_matrix, lam: float, idxQ, idxR
    ) -> float | None:
        # Scalar product approximation similar in spirit to analyzer implementation
        try:
            PA = P @ A
            val = 1.0
            for i in range(len(cycle)):
                cur = cycle[i]
                nxt = cycle[(i + 1) % len(cycle)]
                if cur[0] == "R" and nxt[0] == "Q":
                    r = idxR.get(cur[1])
                    q = idxQ.get(nxt[1])
                    if r is None or q is None:
                        continue
                    # For structure edges from B, use 1.0 as adjacency
                    val *= 1.0
                elif cur[0] == "Q" and nxt[0] == "R":
                    q = idxQ.get(cur[1])
                    r = idxR.get(nxt[1])
                    if q is None or r is None:
                        continue
                    if q < PA.shape[0] and r < PA.shape[1]:
                        val *= float(PA[q, r])
            return abs(val) * (lam ** len(cycle))
        except Exception:
            return None
