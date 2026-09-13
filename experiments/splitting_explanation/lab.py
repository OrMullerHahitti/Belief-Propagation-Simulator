"""lab: fast vectorized synchronous min-sum on pairwise factor graphs, with splitting, damping,
mid-run splitting, the reduced "cavity + belief" update and its ingredient variants, commitment
metrics, period detection and local-search neighbourhoods.

propflow semantics (checked by exp0_checks.py against BPEngine / SplitEngine / DampingEngine /
DampingSCFGEngine / DampedMidRunSplitEngine and against the recorded AAAI cost curves):
  iteration t: (1) every variable computes Q_t from the R_{t-1} inbox, excluding the recipient
                   factor; with damping the sent message is lam*Q_sent_{t-1} + (1-lam)*Q_computed
               (2) every factor computes R_t from the sent Q_t
  assignment:  argmin of the belief b_i = theta_i + sum of incoming R_t
splitting: every factor is replaced by two copies carrying p*C and (1-p)*C (an SCFG).

rule="cav+belief" runs on the ORIGINAL graph with the variable update Q = cavity + belief and
reproduces the symmetric SCFG exactly (Theorem 1 of the explanation); rule="belief" keeps only
the echo ingredient and rule="2cav" only the doubling ingredient.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EPS = 1e-9


# ----------------------------------------------------------------------------------------------
# instances
# ----------------------------------------------------------------------------------------------
@dataclass
class Inst:
    n: int
    m: int
    edges: list  # (i, j) with i < j
    C: list  # C[k][x_i, x_j]
    theta: np.ndarray  # (n, m) unary costs (tie-break preferences)
    ei: np.ndarray = field(init=False, repr=False)
    ej: np.ndarray = field(init=False, repr=False)
    Cs: np.ndarray = field(init=False, repr=False)
    E: int = field(init=False)

    def __post_init__(self):
        self.E = len(self.edges)
        self.ei = np.array([i for i, _ in self.edges], dtype=int)
        self.ej = np.array([j for _, j in self.edges], dtype=int)
        self.Cs = np.stack(self.C) if self.E else np.zeros((0, self.m, self.m))

    def cost(self, x) -> float:
        x = np.asarray(x)
        return float(self.theta[np.arange(self.n), x].sum() + self.edge_costs(x).sum())

    def edge_costs(self, x) -> np.ndarray:
        x = np.asarray(x)
        return self.Cs[np.arange(self.E), x[self.ei], x[self.ej]]

    def pair_cost(self, x, y) -> float:
        """alternation cost cost_2(x, y) = sum_e C(x_i, y_j) + C(y_i, x_j) + unaries of both layers."""
        x = np.asarray(x)
        y = np.asarray(y)
        ar = np.arange(self.E)
        c = (
            self.theta[np.arange(self.n), x].sum()
            + self.theta[np.arange(self.n), y].sum()
        )
        c += (
            self.Cs[ar, x[self.ei], y[self.ej]].sum()
            + self.Cs[ar, y[self.ei], x[self.ej]].sum()
        )
        return float(c)

    def degrees(self) -> np.ndarray:
        d = np.zeros(self.n, dtype=int)
        np.add.at(d, self.ei, 1)
        np.add.at(d, self.ej, 1)
        return d

    def adjacency(self) -> list[list[int]]:
        adj = [[] for _ in range(self.n)]
        for i, j in self.edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj


def _components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), set()).add(i)
    return list(comps.values())


def _connect(n, edges, rng):
    comp = _components(n, edges)
    for c in comp[1:]:
        i = int(rng.choice(sorted(comp[0])))
        j = int(rng.choice(sorted(c)))
        edges.append((min(i, j), max(i, j)))
        comp[0] |= c
    return sorted(set(edges))


def random_inst(n, m, density, low=100, high=200, seed=0, tiebreak=1e-2) -> Inst:
    """the AAAI random family: integer costs U[low, high), tiny unary tie-break preferences."""
    rng = np.random.default_rng(seed)
    edges = [
        (i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < density
    ]
    edges = _connect(n, edges, rng)
    C = [rng.integers(low, high, size=(m, m)).astype(float) for _ in edges]
    theta = rng.uniform(0, tiebreak, size=(n, m))
    return Inst(n, m, edges, C, theta)


def bipartite_inst(
    n_left, n_right, m, density, low=100, high=200, seed=0, tiebreak=1e-2
) -> Inst:
    """random bipartite constraint graph (edges only between the two sides)."""
    rng = np.random.default_rng(seed)
    n = n_left + n_right
    edges = [
        (i, j)
        for i in range(n_left)
        for j in range(n_left, n)
        if rng.random() < density
    ]
    if not edges:
        edges = [(0, n_left)]
    comp = sorted(_components(n, edges), key=len, reverse=True)
    # the largest component holds at least one edge, so it has nodes on both sides; every other
    # component is joined to it by one edge across the sides (a lone left node needs a right partner)
    main = comp[0]
    main_left = sorted(v for v in main if v < n_left)
    main_right = sorted(v for v in main if v >= n_left)
    for c in comp[1:]:
        u = min(c)
        if u < n_left:
            i, j = u, int(rng.choice(main_right))
        else:
            i, j = int(rng.choice(main_left)), u
        edges.append((i, j))
    edges = sorted(set(edges))
    C = [rng.integers(low, high, size=(m, m)).astype(float) for _ in edges]
    theta = rng.uniform(0, tiebreak, size=(n, m))
    return Inst(n, m, edges, C, theta)


def coloring_inst(n, colors=3, density=0.1, cost=10.0, seed=0, tiebreak=1e-2) -> Inst:
    rng = np.random.default_rng(seed)
    edges = [
        (i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < density
    ]
    edges = _connect(n, edges, rng)
    C = [np.eye(colors) * cost for _ in edges]
    theta = rng.uniform(0, tiebreak, size=(n, colors))
    return Inst(n, colors, edges, C, theta)


def aaai_inst(benchmark: str, seed: int) -> Inst:
    """the exact AAAI benchmark instance (experiments/aaai/code/problems.py) as an Inst."""
    code = str(REPO / "experiments" / "aaai" / "code")
    if code not in sys.path:
        sys.path.insert(0, code)
    from problems import BENCHMARKS  # noqa: E402  (propflow-based builders)

    fg = BENCHMARKS[benchmark](seed)
    names = sorted((v.name for v in fg.variables), key=lambda s: int(s[1:]))
    idx = {nm: k for k, nm in enumerate(names)}
    n = len(names)
    m = fg.variables[0].domain
    theta = np.zeros((n, m))
    edges, C = [], []
    for f, vs in fg.edges.items():
        table = np.array(f.cost_table, dtype=float)
        if len(vs) == 1:
            theta[idx[vs[0].name]] += table
        elif len(vs) == 2:
            # cost-table axis k belongs to the k-th variable of the edge list (connection_number)
            a, b = idx[vs[0].name], idx[vs[1].name]
            if a < b:
                edges.append((a, b))
                C.append(table)
            else:
                edges.append((b, a))
                C.append(table.T)
        else:
            raise ValueError("pairwise benchmarks only")
    order = sorted(range(len(edges)), key=lambda k: edges[k])
    return Inst(n, m, [edges[k] for k in order], [C[k] for k in order], theta)


# ----------------------------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------------------------
RULES = ("cav", "belief", "2cav", "cav+belief")


class FastEngine:
    """synchronous min-sum with splitting (split=p), Q-damping (lam) and a mid-run split.

    rule (only without splitting) selects the variable update on the original graph:
      cav        : Q = belief - R_own            (plain min-sum)
      belief     : Q = belief                    (echo only: the recipient's own R stays in)
      2cav       : Q = 2 (belief - R_own)        (doubling only)
      cav+belief : Q = 2 belief - R_own          (both; identical to the 0.5/0.5 split, Theorem 1)
    """

    def __init__(
        self,
        inst: Inst,
        split: float | None = None,
        lam: float = 0.0,
        rule: str = "cav",
        floor_q: bool = False,
        norm_every: int | None = None,
    ):
        """floor_q / norm_every reproduce two propflow implementation details bit for bit:
        - propflow's compute_R casts every Q message to the cost table's dtype, so on the integer
          AAAI tables (random_dense, random_sparse) the factors see trunc(Q) until the factors are
          split (the clones p*C are float tables) -> floor_q=True while the graph is unsplit;
        - propflow subtracts the per-message minimum from the R inbox and the last sent Q only every
          graph_diameter iterations (normalize_messages=True) -> norm_every=graph_diameter.
        with floats both are equivalent to the default per-step normalisation (constant shifts never
        move an argmin); with truncation the schedule matters, hence the exact mode."""
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}")
        if split is not None and rule != "cav":
            raise ValueError("rules other than 'cav' run on the unsplit graph only")
        if split is not None and floor_q:
            raise ValueError(
                "floor_q models integer tables; split clones are float tables in propflow"
            )
        self.inst = inst
        self.lam = float(lam)
        self.rule = rule
        self.floor_q = bool(floor_q)
        self.norm_every = norm_every
        self.t = 0
        self._build([split, 1.0 - split] if split is not None else [1.0])
        self.R = np.zeros((self.A, inst.m))  # R_{t-1} messages arriving at dst
        self.Q = np.zeros(
            (self.A, inst.m)
        )  # last sent Q along arc (src -> the arc's factor)
        self.theta_msg = np.asarray(
            inst.theta, dtype=float
        ).copy()  # the unary factors' message
        self._undamped_next = True  # no damping history yet

    def _build(self, weights):
        inst = self.inst
        src, dst, Ct, rev, eid, cid = [], [], [], [], [], []
        for k, (i, j) in enumerate(inst.edges):
            for c, w in enumerate(weights):
                a1 = len(src)
                src.append(i)
                dst.append(j)
                Ct.append(w * inst.C[k])
                eid.append(k)
                cid.append(c)
                a2 = len(src)
                src.append(j)
                dst.append(i)
                Ct.append(w * inst.C[k].T)
                eid.append(k)
                cid.append(c)
                rev.extend([a2, a1])
        self.src = np.array(src)
        self.dst = np.array(dst)
        self.Ct = np.stack(Ct)  # (A, m, m): Ct[a][x_src, x_dst]
        self.rev = np.array(rev)
        self.eid = np.array(eid)
        self.cid = np.array(cid)
        self.A = len(src)
        self.weights = list(weights)

    def split_now(self, p: float = 0.5) -> None:
        """mid-run split in propflow's transfer mode: every clone inherits its share (p, 1-p) of the
        current R message on each arc, and the damping history is dropped (first step undamped)."""
        if len(self.weights) != 1:
            raise RuntimeError("already split")
        R_old = self.R
        self._build([p, 1.0 - p])
        self.R = np.empty((self.A, self.inst.m))
        for c, w in enumerate(self.weights):
            self.R[self.cid == c] = (
                w * R_old
            )  # arcs of one clone are laid out in the old arc order
        self.Q = np.zeros((self.A, self.inst.m))
        self.floor_q = False  # the clones carry float tables
        self._undamped_next = True

    def beliefs(self) -> np.ndarray:
        S = self.theta_msg.copy()
        np.add.at(S, self.dst, self.R)
        return S

    def q_seen(self) -> np.ndarray:
        """the Q message as the factors see it (truncated on integer tables)."""
        return np.trunc(self.Q) if self.floor_q else self.Q

    def assignment(self) -> np.ndarray:
        return self.beliefs().argmin(axis=1)

    def step(self) -> None:
        S = self.beliefs()
        # propflow carries the unary tie-break preferences as factors, so they are absent from the
        # very first Q messages (nothing has been sent yet) and present from the second step on
        Sq = S if self.t > 0 else S - self.theta_msg
        cav = Sq[self.src] - self.R[self.rev]  # exclude the recipient factor's own R
        if self.rule == "cav":
            Qc = cav
        elif self.rule == "belief":
            Qc = Sq[self.src]
        elif self.rule == "2cav":
            Qc = 2.0 * cav
        else:  # cav+belief
            Qc = cav + Sq[self.src]
        if self.norm_every is None:
            Qc = Qc - Qc.min(
                axis=1, keepdims=True
            )  # per-message constant, no effect on any argmin
        if self.lam > 0.0 and not self._undamped_next:
            Q = self.lam * self.Q + (1.0 - self.lam) * Qc
        else:
            Q = Qc
        self._undamped_next = False
        self.Q = Q
        self.R = (self.Ct + self.q_seen()[:, :, None]).min(axis=1)  # R[a][x_dst]
        if self.norm_every is None:
            self.R -= self.R.min(axis=1, keepdims=True)
        elif (
            self.t % self.norm_every == 0
        ):  # propflow's cycle event: inbox R and last sent Q
            self.R -= self.R.min(axis=1, keepdims=True)
            self.Q -= self.Q.min(axis=1, keepdims=True)
            self.theta_msg -= self.theta_msg.min(axis=1, keepdims=True)
        self.t += 1

    # -- commitment (row forwarding) -----------------------------------------------------------
    def selected_rows(self) -> np.ndarray:
        """for every arc and receiver value v, the sender row attaining min_u [C(u, v) + Q(u)]."""
        return (self.Ct + self.q_seen()[:, :, None]).argmin(axis=1)  # (A, m)

    def commit_mask(self) -> np.ndarray:
        """arc is committed iff the same sender row is selected for every receiver value
        (the factor then forwards one row of its table: a decision, not a magnitude)."""
        arg = self.selected_rows()
        return (arg == arg[:, :1]).all(axis=1)

    def saturation(self) -> float:
        return float(self.commit_mask().mean())

    def commit_margin(self) -> np.ndarray:
        """signed margin of the sender's preferred row u0 = argmin Q: the smallest amount by which
        row u0 beats every other row over all receiver values (>= 0 iff committed at u0)."""
        Qs = self.q_seen()
        Z = self.Ct + Qs[:, :, None]  # (A, m, m) [u, v]
        u0 = Qs.argmin(axis=1)
        best = Z[np.arange(self.A), u0, :]  # (A, m)
        Zm = Z.copy()
        Zm[np.arange(self.A), u0, :] = np.inf
        return (Zm.min(axis=1) - best).min(axis=1)


def fg_diameter(inst: Inst) -> int:
    """diameter of the propflow factor graph (variables, binary factors and the unary tie-break
    factors as leaves): the period of propflow's normalisation cycle events."""
    import networkx as nx

    G = nx.Graph()
    for k, (i, j) in enumerate(inst.edges):
        G.add_edge(("v", i), ("f", k))
        G.add_edge(("v", j), ("f", k))
    for i in range(inst.n):
        G.add_edge(("v", i), ("u", i))
    return int(nx.diameter(G))


def run_record(
    engine: FastEngine,
    iters: int,
    split_at: int | None = None,
    split_p: float = 0.5,
    record_assign: bool = True,
    record_dq: bool = False,
) -> dict:
    """step the engine; return per-iteration cost, commitment fraction and the number of variables
    whose assignment changed (and the assignments themselves; optionally the largest change of any
    Q message between consecutive iterations)."""
    inst = engine.inst
    costs = np.empty(iters)
    sats = np.empty(iters)
    changes = np.empty(iters, dtype=int)
    dq = np.empty(iters) if record_dq else None
    assigns = []
    prev = None
    for t in range(iters):
        if split_at is not None and t == split_at:
            engine.split_now(split_p)
        Q_prev = engine.Q if record_dq else None
        engine.step()
        if record_dq:
            dq[t] = (
                np.inf
                if Q_prev.shape != engine.Q.shape
                else float(np.abs(engine.Q - Q_prev).max())
            )
        x = engine.assignment()
        costs[t] = inst.cost(x)
        sats[t] = engine.saturation()
        changes[t] = inst.n if prev is None else int((x != prev).sum())
        prev = x
        if record_assign:
            assigns.append(x)
    out = dict(costs=costs, sats=sats, changes=changes)
    if record_assign:
        out["assigns"] = np.array(assigns)
    if record_dq:
        out["dq"] = dq
    return out


MIN_QUIET = 100


def freeze_time(changes: np.ndarray, min_quiet: int = MIN_QUIET) -> int:
    """first iteration after which the assignment never changes again, counted only when at least
    min_quiet unchanged iterations follow it (a run whose last change is 3 iterations before the end has
    not frozen, it was caught between changes); len(changes) if the run did not freeze."""
    T = len(changes)
    nz = np.flatnonzero(changes)
    f = int(nz[-1] + 1) if len(nz) else 0
    return f if f <= T - min_quiet else T


def strict_freeze(fr, T: int, min_quiet: int = MIN_QUIET) -> np.ndarray:
    """apply the freeze_time quiet-tail rule to freeze values that were stored before it existed."""
    fr = np.asarray(fr)
    return np.where(fr <= T - min_quiet, fr, T)


def detect_period(assigns, pmax=64, window=None) -> int:
    """minimal p such that a[t] == a[t-p] for all t in the tail window; 0 if none."""
    T = len(assigns)
    if window is None:
        window = max(20, T // 4)
    for p in range(1, pmax + 1):
        if T < window + p:
            break
        tail = assigns[-window:]
        prev = assigns[-window - p : -p]
        if (tail == prev).all():
            return p
    return 0


# ----------------------------------------------------------------------------------------------
# best response, local search, neighbourhood optimality
# ----------------------------------------------------------------------------------------------
def br_field(inst: Inst, x) -> np.ndarray:
    """F[i, v] = theta_i(v) + sum_{j~i} C_ij(v, x_j): the cost of value v at i against the
    neighbours' current values (the field a fully committed split graph presents to X_i)."""
    x = np.asarray(x)
    F = inst.theta.copy()
    ar = np.arange(inst.E)
    np.add.at(F, inst.ei, inst.Cs[ar, :, x[inst.ej]])
    np.add.at(F, inst.ej, inst.Cs[ar, x[inst.ei], :])
    return F


def cross_field(inst: Inst, y) -> np.ndarray:
    """field of one layer against the other layer of a pair: F[i, v] = theta_i(v) + sum_j C_ij(v, y_j)."""
    return br_field(inst, y)


def sync_best_response(inst: Inst, x0, iters=300) -> np.ndarray:
    """synchronous local selection: every variable simultaneously moves to its best value."""
    x = np.asarray(x0).copy()
    hist = []
    for _ in range(iters):
        x = br_field(inst, x).argmin(axis=1)
        hist.append(x.copy())
    return np.array(hist)


def improving_single_moves(inst: Inst, x) -> int:
    """number of variables that can lower the cost by changing their own value alone."""
    x = np.asarray(x)
    F = br_field(inst, x)
    cur = F[np.arange(inst.n), x]
    return int((F.min(axis=1) < cur - EPS).sum())


def is_pair_local_min(inst: Inst, x, y) -> tuple[bool, int]:
    """(x, y) is a layer-wise local minimum of cost_2 iff x is a best response to y and y to x."""
    x = np.asarray(x)
    y = np.asarray(y)
    Fx = cross_field(inst, y)
    Fy = cross_field(inst, x)
    vx = int((Fx.min(axis=1) < Fx[np.arange(inst.n), x] - EPS).sum())
    vy = int((Fy.min(axis=1) < Fy[np.arange(inst.n), y] - EPS).sum())
    return vx + vy == 0, vx + vy


def greedy_1opt(inst: Inst, x0, rng=None, max_sweeps=10_000) -> tuple[np.ndarray, int]:
    """sequential best-improvement local search over single-variable moves (random variable order
    per sweep) until no variable can improve; returns the local minimum and the number of moves."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x0).copy()
    adj = inst.adjacency()
    # per-variable incident tables oriented as C[x_i, x_j]
    inc = [[] for _ in range(inst.n)]
    for k, (i, j) in enumerate(inst.edges):
        inc[i].append((j, inst.C[k]))
        inc[j].append((i, inst.C[k].T))
    moves = 0
    for _ in range(max_sweeps):
        improved = False
        for i in rng.permutation(inst.n):
            f = inst.theta[i].copy()
            for j, tab in inc[i]:
                f += tab[:, x[j]]
            v = int(f.argmin())
            if f[v] < f[x[i]] - EPS:
                x[i] = v
                moves += 1
                improved = True
        if not improved:
            break
    return x, moves


def _dF(inst: Inst, x) -> np.ndarray:
    F = br_field(inst, x)
    return F - F[np.arange(inst.n), x][:, None]


def improving_edge_moves(inst: Inst, x) -> int:
    """number of edges (i, j) admitting a cost-lowering move that changes BOTH endpoints
    (the size-2 connected moves of the split graph's Weiss-Freeman neighbourhood)."""
    x = np.asarray(x)
    dF = _dF(inst, x)
    ar = np.arange(inst.E)
    Cxj = inst.Cs[ar, :, x[inst.ej]]  # C[v_i, x_j]  (E, m)
    Cxi = inst.Cs[ar, x[inst.ei], :]  # C[x_i, v_j]  (E, m)
    Cxx = inst.Cs[ar, x[inst.ei], x[inst.ej]]  # (E,)
    # delta = dF_i + dF_j - [counted change on edge ij] + [true change on edge ij]
    counted = (Cxj - Cxx[:, None])[:, :, None] + (Cxi - Cxx[:, None])[:, None, :]
    true = inst.Cs - Cxx[:, None, None]
    delta = (
        dF[inst.ei][:, :, None] + dF[inst.ej][:, None, :] - counted + true
    )  # (E, m, m)
    delta[ar, x[inst.ei], :] = np.inf  # v_i must change
    delta[ar, :, x[inst.ej]] = np.inf  # v_j must change
    return int((delta.min(axis=(1, 2)) < -EPS).sum())


def improving_path_moves(inst: Inst, x, n_samples=2000, rng=None) -> tuple[int, int]:
    """sample induced paths i-j-k (i~j, j~k, i not~ k) and count those admitting a cost-lowering
    move that changes all three variables (size-3 tree moves: inside the unsplit graph's
    Weiss-Freeman neighbourhood, outside the split graph's). returns (improving, sampled)."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x)
    adj = inst.adjacency()
    adjset = [set(a) for a in adj]
    tab = {}
    for k, (i, j) in enumerate(inst.edges):
        tab[(i, j)] = inst.C[k]
        tab[(j, i)] = inst.C[k].T
    dF = _dF(inst, x)
    m = inst.m
    mids = [j for j in range(inst.n) if len(adj[j]) >= 2]
    if not mids:
        return 0, 0
    improving = sampled = 0
    for _ in range(n_samples):
        j = int(rng.choice(mids))
        i, k = rng.choice(adj[j], size=2, replace=False)
        i, k = int(i), int(k)
        if k in adjset[i]:
            continue  # triangle, not a tree move
        sampled += 1
        Cij, Cjk = tab[(i, j)], tab[(j, k)]

        def corr(C, a, b):
            # counted-minus-true change of one edge for a joint move (v_a, v_b)
            return (
                (C[:, x[b]] - C[x[a], x[b]])[:, None]
                + (C[x[a], :] - C[x[a], x[b]])[None, :]
                - (C - C[x[a], x[b]])
            )

        delta = dF[i][:, None, None] + dF[j][None, :, None] + dF[k][None, None, :]
        delta = delta - corr(Cij, i, j)[:, :, None] - corr(Cjk, j, k)[None, :, :]
        delta[x[i], :, :] = np.inf
        delta[:, x[j], :] = np.inf
        delta[:, :, x[k]] = np.inf
        if delta.min() < -EPS:
            improving += 1
    return improving, sampled


def edge_flip_classes(inst: Inst, x, y) -> np.ndarray:
    """for every edge, how many of its endpoints differ between the two layers x and y (0, 1, 2)."""
    x = np.asarray(x)
    y = np.asarray(y)
    flip = x != y
    return flip[inst.ei].astype(int) + flip[inst.ej].astype(int)
